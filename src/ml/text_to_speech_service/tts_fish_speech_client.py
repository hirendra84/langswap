import sys
import os
from pathlib import Path
import logging
from typing import Optional, List
import uuid # Added for unique temporary file naming

from more_itertools import run_length # Added List for type hinting

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm # Changed from tqdm.auto to just tqdm for consistency

# Adjust path to fish-speech if it's not directly in PYTHONPATH
# This assumes fish-speech directory is three levels up and then into 'fish-speech'
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../fish-speech"))

# Direct imports from fish_speech submodules
from fish_speech.lib import Pipeline # New import
from fish_speech.utils.schema import ServeReferenceAudio # Kept for reference structure, though Pipeline might abstract it
from fish_speech.utils.file import AUDIO_EXTENSIONS

from src.file_repository import FileRepository # Already present
# Assuming TTSClient and TranslatedTextedSegment are correctly pathed for import
from .tts_client import TTSClient 
from src.pipeline_models.models import TranslatedTextedSegment
# from src.utils.ml_processing.lang2code_mapper import map_language_to_code # Keep if used elsewhere

logger = logging.getLogger(__name__)
os.environ["EINX_FILTER_TRACEBACK"] = "false"


class FishSpeechClient(TTSClient):
    ACCENT_REMOVAL_PREFIXES = {
        "en": "Let me say it without an accent: ",
        "zh": "让我无口音地说: ",  # Chinese (Simplified)
        "ja": "アクセントなしで言います: ",  # Japanese
        "de": "Lass es mich ohne Akzent sagen: ",  # German
        "fr": "Laissez-moi le dire sans accent: ",  # French
        "es": "Déjame decirlo sin acento: ",  # Spanish
        "ko": "억양 없이 말할게요: ",  # Korean
        "ar": "دعني أقولها بدون لهجة: ",  # Arabic
        "ru": "Скажу без акцента: ",  # Russian
        "nl": "Laat het me zonder accent zeggen: ",  # Dutch
        "it": "Lascia che lo dica senza accento: ",  # Italian
        "pl": "Pozwól, że powiem to bez akcentu: ",  # Polish
        "pt": "Deixe-me dizer isso sem sotaque: ",  # Portuguese
    }
    DEFAULT_ACCENT_REMOVAL_PREFIX = ACCENT_REMOVAL_PREFIXES["en"]

    def __init__(
        self,
        file_repository: FileRepository,
        llama_checkpoint_path: str | Path = "./models_weights/fish-speech-1.5",
        decoder_checkpoint_path: str | Path = "./models_weights/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        device: str = "cuda",
        compile_models: bool = True, # Renamed from 'compile' for clarity with existing style
    ):
        super().__init__() # Call to base TTSClient
        self._file_repository = file_repository 
        self.device = device
        self.sample_rate = 44100 # Default sample rate, confirmed by user example for Pipeline output
        
        self.llama_checkpoint_path = Path(os.path.abspath(llama_checkpoint_path))
        self.decoder_checkpoint_path = Path(os.path.abspath(decoder_checkpoint_path))
        self.compile_models = compile_models # Changed from compile to compile_models
        
        self.model: Optional[Pipeline] = None
        
        self.load_models()

    def load_models(self):
        logger.info("Initializing FishSpeechClient with fish_speech.lib.Pipeline...")

        if not self.llama_checkpoint_path.exists():
            raise FileNotFoundError(f"LLaMA checkpoint path not found: {self.llama_checkpoint_path}")
        if not self.decoder_checkpoint_path.exists():
            raise FileNotFoundError(f"Decoder checkpoint path not found: {self.decoder_checkpoint_path}")

        logger.info(f"Loading Fish Speech Pipeline...")
        self.model = Pipeline(
            llama_path=str(self.llama_checkpoint_path),
            vqgan_path=str(self.decoder_checkpoint_path),
            device=self.device,
            compile=self.compile_models,
        )
        logger.info("Fish Speech Pipeline loaded successfully.")

    def __enter__(self):
        if not self.model: 
            self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("FishSpeechClient exiting. Releasing resources...")
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        
        if self.device in ["cuda", "mps"] and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("FishSpeechClient resources released.")

    def generate_audio(
        self,
        text: str, 
        source_audio_file: str, 
        source_text: str, 
        save_path: str,
        language: Optional[str] = "en", # Defaulting to 'en' for prefix lookup
        chunk_length: int = 200,
        top_p: float = 0.7,
        repetition_penalty: float = 1.5, 
        temperature: float = 0.7,
        seed: Optional[int] = None, 
        max_new_tokens: int = 1024, 
    ) -> bool:
        if not self.model:
            logger.error("Model not loaded. Call load_models() first or use a context manager.")
            return False

        if not source_text or not source_text.strip():
            logger.error(f"Source audio text is missing or empty for source audio {source_audio_file}. FishSpeech requires this for voice cloning.")
            return False

        source_audio_path_obj = Path(source_audio_file)
        if not source_audio_path_obj.exists():
            logger.error(f"Source audio file not found: {source_audio_file}")
            return False
        
        save_path_obj = Path(save_path)
        temp_intermediate_audio_path = save_path_obj.parent / f"intermediate_{uuid.uuid4()}.wav"

        try:
            # === Stage 1: Generate intermediate audio with accent removal prompt ===
            logger.info(f"Stage 1: Making initial reference (ref1) from audio: {source_audio_file} and text: '{source_text[:30]}...'")
            ref1 = self.model.make_reference(str(source_audio_path_obj), source_text)

            # Select language-specific prefix, default to English if not found or language is None
            current_language_key = language.lower() if language else "en"
            accent_removal_prefix = self.ACCENT_REMOVAL_PREFIXES.get(current_language_key, self.DEFAULT_ACCENT_REMOVAL_PREFIX)
            
            text_for_stage1_generation = accent_removal_prefix + text
            
            logger.info(f"Stage 1: Generating intermediate audio for language '{current_language_key}' with prompt: '{text_for_stage1_generation[:50]}...'")
            
            intermediate_audio_data = self.model.generate(
                text=text_for_stage1_generation,
                references=ref1,
                chunk_length=chunk_length, 
                top_p=top_p,
                repetition_penalty=repetition_penalty, 
                temperature=0.2,
                seed=seed,
                max_new_tokens=max_new_tokens,
            )

            if intermediate_audio_data is None:
                logger.error("Stage 1: No audio data received from inference engine for intermediate audio.")
                return False

            sf.write(str(temp_intermediate_audio_path), intermediate_audio_data, self.sample_rate)
            logger.info(f"Stage 1: Intermediate audio saved to {temp_intermediate_audio_path}")

            # === Stage 2: Generate final audio using original and intermediate references ===
            logger.info(f"Stage 2: Making second reference (ref2) from intermediate audio: {temp_intermediate_audio_path} and original text: '{text[:30]}...'")
            ref2 = self.model.make_reference(str(temp_intermediate_audio_path), text) 
            
            logger.info("Stage 2: Generating final waveform using ref1 and ref2...")
            final_audio_data = self.model.generate(
                text=text, 
                references=[ref1, ref2], 
                chunk_length=chunk_length, 
                top_p=top_p,
                repetition_penalty=repetition_penalty, 
                temperature=0.5,
                seed=seed, 
                max_new_tokens=max_new_tokens,
            )
            
            if final_audio_data is None: 
                logger.error("Stage 2: No audio data received from inference engine for final audio.")
                # Clean up before returning False, as 'finally' block won't be reached if we return here.
                # However, the finally block will execute on function exit regardless of how it exits (return, exception).
                return False

            sf.write(str(save_path_obj), final_audio_data, self.sample_rate)
            logger.info(f"Stage 2: Final generated audio saved to {save_path_obj}")
            return True
        
        except Exception as e:
            logger.error(f"Error during two-stage TTS generation: {e}", exc_info=True)
            return False
        finally:
            # Clean up the temporary intermediate audio file
            if temp_intermediate_audio_path.exists():
                try:
                    os.remove(temp_intermediate_audio_path)
                    logger.info(f"Cleaned up temporary file: {temp_intermediate_audio_path}")
                except OSError as e_os:
                    logger.error(f"Error deleting temporary file {temp_intermediate_audio_path}: {e_os}")

    def tts_pipeline(self, video_translation, temp_folder: str, language: str = "en") -> List[TranslatedTextedSegment]:
        Path(temp_folder).mkdir(parents=True, exist_ok=True)
        
        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="FishSpeech TTS Pipeline",
                leave=True,
            )
        ):  
            file_path = Path(temp_folder) / f"{segment.start}_{segment.end}.wav"
            segment.speaker #
            
            if not file_path.exists():
                if not segment.source_file:
                    logger.warning(f"Segment {idx} (time {segment.start}-{segment.end}) is missing 'source_file'. Skipping.")
                    video_translation.translated_texts[idx].generated_file = None
                    continue
                
                source_text_for_cloning = getattr(segment, 'text', None) 
                if not source_text_for_cloning: 
                     source_text_for_cloning = getattr(segment, 'source_text', None)

                if not source_text_for_cloning:
                    logger.warning(
                        f"Segment {idx} (time {segment.start}-{segment.end}) is missing 'text' or 'source_text' (original transcript). "
                        f"Skipping audio generation as FishSpeech requires it for cloning."
                    )
                    video_translation.translated_texts[idx].generated_file = None
                    continue

                logger.info(f"Processing segment {idx}: text='{segment.translation[:30]}...', source_audio='{segment.source_file}', source_text='{source_text_for_cloning[:30]}...'")
                
                success = self.generate_audio(
                    text=segment.translation,
                    source_audio_file=segment.source_file,
                    source_text=source_text_for_cloning,
                    save_path=str(file_path), 
                    language=language,
                )
                
                if success:
                    video_translation.translated_texts[idx].generated_file = str(file_path)
                else:
                    video_translation.translated_texts[idx].generated_file = None
            else:
                 video_translation.translated_texts[idx].generated_file = str(file_path)
                 logger.info(f"Audio file {file_path} already exists for segment {idx}. Skipping generation.")
        return video_translation
