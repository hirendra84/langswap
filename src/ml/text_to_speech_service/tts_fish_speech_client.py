import sys
import os
from pathlib import Path
import logging
from typing import Optional, List # Added List for type hinting

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm # Changed from tqdm.auto to just tqdm for consistency

# Adjust path to fish-speech if it's not directly in PYTHONPATH
# This assumes fish-speech directory is three levels up and then into 'fish-speech'
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../fish-speech"))

# Direct imports from fish_speech submodules
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_decoder_model
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from fish_speech.utils.file import AUDIO_EXTENSIONS

from src.file_repository import FileRepository # Already present
# Assuming TTSClient and TranslatedTextedSegment are correctly pathed for import
from .tts_client import TTSClient 
from src.pipeline_models.models import TranslatedTextedSegment
# from src.utils.ml_processing.lang2code_mapper import map_language_to_code # Keep if used elsewhere

logger = logging.getLogger(__name__)
os.environ["EINX_FILTER_TRACEBACK"] = "false"


class FishSpeechClient(TTSClient):
    def __init__(
        self,
        file_repository: FileRepository,
        llama_checkpoint_path: str | Path = "./models_weights/fish-speech-1.5",
        decoder_checkpoint_path: str | Path = "./models_weights/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        decoder_config_name: str = "firefly_gan_vq", # Added, crucial for load_decoder_model
        device: str = "cuda",
        half_precision: bool = True,
        compile_models: bool = False,
    ):
        super().__init__() # Call to base TTSClient
        self._file_repository = file_repository 
        self.device = device
        self.sample_rate = 44100
        
        self.llama_checkpoint_path = Path(os.path.abspath(llama_checkpoint_path))
        self.decoder_checkpoint_path = Path(os.path.abspath(decoder_checkpoint_path))
        self.decoder_config_name = decoder_config_name
        self.half_precision = half_precision
        self.compile_models = compile_models
        
        self.precision: Optional[torch.dtype] = None
        self.llama_queue = None
        self.decoder_model = None
        self.tts_inference_engine: Optional[TTSInferenceEngine] = None
        
        self.load_models()

    def load_models(self):
        logger.info("Initializing FishSpeechClient models...")
        
        if not torch.cuda.is_available() and self.device == "cuda":
            logger.warning("CUDA not available, falling back to CPU.")
            self.device = "cpu"
        elif torch.backends.mps.is_available() and self.device == "cuda": # Apple Silicon GPU
            logger.info("MPS is available, using MPS for CUDA request.")
            self.device = "mps"
        
        self.precision = torch.half if self.half_precision and self.device != "cpu" else torch.bfloat16
        if self.device == "cpu":
            self.precision = torch.bfloat16 # float16 not well supported on CPU
            logger.info("Running on CPU, using bfloat16 precision.")
        
        logger.info(f"Using device: {self.device}, precision: {self.precision}")

        if not self.llama_checkpoint_path.exists():
            raise FileNotFoundError(f"LLaMA checkpoint path not found: {self.llama_checkpoint_path}")
        if not self.decoder_checkpoint_path.exists():
            raise FileNotFoundError(f"Decoder checkpoint path not found: {self.decoder_checkpoint_path}")

        try:
            logger.info(f"Loading LLaMA model from: {self.llama_checkpoint_path}")
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=self.llama_checkpoint_path,
                device=self.device,
                precision=self.precision,
                compile=self.compile_models,
            )

            logger.info(f"Loading VQ-GAN decoder model from: {self.decoder_checkpoint_path}")
            self.decoder_model = load_decoder_model(
                config_name=self.decoder_config_name,
                checkpoint_path=self.decoder_checkpoint_path,
                device=self.device,
            )

            logger.info("Initializing TTSInferenceEngine...")
            self.tts_inference_engine = TTSInferenceEngine(
                llama_queue=self.llama_queue,
                decoder_model=self.decoder_model,
                precision=self.precision,
                compile=self.compile_models,
            )
            logger.info("FishSpeech models and inference engine loaded. Performing warmup...")
            self._warmup()
            logger.info("Warmup complete.")

        except FileNotFoundError as e: # Should be caught by checks above, but good practice
            logger.error(f"Error loading FishSpeech models (FileNotFound): {e}.")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred while loading FishSpeech models: {e}")
            raise

    def _warmup(self):
        if not self.tts_inference_engine:
            logger.error("TTS Inference Engine not initialized. Cannot perform warmup.")
            return
        try:
            logger.info("Performing warmup with a test inference...")
            warmup_request = ServeTTSRequest(text="Hello.", references=[], seed=42)
            _ = list(self.tts_inference_engine.inference(warmup_request)) # Consume generator
        except Exception as e:
            logger.error(f"Error during warmup: {e}")

    def __enter__(self):
        if not self.tts_inference_engine:
            self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("FishSpeechClient exiting. Releasing resources...")
        if hasattr(self, 'llama_queue') and self.llama_queue and hasattr(self.llama_queue, 'put'):
            try:
                self.llama_queue.put(None) 
            except Exception as e:
                logger.warning(f"Error signaling LLaMA queue to stop: {e}")
        
        if hasattr(self, 'decoder_model'):
            del self.decoder_model
            self.decoder_model = None
        if hasattr(self, 'tts_inference_engine'):
            del self.tts_inference_engine
            self.tts_inference_engine = None
        
        if self.device in ["cuda", "mps"] and torch.cuda.is_available(): # Check MPS specific cleanup if any
            torch.cuda.empty_cache()
        logger.info("FishSpeechClient resources released.")

    def generate_audio(
        self,
        text: str, 
        source_audio_file: str, 
        source_text: str, 
        save_path: str,
        language: Optional[str] = None,
        output_format: str = "wav", # For ServeTTSRequest, actual save format is WAV via soundfile
        chunk_length: int = 200,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        max_new_tokens: int = 1024,
    ) -> bool:
        if not self.tts_inference_engine:
            logger.error("FishSpeech model is not loaded. Cannot generate audio.")
            return False

        if not source_text or not source_text.strip():
            logger.error(f"Source audio text is missing or empty for source audio {source_audio_file}. FishSpeech requires this for voice cloning.")
            return False
        
        logger.info(f"Generating audio for text: '{text[:50]}...' with source audio: {source_audio_file}")
        if language:
            logger.debug(f"Language hint: {language} (FishSpeech infers from model/text)")

        source_audio_path_obj = Path(source_audio_file)
        if not source_audio_path_obj.exists():
            logger.error(f"Source audio path not found: {source_audio_path_obj}")
            return False
        
        if source_audio_path_obj.suffix.lower() not in AUDIO_EXTENSIONS:
            logger.warning(
                f"Source audio file {source_audio_path_obj} has extension '{source_audio_path_obj.suffix}', "
                f"which may not be ideal. Supported audio types are typically: {AUDIO_EXTENSIONS}"
            )

        try:
            with open(source_audio_path_obj, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            logger.error(f"Error reading source audio file {source_audio_path_obj}: {e}")
            return False

        reference_audio = ServeReferenceAudio(audio=audio_bytes, text=source_text)
        request = ServeTTSRequest(
            text=text,
            references=[reference_audio],
            format=output_format, 
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        
        final_audio_data = None
        audio_sample_rate = None

        try:
            for result in self.tts_inference_engine.inference(request):
                if result.code == "error":
                    logger.error(f"TTS Inference error from engine: {result.error}")
                    return False
                elif result.code == "final" and result.audio is not None:
                    audio_sample_rate, final_audio_data = result.audio
                    logger.info(f"Inference successful. Audio duration: {len(final_audio_data) / audio_sample_rate:.2f}s")
                    break 
            
            if final_audio_data is None or audio_sample_rate is None:
                logger.error("No final audio data received from inference engine.")
                return False

            sf.write(str(save_path), final_audio_data, audio_sample_rate)
            logger.info(f"Generated audio saved to {save_path}")
            return True

        except Exception as e:
            logger.exception(f"An unexpected error occurred during TTS inference or saving: {e}")
            return False

    def tts_pipeline(self, video_translation, temp_folder: str, language: str = "en") -> List[TranslatedTextedSegment]:
        if not self.tts_inference_engine:
            logger.error("FishSpeech model is not loaded. Cannot run TTS pipeline.")
            return video_translation.translated_texts # Return the original list

        Path(temp_folder).mkdir(parents=True, exist_ok=True)
        
        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="FishSpeech TTS Pipeline",
                leave=True,
            )
        ):  
            file_path = Path(temp_folder) / f"{segment.start}_{segment.end}.wav"
            
            if not file_path.exists():
                if not segment.source_file:
                    logger.warning(f"Segment {idx} (time {segment.start}-{segment.end}) is missing 'source_file'. Skipping.")
                    video_translation.translated_texts[idx].generated_file = None
                    continue
                
                # FishSpeech requires the transcript of the source audio.
                # 'segment.text' is assumed to be the original transcript for voice cloning.
                source_text_for_cloning = getattr(segment, 'text', None) 
                if not source_text_for_cloning: # Fallback if 'text' attribute is not present
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
                    save_path=file_path,
                    language=language
                )
                
                if success:
                    video_translation.translated_texts[idx].generated_file = str(file_path)
                else:
                    video_translation.translated_texts[idx].generated_file = None
            else:
                 video_translation.translated_texts[idx].generated_file = str(file_path)
                 logger.info(f"Audio file {file_path} already exists for segment {idx}. Skipping generation.")
        return video_translation.translated_texts # Ensure it returns the list of segments
