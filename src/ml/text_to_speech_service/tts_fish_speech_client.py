import sys
import os
from pathlib import Path
import logging
from typing import Optional, List
import uuid # Added for unique temporary file naming

from more_itertools import run_length # Added List for type hinting
import whisperx # Added for audio alignment and prefix removal

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
from src.utils.ml_processing.lang2code_mapper import map_language_to_code # Added for whisperx lang code

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
        compile_models: bool = True,
        remove_accent: bool = False,  # Add accent removal toggle
    ):
        super().__init__()
        self._file_repository = file_repository
        self.device = device
        self.sample_rate = 44100
        self.remove_accent = remove_accent  # Store the accent removal preference
        
        self.llama_checkpoint_path = Path(os.path.abspath(llama_checkpoint_path))
        self.decoder_checkpoint_path = Path(os.path.abspath(decoder_checkpoint_path))
        self.compile_models = compile_models
        
        self.model: Optional[Pipeline] = None
        self._whisperx_align_models = {}
        self.load_models()

    def load_models(self):
        if not self.llama_checkpoint_path.exists() or not self.decoder_checkpoint_path.exists():
            raise FileNotFoundError("One or both checkpoint paths not found.")
        self.model = Pipeline(
            llama_path=str(self.llama_checkpoint_path),
            vqgan_path=str(self.decoder_checkpoint_path),
            device=self.device,
            compile=self.compile_models,
        )
        logger.info("Fish Speech Pipeline loaded.")

    def generate_and_trim_audio(
        self, 
        text: str, 
        references,  # Changed from source_audio_file/source_text to references
        save_path: str, 
        language: Optional[str] = "en",
        chunk_length: int = 200, 
        top_p: float = 0.7,
        repetition_penalty: float = 1.2, 
        temperature: float = 0.5,
        seed: Optional[int] = None, 
        max_new_tokens: int = 1024,
    ) -> bool:
        if not self.model:
            return False

        # Generate audio using the provided reference
        audio = self.model.generate(
            text=text,
            references=references,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        if audio is None:
            return False

        # Save the generated audio to the specified path
        save_path_p = Path(save_path).resolve()
        save_path_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(save_path_p), audio, self.sample_rate)
        return True

    def trim_prefix_from_audio(self, audio, full_text, prefix, language_code):
        if language_code not in self._whisperx_align_models:
            align_model, meta = whisperx.load_align_model(language_code, self.device)
            self._whisperx_align_models[language_code] = (align_model, meta)
        else:
            align_model, meta = self._whisperx_align_models[language_code]

        segments_for_align = [{
            "text": full_text,
            "start": 0.0,
            "end": audio.shape[0] / self.sample_rate
        }]
        align_result = whisperx.align(segments_for_align, align_model, meta, audio, self.device, return_char_alignments=False)

        if align_result and align_result.get('segments'):
            words = align_result['segments'][0].get('words', [])
            prefix_word_count = len(prefix.strip().split())
            if len(words) > prefix_word_count and 'start' in words[prefix_word_count]:
                start_frame = int(words[prefix_word_count]['start'] * self.sample_rate)
                trimmed_audio = audio[start_frame:]
                return trimmed_audio
            else:
                logger.warning(f"WhisperX could not reliably find prefix end for '{prefix}'.")
                return None
        else:
            logger.warning("WhisperX alignment failed.")
            return None

    def generate_reference_for_speaker(self, speaker, segments, language):
        """
        Generate up to 4 trimmed references for a speaker from their segments, preferring longer ones.
        Returns a list of references.
        """
        # Sort segments by duration (prefer longer)
        valid_segments = [
            s for s in segments
            if getattr(s, 'source_file', None) and (getattr(s, 'text', None) or getattr(s, 'source_text', None))
        ]
        # Remove duplicates by (source_file, text/source_text)
        seen = set()
        unique_segments = []
        for s in sorted(valid_segments, key=lambda s: -(getattr(s, 'end', 0) - getattr(s, 'start', 0))):
            key = (getattr(s, 'source_file', None), getattr(s, 'text', None) or getattr(s, 'source_text', None))
            if key not in seen:
                seen.add(key)
                unique_segments.append(s)
            if len(unique_segments) >= 4:
                break

        references = []
        for s in unique_segments:
            source_audio_file = getattr(s, 'source_file', None)
            source_text = getattr(s, 'text', None) or getattr(s, 'source_text', None)
            ref = self._generate_single_reference(speaker, source_audio_file, source_text, language)
            if ref is not None:
                references.append(ref)
        return references

    def _generate_single_reference(self, speaker, source_audio_file, source_text, language):
        if self.remove_accent:
            # Current behavior: generate with prefix and trim
            current_lang_key = language.lower()
            whisper_lang_code = map_language_to_code(current_lang_key, system="whisper")
            prefix = self.ACCENT_REMOVAL_PREFIXES.get(whisper_lang_code, self.DEFAULT_ACCENT_REMOVAL_PREFIX)
            text_with_prefix = prefix + source_text

            # Generate audio with the accent removal prefix
            audio_with_prefix = self.model.generate(
                text=text_with_prefix,
                references=self.model.make_reference(str(Path(source_audio_file).resolve()), source_text),
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.5,
                seed=None,
                max_new_tokens=1024,
            )
            if audio_with_prefix is None:
                return None

            # Trim the prefix from the audio using WhisperX
            trimmed_audio = self.trim_prefix_from_audio(audio_with_prefix, text_with_prefix, prefix, whisper_lang_code)
            if trimmed_audio is None:
                return None

            # Save the trimmed audio as a temporary reference file
            temp_ref_file = Path(f"/tmp/{speaker}_{uuid.uuid4().hex}_reference.wav")
            sf.write(str(temp_ref_file), trimmed_audio, self.sample_rate)

            # Create a reference from the trimmed audio
            reference = self.model.make_reference(str(temp_ref_file), source_text)
            return reference
        else:
            # One-pass generation without accent removal
            # Generate audio directly from the source reference
            audio = self.model.generate(
                text=source_text,
                references=self.model.make_reference(str(Path(source_audio_file).resolve()), source_text),
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.5,
                seed=None,
                max_new_tokens=1024,
            )
            if audio is None:
                return None

            # Save the generated audio as a temporary reference file
            temp_ref_file = Path(f"/tmp/{speaker}_{uuid.uuid4().hex}_reference.wav")
            sf.write(str(temp_ref_file), audio, self.sample_rate)

            # Create a reference from the generated audio
            reference = self.model.make_reference(str(temp_ref_file), source_text)
            return reference

    def tts_pipeline(self, video_translation, temp_folder: str, language: str = "en") -> List[TranslatedTextedSegment]:
        Path(temp_folder).mkdir(parents=True, exist_ok=True)
        references_per_speaker = {}  # Store references for each speaker

        # Group segments by speaker
        from collections import defaultdict
        speaker_segments = defaultdict(list)
        for segment in video_translation.translated_texts:
            speaker = getattr(segment, 'speaker', None)
            if speaker:
                speaker_segments[speaker].append(segment)

        for idx, segment in enumerate(tqdm(video_translation.translated_texts, desc="FishSpeech TTS")):
            file_path = (Path(temp_folder) / f"{segment.start}_{segment.end}.wav").resolve()

            if file_path.exists():
                video_translation.translated_texts[idx].generated_file = str(file_path)
                continue

            source_audio = getattr(segment, 'source_file', None)
            source_txt_clone = getattr(segment, 'text', None) or getattr(segment, 'source_text', None)
            speaker = getattr(segment, 'speaker', None)  # Access speaker information

            if not source_audio or not source_txt_clone or not segment.translation or not speaker:
                video_translation.translated_texts[idx].generated_file = None
                continue

            # Create up to 4 references for each speaker
            if speaker not in references_per_speaker:
                references = self.generate_reference_for_speaker(
                    speaker, speaker_segments[speaker], language
                )
                references_per_speaker[speaker] = references
            else:
                references = references_per_speaker[speaker]

            if not references:
                video_translation.translated_texts[idx].generated_file = None
                continue

            success = self.generate_and_trim_audio(
                text=segment.translation,
                references=references,  # Use the list of references for this speaker
                save_path=str(file_path),
                language=language,
            )
            video_translation.translated_texts[idx].generated_file = str(file_path) if success else None
        return video_translation
