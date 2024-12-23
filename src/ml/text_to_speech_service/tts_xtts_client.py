import sys

sys.path.append("/app/coqui")

import soundfile as sf
from tqdm.auto import tqdm

from coqui.TTS.api import TTS
from src.file_repository import FileRepository
import os
from src.utils.ml_processing.lang2code_mapper import map_language_to_code
from .utils import add_pauses, merge_speaker_files


class XTTSClient:
    def __init__(
        self,
        file_repository: FileRepository,
        tts_model_path="./models_weights/xtts_model/tts_models/multilingual/multi-dataset/xtts_v2",
        device="cuda",
    ):
        self.device = device

        self.tts_model_path = os.path.abspath(tts_model_path)
        self.config_path = os.path.join(tts_model_path, "config.json")

        self._file_repository = file_repository

        self.model = None

    def load_models(self):
        gpu = True if self.device == "cuda" else False
        self.model = TTS(model_path=self.tts_model_path, config_path=self.config_path, gpu=gpu)

    def generate_audio(
        self, text: str, source_audio_path: str, save_path: str, language: str
    ):
        """
        Generates audio without voice conversion.
        """
        self.model.tts_to_file(
            text=text,
            file_path=save_path,
            speaker_wav=source_audio_path,
            language=language,
            enable_text_splitting=False,
            repetition_penalty=2.0,
        )

    def tts_pipeline(self, video_translation, temp_folder, language="en"):
        language = map_language_to_code(language, "whisper")

        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="Voice generation pipeline.",
                leave=True,
            )
        ):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")

            if not os.path.exists(file_path):
                if self.model is None:
                    self.load_models()
                if segment.end - segment.start < 4:
                    source_file_updated = segment.source_file.replace(".wav", "_extended.wav")

                    merge_speaker_files(video_translation,
                                    segment.speaker,
                                    idx,
                                    source_file_updated
                                    )
                else:
                    add_pauses(segment.source_file)


                self.generate_audio(
                    segment.translation, segment.source_file, file_path, language
                )
            video_translation.translated_texts[idx].generated_file = file_path
        return video_translation
