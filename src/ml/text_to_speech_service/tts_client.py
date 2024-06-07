from abc import ABC

from tqdm.auto import tqdm

from src.pipeline_models.models import TranslatedTextedSegment
from TTS.api import TTS
import os
from src.file_repository import FileRepository

import torchaudio


class TTSClient(ABC):

    def __init__(self):
        ...

    def clone_voice(self, voice_path: str, voice_descr: str = '', voice_name = '' ):
        ...

    def generate_audio(self, data: list[TranslatedTextedSegment], output_folder: str, source_audio_path: str, lang: str) \
            -> list[tuple[str, str]]:
        ...

    def style_audio(self, output_directory: str, df):
        ...

    def generate_style_sample(self,  text: str, source_audio_path: str, save_path: str, style=True):
        ...

class XTTSClient:
    def __init__(self,
                file_repository: FileRepository,
                tts_model_id="tts_models/multilingual/multi-dataset/xtts_v2",
                style_model_id="voice_conversion_models/multilingual/vctk/freevc24",
                language="en"):
        self.tts = TTS(tts_model_id)
        self.style_tts = TTS(model_name=style_model_id, progress_bar=False)

        self._file_repository = file_repository

        self.lang = language
    
    def generate_style_sample(self, text: str, source_audio_path: str, save_path: str, style=False):
        """
        Generates and styles audio, saves according to the path.
        : param style: whether voice conversion should be applied
        """
        # text = text.replace('"', "")
        self.tts.tts_to_file(text=text, file_path=save_path, speaker_wav=source_audio_path, language=self.lang)

        if style:
            self.style_tts.voice_conversion_to_file(source_wav=save_path, target_wav=source_audio_path, file_path=save_path)
