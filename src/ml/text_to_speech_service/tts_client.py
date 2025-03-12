from abc import ABC
from src.pipeline_models.models import TranslatedTextedSegment


class TTSClient(ABC):

    def __init__(self):
        ...

    def clone_voice(self, voice_path: str, voice_descr: str = '', voice_name = '' ):
        ...

    def generate_audio(self, text: str, source_audio_file: str, save_path: str):
        ...
        
    def tts_pipelene(self, data: list[TranslatedTextedSegment], output_folder: str) -> list[TranslatedTextedSegment]:
        ...
        
    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...
