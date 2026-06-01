from abc import ABC


class TTSClient(ABC):

    def __init__(self):
        ...

    def clone_voice(self, voice_path: str, voice_descr: str = '', voice_name = '' ):
        ...

    def generate_audio(self, text: str, source_audio_file: str, save_path: str):
        ...

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...
