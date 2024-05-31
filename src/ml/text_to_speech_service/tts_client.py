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

    @staticmethod
    def get_audio_length(audio_path):
        audio, sr = torchaudio.load(audio_path)
        return audio.shape[1] / sr
    
    def generate_audio(self, data: list[TranslatedTextedSegment], source_audio, df) -> list[tuple[str, str]]:
        """
        Creates a VAD filtered audio file, generates the audio samples based on this voice.
        """
        # dum a file - with resample if needed, caution - long file
        # TODO: rewrite to one file pipeline

        temp_folder = os.path.join(self._file_repository._directory, "generated_audio")
        
        os.makedirs(temp_folder, exist_ok=True)

        for idx, segment in enumerate(data):
            save_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")
            self.tts.tts_to_file(text=segment.translation, file_path=save_path, speaker_wav=source_audio.file_path, language=self.lang)

            df.loc[idx, "generated_path"] = save_path
        return df
    
    def style_audio(self, df):
        """
        : audio_vad_path: audio path cleaned from the background noise (can be VAD filtered speech)
        """

        temp_folder = os.path.join(self._file_repository._directory, "styled_generated_audio")
        os.makedirs(temp_folder, exist_ok=True)

        for row in tqdm(df.iterrows()):
            idx = row[0]
            
            save_path = os.path.join(temp_folder, f"{row[1].start}_{row[1].end}.wav")
            df.loc[idx, 'styled_generated_path'] = save_path

            self.style_tts.voice_conversion_to_file(source_wav=row[1].generated_path, target_wav=row[1].source_path, file_path=save_path)
        return df
