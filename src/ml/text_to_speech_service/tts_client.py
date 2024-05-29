from abc import ABC

from tqdm.auto import tqdm

from src.pipeline_models.models import TranslatedTextedSegment
from TTS.api import TTS
import os
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


class XTTSClient(TTSClient):
    def __init__(self):
        super().__init__()
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.style_tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False)

    @staticmethod
    def get_audio_length(audio_path):
        audio, sr = torchaudio.load(audio_path)
        return audio.shape[1] / sr
    
    def generate_audio(self, data: list[TranslatedTextedSegment], output_directory: str, source_audio_path: str, lang: str) \
            -> list[tuple[str, str]]:
        """
        Creates a VAD filtered audio file, generates the audio samples based on this voice.
        """
        # dum a file - with resample if needed, caution - long file
        # TODO: rewrite to one file pipeline

        file_name_paths = []
        for idx, segment in enumerate(data):
            file_name = f"{segment.start}_{segment.end}.wav"
            save_path = os.path.join(output_directory, file_name)
            self.tts.tts_to_file(text=segment.translation,
                                 file_path=save_path,
                                 speaker_wav=source_audio_path,
                                 language=lang)
            file_name_paths.append((file_name, save_path))
        return file_name_paths
    
    def style_audio(self, output_directory: str, df):
        """
        : audio_vad_path: audio path cleaned from the background noise (can be VAD filtered speech)
        """

        for row in tqdm(df.iterrows()):
            idx = row[0]
            file_name = f"{row[1].start}_{row[1].end}.wav"
            
            save_path = os.path.join(output_directory, file_name)
            df.loc[idx, 'styled_generated_path'] = save_path

            self.style_tts.voice_conversion_to_file(source_wav=row[1].generated_path, target_wav=row[1].source_path, file_path=save_path)
        return df
