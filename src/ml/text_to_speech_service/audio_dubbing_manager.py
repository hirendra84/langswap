import os

import torchaudio

import pandas as pd
import torchaudio.transforms as transforms

from pydub import AudioSegment
from src.file_repository import FileRepository
from src.pipeline_models import TextedSegment


class AudioDubbingManager:
    tts_sample_rate: int
    _file_repository: FileRepository

    def __init__(self, tts_sample_rate: int, file_repository: FileRepository):
        self.tts_sample_rate = tts_sample_rate
        self._file_repository = file_repository
    
    @classmethod
    def resample_save(self, audio_path: str, target_sr=16000):
        """
        Function resamples audio and rewrites the file.
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            transform = transforms.Resample(sr, target_sr)
            resampled_audio = transform(waveform)

            # dump a file
            torchaudio.save(audio_path, resampled_audio, sample_rate=target_sr)
        return audio_path
    
    def split_audio_seconds(self, segments: list[TextedSegment], audio_path, sample_rate=24000):
        """
        Splits the audio in seconds mentioned in df. 
        Audio fragments are then used for style transfering.

        Sample rate is set to the TTS engine setting, xtts: 24000
        """
        df = pd.DataFrame(
            [{
                'text': t.text,
                'start': t.start,
                'end': t.end,
            } for t in segments]
        )

        sound_file = AudioSegment.from_wav(audio_path)
        temp_folder = os.path.join(self._file_repository.directory, "splitted_audio")
        os.makedirs(temp_folder, exist_ok=True)
        
        for i, sample in enumerate(segments):
            file_path = os.path.join(temp_folder, f"{sample.start}_{sample.end}.wav")

            sound = sound_file[sample.start * 1000: sample.end * 1000]
            sound = sound.set_frame_rate(sample_rate).set_channels(1)

            df.loc[i, 'source_path'] = file_path
            sound.export(file_path, format="wav")
        return df
