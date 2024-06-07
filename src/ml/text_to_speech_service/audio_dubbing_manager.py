import os

import torchaudio
import subprocess
import torchaudio.transforms as transforms

from pydub import AudioSegment
from src.file_repository import FileRepository


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

            torchaudio.save(audio_path, resampled_audio, sample_rate=target_sr)
        return audio_path
    
    def enhance_audio(self, input_dir, output_dir):
        try:
            command = [
                'resemble-enhance',
                input_dir,
                output_dir
            ]

            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"Error during the enhancement process: {e}")
            print(f"Output: {e.output}")
            print(f"Error: {e.stderr}")

    
    def split_audio_seconds(self, video_translation, audio_path, temp_folder, sample_rate=24000):
        """
        Splits the audio in seconds mentioned in df. 
        Audio fragments are then used for style transfering.

        Sample rate is set to the TTS engine setting, xtts: 24000
        """
        
        sound_file = AudioSegment.from_wav(audio_path)
        
        for idx, segment in enumerate(video_translation.translated_texts):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")

            sound = sound_file[segment.start * 1000: segment.end * 1000]
            sound = sound.set_frame_rate(sample_rate).set_channels(1)

            sound.export(file_path, format="wav")
            video_translation.translated_texts[idx].source_file = file_path
        return video_translation
