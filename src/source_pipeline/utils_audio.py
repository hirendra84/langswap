

import os
import torchaudio
import torchaudio.transforms as transforms
from pydub import utils, AudioSegment



def change_path_name(file_path, postfix):
    folder, audio_name = os.path.split(file_path)
    path_base, ext = os.path.splitext(audio_name)
    updated_path = os.path.join(folder, path_base + postfix + ext)
    updated_path = updated_path.replace(" ", "_")
    return updated_path

def resample_save(audio_path: str, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    transform = transforms.Resample(sr, target_sr)
    resampled_audio = transform(waveform)
    updated_path = change_path_name(audio_path, "_16000")
    torchaudio.save(updated_path, resampled_audio, sample_rate=target_sr)
    return updated_path

def prepare_split(audio_path: str, seconds_start=0, seconds_end=60):
    sound_file = AudioSegment.from_wav(audio_path)
    updated_path = change_path_name(audio_path, f"_{seconds_end}")
    if seconds_end != 0:
        sound_file = sound_file[seconds_start*1000: seconds_end*1000]
    else:
        sound_file = sound_file[seconds_start*1000:]

    sound_file.export(updated_path, format="wav")
    return updated_path