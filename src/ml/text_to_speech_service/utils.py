import torchaudio
import torch


def add_pauses(audio_path: str, num_sec=2):
    audio, sr = torchaudio.load(audio_path)

    pause_start = torch.zeros((1, sr*num_sec))
    pause_end = torch.zeros((1, sr*num_sec))

    audio = torch.cat([pause_start, audio, pause_end], dim=1)

    torchaudio.save(audio_path, audio, sr)