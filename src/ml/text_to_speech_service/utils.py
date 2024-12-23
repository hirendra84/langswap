import torchaudio
import torch

from src.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager


def add_pauses(audio_path: str, num_sec=2):
    audio, sr = torchaudio.load(audio_path)

    pause_start = torch.zeros((1, sr*num_sec))
    pause_end = torch.zeros((1, sr*num_sec))

    audio = torch.cat([pause_start, audio, pause_end], dim=1)

    torchaudio.save(audio_path, audio, sr)


def merge_speaker_files(video_translation, target_speaker: str, idx: int,
                    audio_path: str, window=2):

    current_files_merge = []

    for i in range(max(idx - window, 0), idx + window):
        if video_translation.translated_texts[i].speaker == target_speaker:
            current_files_merge.append(video_translation.translated_texts[i].source_file)
        
    AudioDubbingManager.merge_audio_files(current_files_merge,
                                        audio_path)
