import demucs
import demucs.api
import torch
import torchaudio
import os
from src.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager


class DemucsClient:
    _separator: demucs.api.Separator

    def __init__(self):
        self._separator = demucs.api.Separator()

    def separate(self, audio_file_path: str, output_directory: str) -> list[tuple[str, str]]:
        separated = self._separator.separate_audio_file(audio_file_path)

        background_files = []
        for file, source in separated[1].items():
            file_name = f'{file}.wav'
            generated_file_path = os.path.join(output_directory, file_name)
            demucs.api.save_audio(source, generated_file_path, samplerate=self._separator.samplerate)

            background_files.append((generated_file_path, file_name))
        return background_files

    def merge_background(self, generated_audio_path, audio_backgrounds: dict[str, str],
                        modes: list | None = None) -> torch.Tensor:
        if modes is None:
            modes = ['other.wav', 'bass.wav', 'drums.wav']
            
        speech_audio, speech_sr = torchaudio.load(generated_audio_path)

        audio = speech_audio
        for m in modes:
            sample_path = audio_backgrounds[m]

            AudioDubbingManager.resample_save(sample_path,
                                target_sr=speech_sr)
                       
            back_sound, sr = torchaudio.load(sample_path)

            if back_sound.shape[1] > audio.shape[1]:
                back_sound = back_sound[0, :speech_audio.shape[1]]
            elif back_sound.shape[1] < audio.shape[1]:
                pause = torch.zeros((1, audio.shape[1] - back_sound.shape[1]))
                back_sound = torch.cat([back_sound, pause], dim=1)

            assert sr == speech_sr, "The sr of generated audio is not equal to background sr"
            audio += back_sound
        return audio, speech_sr
