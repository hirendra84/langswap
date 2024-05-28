import demucs
import demucs.api
import torchaudio
import os
from src.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager


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

    def merge_background(self, generated_audio_path, repo,
                        modes=['other', 'bass', 'drums'],
                        target_sr=48000):
        speech_audio, sr = torchaudio.load(generated_audio_path)

        audio = speech_audio
        for m in modes:
            sample_path = repo.get_file(f'{m}.wav')

            # TODO: make it return a tensor
            AudioDubbingManager.resample_save(sample_path.file_path,
                                target_sr=target_sr)            
            back_sound, sr = torchaudio.load(sample_path.file_path)

            audio += back_sound[0, :speech_audio.shape[1]]
        return audio