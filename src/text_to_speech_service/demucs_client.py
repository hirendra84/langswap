import demucs
import demucs.api
import torchaudio
import os
from src.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager

class DemucsClient:
    def __init__(self):
        pass

    def separate(self, audio_file: str, repo):
        separator = demucs.api.Separator()
        source_file_path = repo.get_file(f'{audio_file.name}.wav')
        source_file_path = audio_file.file_path

        separated = separator.separate_audio_file(source_file_path)

        background_files = []
        for file, source in separated[1].items():
            save_audio = repo.get_file(f'{file}.wav')
            demucs.api.save_audio(source, save_audio.file_path, samplerate=separator.samplerate)
            save_audio = repo.save_file(save_audio, force=True)

            background_files.append(save_audio)
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