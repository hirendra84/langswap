import demucs
import demucs.api


class DemucsClient:
    def __init__(self):
        pass

    def separate(self, source_file_path: str, target_file_path):
        separator = demucs.api.Separator()
        separated = separator.separate_audio_file(source_file_path)

        for file, source in separated[1].items():
            if file == "other":
                demucs.api.save_audio(source, target_file_path, samplerate=separator.samplerate)
