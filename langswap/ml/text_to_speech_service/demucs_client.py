import torch
import torchaudio
import os
from typing import Optional
from langswap.ml.text_to_speech_service.audio_dubbing_manager import AudioDubbingManager


class DemucsClient:

    def __init__(self, load_separator: bool = True):
        try:
            import demucs.api
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `demucs` required for audio separation. "
                "Install it with `pip install demucs`."
            ) from e
        # The separator model load takes several seconds.  merge_background()
        # never uses it, so callers on the merge-only path pass
        # load_separator=False; separate() lazily loads it if ever needed.
        self._separator = demucs.api.Separator() if load_separator else None

    def _ensure_separator(self):
        if self._separator is None:
            import demucs.api
            self._separator = demucs.api.Separator()

    def separate(self, audio_file_path: str, output_directory: str) -> list[tuple[str, str]]:
        self._ensure_separator()
        background_files = []
        # check of already separated files
        if os.listdir(output_directory):
            modes = ['bass', 'drums', 'other', 'vocals']
            for m in modes:
                file_name = f'{m}.wav'
                generated_file_path = os.path.join(output_directory, file_name)

                background_files.append((generated_file_path, file_name))
        else:
            separated = self._separator.separate_audio_file(audio_file_path)
            for file, source in separated[1].items():
                file_name = f'{file}.wav'
                generated_file_path = os.path.join(output_directory, file_name)
                import demucs.api
                demucs.api.save_audio(source, generated_file_path, samplerate=self._separator.samplerate)

                background_files.append((generated_file_path, file_name))
        return background_files

    def merge_background(self, generated_audio_path, audio_backgrounds: dict[str, str],
                        modes: Optional[list] = None,
                        normalize: bool = True,
                        target_peak: float = 0.95) -> torch.Tensor:
        """
        Merge generated speech audio with background stems (drums, bass, other).

        Args:
            generated_audio_path: Path to the generated dubbed vocals
            audio_backgrounds: Dict mapping stem names to file paths
            modes: List of stems to include (default: drums, bass, other)
            normalize: Whether to apply peak normalization to prevent clipping
            target_peak: Target peak level for normalization (default 0.95 to leave headroom)

        Returns:
            Tuple of (merged audio tensor, sample rate)
        """
        if modes is None:
            modes = ['drums.wav', 'bass.wav', 'other.wav']

        speech_audio, speech_sr = torchaudio.load(generated_audio_path)

        if speech_audio.shape[0] == 2:
            speech_audio = torch.mean(speech_audio, 0).unsqueeze(0)

        audio = speech_audio
        for m in modes:
            sample_path = audio_backgrounds[m]

            AudioDubbingManager.resample_save(sample_path,
                                target_sr=speech_sr)

            back_sound, sr = torchaudio.load(sample_path)
            if back_sound.shape[0] == 2:
                back_sound = torch.mean(back_sound, 0).unsqueeze(0)

            if back_sound.shape[1] > audio.shape[1]:
                back_sound = back_sound[:, :speech_audio.shape[1]]
            elif back_sound.shape[1] < audio.shape[1]:
                pause = torch.zeros((1, audio.shape[1] - back_sound.shape[1]))
                back_sound = torch.cat([back_sound, pause], dim=1)

            assert sr == speech_sr, "The sr of generated audio is not equal to background sr"
            audio += back_sound

        # Apply peak normalization to prevent clipping
        if normalize:
            peak = torch.abs(audio).max()
            if peak > target_peak:
                audio = audio * (target_peak / peak)

        return audio, speech_sr
