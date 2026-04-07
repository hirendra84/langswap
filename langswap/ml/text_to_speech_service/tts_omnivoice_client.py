import os
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from langswap.model_downloader import ensure_omnivoice_model


class OmniVoiceClient:
    """
    OmniVoice TTS client with voice cloning and duration control.
    Supports 600+ languages.

    Voice is cloned from source_audio_file + source_text reference.
    Duration of generated audio is pinned to the segment duration.
    """

    def __init__(
        self,
        model_id: str = "k2-fsa/OmniVoice",
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model_id = str(ensure_omnivoice_model(model_path))
        self.device = device
        self.sample_rate = 24000
        self.model = None
        self.load_models()

    def load_models(self):
        try:
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `torch` required for OmniVoice. "
                "Install PyTorch for your platform, then try again."
            ) from e

        try:
            from omnivoice import OmniVoice
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `omnivoice` required for OmniVoice TTS. "
                "Install it with `pip install omnivoice`."
            ) from e

        if self.model is not None:
            return

        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        is_mps = self.device == "mps"
        device_map = self.device if (is_cuda or is_mps) else "cpu"
        dtype = torch.float16 if is_cuda else torch.float32

        self.model = OmniVoice.from_pretrained(
            self.model_id,
            device_map=device_map,
            dtype=dtype,
        )

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model = None

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        source_text: str,
        save_path: str,
        language: str,
        duration: Optional[float] = None,
    ):
        """Generate speech with voice cloning, pinned to `duration` seconds."""
        if self.model is None:
            self.load_models()

        import torch

        ref_text = (source_text or "").strip() or None

        audios = self.model.generate(
            text=text,
            ref_audio=source_audio_file,
            ref_text=ref_text,
            language=language,
            duration=duration,
        )

        wav = audios[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().float().cpu().numpy()
        if wav.ndim == 2:
            wav = wav[0]

        wav = np.asarray(wav, dtype=np.float32)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, wav, self.sample_rate)

    def tts_pipeline(self, video_translation, temp_folder, language="en"):
        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="OmniVoice generation.",
                leave=True,
            )
        ):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")
            if not os.path.exists(file_path):
                self.generate_audio(
                    text=segment.translation,
                    source_audio_file=segment.source_file,
                    source_text=segment.text,
                    save_path=file_path,
                    language=language,
                    duration=(segment.end - segment.start),
                )
            video_translation.translated_texts[idx].generated_file = file_path
        return video_translation
