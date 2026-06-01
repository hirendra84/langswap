import os
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code
from langswap.model_config import resolve_model


class Qwen3TTSClient:
    """
    Qwen3-TTS voice clone client.

    Uses `source_audio_file` + `source_text` as the reference for voice cloning.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "auto",
        attn_implementation: Optional[str] = "flash_attention_2",
    ):
        # Auto-downloads from HF into models_weights on first use.
        self.model_id = resolve_model(
            "LANGSWAP_QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", model_path)
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation

        # Best-effort default; will be overwritten if model exposes/returns a different SR.
        self.sample_rate = 24000

        self.model = None
        self.load_models()

    def _to_qwen_language(self, language: str) -> str:
        """
        Qwen3-TTS expects full language names (e.g. "English", "Russian").

        This project sometimes passes:
        - language name keys from `language_codes.json` (e.g. "russian")
        - whisper codes (e.g. "ru")
        """
        if not language:
            return "English"

        lang = language.strip().lower()

        # whisper code -> project key (e.g. "ru" -> "russian")
        if len(lang) <= 3 and lang in ("en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "it", "ar"):
            try:
                lang = map_language_to_code(lang, system="reverse_from_whisper")
            except Exception:
                # Fall back to original code if mapping fails.
                pass

        # project key -> proper name (e.g. "russian" -> "Russian")
        try:
            return map_language_to_code(lang, system="cohere")
        except Exception:
            # As a safe fallback, title-case the string ("english" -> "English")
            return lang.title()

    def load_models(self):
        # Lazy import so environments without qwen-tts can still import the package.
        try:
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `torch` required for Qwen3-TTS. "
                "Install PyTorch for your platform, then try again."
            ) from e

        try:
            # transformers 5.x compat: qwen-tts imports `check_model_inputs`,
            # which was removed in transformers 5.x. Re-add a no-op so the
            # import succeeds regardless of which backend loaded first.
            import transformers.utils.generic as _tug
            if not hasattr(_tug, "check_model_inputs"):
                def _check_model_inputs(*_a, **_k):
                    def _decorator(fn):
                        return fn
                    return _decorator
                _tug.check_model_inputs = _check_model_inputs

            from qwen_tts import Qwen3TTSModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `qwen-tts` required for Qwen3-TTS. "
                "Install it with `pip install -U qwen-tts`."
            ) from e

        if self.model is not None:
            return

        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        device_map = self.device if is_cuda else "cpu"

        if self.dtype == "auto":
            dtype = torch.bfloat16 if is_cuda else torch.float32
        else:
            dtype = getattr(torch, self.dtype)

        # Try with attention optimization, then gracefully fall back.
        if self.attn_implementation and is_cuda:
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    device_map=device_map,
                    dtype=dtype,
                    attn_implementation=self.attn_implementation,
                )
            except Exception:
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    device_map=device_map,
                    dtype=dtype,
                )
        else:
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=device_map,
                dtype=dtype,
            )

        # Prefer model-advertised SR if present.
        for attr in ("sample_rate", "sampling_rate", "sr"):
            sr = getattr(self.model, attr, None)
            if isinstance(sr, int) and sr > 0:
                self.sample_rate = sr
                break

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
        duration=None,  # kept for signature compatibility
    ):
        """
        Generates audio with voice cloning.
        """
        if self.model is None:
            self.load_models()

        qwen_language = self._to_qwen_language(language)

        # Qwen3-TTS voice cloning expects a short transcript aligned to ref audio.
        ref_text = (source_text or "").strip()
        if not ref_text:
            # Avoid hard failure if ASR text is missing.
            ref_text = " "

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=qwen_language,
            ref_audio=source_audio_file,
            ref_text=ref_text,
        )

        if isinstance(sr, int) and sr > 0:
            self.sample_rate = sr

        wav = wavs[0]
        try:
            import torch

            if isinstance(wav, torch.Tensor):
                wav = wav.detach().float().cpu().numpy()
        except Exception:
            pass

        wav = np.asarray(wav, dtype=np.float32)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, wav, self.sample_rate)

    def tts_pipeline(self, video_translation, temp_folder, language="en"):
        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="Voice generation pipeline.",
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

