"""
ElevenLabs voice-cloning TTS client with the same generate_audio() interface
as Qwen3TTSClient, suitable for languages not supported by Qwen3TTS (e.g. Arabic).

Voice IDs are cached per source_audio_file so each reference speaker is only
cloned once per client instance.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


class ElevenVoiceCloningClient:
    """
    Voice-cloning TTS using ElevenLabs Instant Voice Cloning (IVC).

    Compatible drop-in for Qwen3TTSClient's generate_audio() signature.
    """

    def __init__(self, api_token: Optional[str] = None):
        try:
            from elevenlabs.client import ElevenLabs
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `elevenlabs`. Install with `pip install elevenlabs`."
            ) from e

        token = api_token or os.environ.get("ELEVEN_API_KEY") or os.environ.get("ELEVEN_LABS_API_KEY")
        if not token:
            raise ValueError("ElevenLabs API key required (ELEVEN_API_KEY env var or api_token param)")

        self.client = ElevenLabs(api_key=token)
        self.sample_rate = 24000
        self._voice_cache: dict[str, str] = {}  # source_audio_path -> voice_id

    def _get_or_clone_voice(self, source_audio_file: str) -> str:
        """Return a cached voice_id, or clone from source_audio_file."""
        key = str(Path(source_audio_file).resolve())
        if key in self._voice_cache:
            return self._voice_cache[key]

        name = f"iwslt_{Path(source_audio_file).stem}"[:100]
        with open(source_audio_file, "rb") as f:
            voice = self.client.voices.ivc.create(name=name, files=[f])
        voice_id = voice.voice_id
        self._voice_cache[key] = voice_id
        print(f"  [ElevenLabs] Cloned voice {name} → {voice_id}")
        return voice_id

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        source_text: str,  # unused by ElevenLabs but kept for API compatibility
        save_path: str,
        language: str = "Arabic",
    ) -> None:
        """Generate voice-cloned audio and save as WAV."""
        # Map Qwen-style full language names or ISO codes to ElevenLabs language_code
        _lang_map = {
            "Arabic": "ar", "arabic": "ar", "ar": "ar",
            "French": "fr", "french": "fr", "fr": "fr",
            "Chinese": "zh", "chinese": "zh", "zh": "zh",
            "English": "en", "english": "en", "en": "en",
        }
        lang_code = _lang_map.get(language, None)

        voice_id = self._get_or_clone_voice(source_audio_file)

        kwargs = dict(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_24000",
        )
        if lang_code:
            kwargs["language_code"] = lang_code

        try:
            audio_bytes = b"".join(self.client.text_to_speech.convert(**kwargs))
        except Exception as e:
            body = getattr(e, "body", None) or {}
            detail = body.get("detail", {}) if isinstance(body, dict) else {}
            status = detail.get("status", "") if isinstance(detail, dict) else ""
            if status == "payment_issue":
                raise RuntimeError(
                    "ElevenLabs account has a payment issue. "
                    "Please resolve the failed invoice at elevenlabs.io/billing."
                ) from e
            raise

        wav_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        sf.write(save_path, wav_array, self.sample_rate)

    def cleanup_voices(self) -> None:
        """Delete all cloned voices from the ElevenLabs account."""
        for voice_id in list(self._voice_cache.values()):
            try:
                self.client.voices.delete(voice_id=voice_id)
            except Exception:
                pass
        self._voice_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup_voices()
