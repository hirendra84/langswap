"""Thin HTTP client for the qwen-asr Docker service.

Drop-in replacement for QwenASRX with the same constructor signature and
transcribe() return type, but offloads all the model-loading work to a
separate container.  Service URL comes from LANGSWAP_QWEN_ASR_URL (default
http://localhost:8000).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import attr
import cattrs
import requests

from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class Segment:
    end: float
    start: float
    text: str
    words: list[dict]
    speaker: str = None


@attr.s(auto_attribs=True)
class Output:
    detected_language: str
    device: str
    model: str
    transcription: str
    translation: str = None
    segments: list[Segment] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class TranscriptionData:
    output: Output


def _group_words_into_segments(words: list[dict], pause_threshold: float = 0.5) -> list[dict]:
    if not words:
        return []
    segments = []
    current = [words[0]]
    for word in words[1:]:
        prev_end = current[-1].get("end", 0.0)
        cur_start = word.get("start", 0.0)
        split = (
            (cur_start - prev_end) > pause_threshold
            or word.get("speaker", "SPEAKER_00") != current[-1].get("speaker", "SPEAKER_00")
        )
        if split:
            segments.append(_make_segment(current))
            current = [word]
        else:
            current.append(word)
    segments.append(_make_segment(current))
    return segments


def _make_segment(words: list[dict]) -> dict:
    return {
        "text": " ".join(w["word"] for w in words),
        "start": words[0].get("start", 0.0),
        "end": words[-1].get("end", 0.0),
        "words": words,
        "speaker": words[0].get("speaker", "SPEAKER_00"),
    }


class QwenASRRemoteClient:
    """HTTP client to the qwen-asr service.  Same surface as ``QwenASRX``."""

    def __init__(
        self,
        device: str,
        language: Optional[str],
        skip_diarization: bool = False,
        service_url: Optional[str] = None,
        timeout: float = 1800.0,
    ) -> None:
        self.device = device
        self.skip_diarization = skip_diarization
        self.service_url = (
            service_url
            or os.environ.get("LANGSWAP_QWEN_ASR_URL")
            or "http://localhost:8000"
        ).rstrip("/")
        self.timeout = timeout

        if language is not None:
            self.language = map_language_to_code(language, system="whisper")
            self.language_full = map_language_to_code(language, system="cohere")
        else:
            self.language = None
            self.language_full = None

        logger.info("QwenASRRemoteClient bound to %s (lang=%s)", self.service_url, self.language_full)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def healthcheck(self) -> bool:
        try:
            r = requests.get(f"{self.service_url}/healthz", timeout=5)
            return r.ok and bool(r.json().get("ok"))
        except Exception:
            return False

    def transcribe(self, source_file: str, num_speakers=None) -> Output:
        with open(source_file, "rb") as f:
            files = {
                "audio": (
                    os.path.basename(source_file),
                    f,
                    "application/octet-stream",
                )
            }
            data: dict[str, str] = {}
            if self.language_full:
                data["language"] = self.language_full
            if num_speakers is not None:
                data["num_speakers"] = str(int(num_speakers))

            logger.info("POST %s/transcribe (%s)", self.service_url, source_file)
            resp = requests.post(
                f"{self.service_url}/transcribe",
                files=files,
                data=data,
                timeout=self.timeout,
            )

        if not resp.ok:
            raise RuntimeError(
                f"Qwen ASR service returned {resp.status_code}: {resp.text[:500]}"
            )
        body = resp.json()

        raw_lang = body.get("detected_language") or self.language_full or "en"
        try:
            detected_language = map_language_to_code(raw_lang.lower(), system="whisper")
        except Exception:
            detected_language = self.language or raw_lang

        words = body.get("words") or []
        # Diarization is not done service-side — stamp a stub speaker so the
        # downstream remap_pauses logic works the same way as the inline client.
        for w in words:
            w.setdefault("speaker", "SPEAKER_00")

        segments = _group_words_into_segments(words)
        full_text = body.get("text") or " ".join(w.get("word", "") for w in words)

        final = {
            "detected_language": detected_language,
            "device": "remote",
            "model": "qwen3-asr",
            "transcription": full_text,
            "translation": "",
            "segments": segments,
        }
        return cattrs.structure(
            {"status": "finished", "output": final}, TranscriptionData
        ).output
