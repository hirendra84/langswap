"""ONNX ASR backend: sherpa-onnx Qwen3-ASR-0.6B-int8 (transcription) + the
Qwen3ForcedAligner (word timestamps).

Drop-in for QwenASRX: same constructor signature, __enter__/__exit__, and
transcribe() return type (an `Output` with word-grouped segments).

Why: the vLLM Qwen3-ASR-1.7B engine pays ~150s of init per cold start.  The
sherpa-onnx 0.6B-int8 ONNX model transcribes a 16s clip in ~7s with no engine
init and runs on CPU — freeing the GPU for TTS/translation.  The forced aligner
(0.6B, torch) supplies the word-level timestamps our pause-based segmentation
needs (sherpa's from_qwen3_asr exposes text only).
"""

import os
import logging
from pathlib import Path
from typing import Optional

from langswap.model_config import resolve_model
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

# Reuse the structures + word-grouping from the vLLM client (single source).
from langswap.ml.speech_to_text_service.asr_qwen_client import (
    Output,
    TranscriptionData,
    _group_words_into_segments,
)
import cattrs

logger = logging.getLogger(__name__)


def _extract_words(align_out) -> list[dict]:
    """Normalize Qwen3ForcedAligner.align(...) output into [{word,start,end}].

    align() returns a list-per-input-sample; we pass a single audio, so the
    word items live in align_out[0], each exposing .text/.start_time/.end_time.
    """
    if not align_out:
        return []
    sample = align_out[0] if isinstance(align_out, (list, tuple)) else align_out
    if not isinstance(sample, (list, tuple)):
        sample = getattr(sample, "items", None) or getattr(sample, "words", None) or []

    words = []
    for item in sample:
        text = getattr(item, "text", None)
        start = getattr(item, "start_time", None)
        end = getattr(item, "end_time", None)
        if text is None or start is None or end is None:
            continue
        words.append({"word": str(text), "start": float(start),
                      "end": float(end), "speaker": "SPEAKER_00"})
    return words


class QwenASROnnx:
    def __init__(
        self,
        device: str,
        language: str,
        skip_diarization: bool = False,
        asr_model_id: Optional[str] = None,
        aligner_model_id: Optional[str] = None,
    ) -> None:
        self.device = device
        self.skip_diarization = skip_diarization
        self.asr_model_id = resolve_model(
            "LANGSWAP_QWEN_ASR_ONNX_MODEL",
            "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
            asr_model_id,
        )
        self.aligner_model_id = resolve_model(
            "LANGSWAP_QWEN_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B", aligner_model_id)

        if language is not None:
            self.language = map_language_to_code(language, system="whisper")
            self.language_full = map_language_to_code(language, system="cohere")
        else:
            self.language = None
            self.language_full = None

        self.recognizer = None
        self.aligner = None
        self.load_models()

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # keep models resident (cheap to hold; reuse-friendly)
        return False

    def load_models(self):
        if self.recognizer is not None and self.aligner is not None:
            return
        if self.recognizer is None:
            self._load_recognizer()
        if self.aligner is None:
            self._load_aligner()

    def _load_recognizer(self):
        import glob
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        d = snapshot_download(self.asr_model_id)
        conv = os.path.join(d, "conv_frontend.onnx")
        enc = sorted(glob.glob(os.path.join(d, "encoder*.onnx")))[0]
        dec = sorted(glob.glob(os.path.join(d, "decoder*.onnx")))[0]
        tok = os.path.join(d, "tokenizer")
        if not os.path.isdir(tok):
            tok = d
        provider = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=conv, encoder=enc, decoder=dec, tokenizer=tok, provider=provider,
        )
        logger.info("sherpa-onnx Qwen3-ASR loaded (provider requested=%s)", provider)

    def _load_aligner(self):
        import torch

        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        device_map = self.device if str(self.device).startswith("cuda") else "cpu"

        def _load():
            from qwen_asr import Qwen3ForcedAligner
            return Qwen3ForcedAligner.from_pretrained(
                self.aligner_model_id, dtype=dtype, device_map=device_map)

        try:
            self.aligner = _load()
        except Exception as e:
            # The aligner may need the transformers-5.x shims; apply and retry.
            logger.info("aligner load failed without shims (%s); applying compat shims", e)
            from langswap.ml.speech_to_text_service._qwen_compat import (
                apply_transformers5_compat_shims,
            )
            apply_transformers5_compat_shims()
            self.aligner = _load()

    def transcribe(self, source_file: str, num_speakers=None) -> Output:
        import soundfile as sf
        import numpy as np

        # sherpa wants mono; resample to 16 kHz to match the model frontend.
        samples, sr = sf.read(str(source_file), dtype="float32")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if sr != 16000:
            import torch
            import torchaudio
            samples = torchaudio.functional.resample(
                torch.from_numpy(samples), sr, 16000).numpy()
            sr = 16000

        import time as _t
        _t0 = _t.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        self.recognizer.decode_stream(stream)
        text = (stream.result.text or "").strip()
        print(f"[timing] asr.sherpa_decode: {_t.perf_counter() - _t0:.1f}s", flush=True)

        lang_full = self.language_full or "English"
        words = []
        if text:
            _t0 = _t.perf_counter()
            # Feed the same 16 kHz mono samples (numpy tuple) so the aligner
            # doesn't re-read/resample the large source file.
            align_out = self.aligner.align(audio=(samples, sr), text=text, language=lang_full)
            print(f"[timing] asr.align: {_t.perf_counter() - _t0:.1f}s", flush=True)
            words = _extract_words(align_out)
            if not words:
                logger.warning(
                    "forced aligner returned no usable words; type=%s sample=%r",
                    type(align_out),
                    (align_out[:1] if isinstance(align_out, (list, tuple)) else align_out),
                )

        if words:
            segments = _group_words_into_segments(words)
        else:
            # No timestamps — emit a single segment spanning the clip so the
            # pipeline still produces output rather than crashing.
            audio_end = float(len(samples)) / sr
            segments = [{"text": text, "start": 0.0, "end": audio_end,
                         "words": [], "speaker": "SPEAKER_00"}]

        detected_language = self.language or map_language_to_code(
            lang_full.lower(), system="whisper")

        final_response = {
            "detected_language": detected_language,
            "device": "cuda" if str(self.device).startswith("cuda") else self.device,
            "model": "qwen3-asr-onnx",
            "transcription": text,
            "translation": "",
            "segments": segments,
        }
        result_data = {"status": "finished", "output": final_response}
        return cattrs.structure(result_data, TranscriptionData).output
