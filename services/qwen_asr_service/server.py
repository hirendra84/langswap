"""FastAPI service wrapping qwen-asr Qwen3-ASR + ForcedAligner.

Runs in its own container with transformers==4.57.6 (qwen-asr's pin), so the
loader works without any of the transformers-5.x compatibility shims the
main app would otherwise need.

Env overrides:
    QWEN_ASR_MODEL        HF repo id or local path (default Qwen/Qwen3-ASR-1.7B)
    QWEN_ALIGNER_MODEL    aligner model id/path  (default Qwen/Qwen3-ForcedAligner-0.6B)
    DEVICE                cuda|cpu|mps (default cuda)
    BACKEND               vllm|transformers|auto (default auto)
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("qwen_asr_service")

ASR_MODEL_ID = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_MODEL_ID = os.environ.get("QWEN_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
DEVICE = os.environ.get("DEVICE", "cuda")
BACKEND = os.environ.get("BACKEND", "auto").lower()

_MODEL = None


def _load_model():
    global _MODEL
    from qwen_asr import Qwen3ASRModel  # transformers 4.57 inside this container

    dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
    log.info(
        "Loading Qwen3-ASR: model=%s aligner=%s device=%s backend=%s",
        ASR_MODEL_ID, ALIGNER_MODEL_ID, DEVICE, BACKEND,
    )

    try_vllm = BACKEND in {"auto", "vllm"}
    if try_vllm:
        try:
            import vllm  # noqa: F401
            _MODEL = Qwen3ASRModel.LLM(
                model=ASR_MODEL_ID,
                forced_aligner=ALIGNER_MODEL_ID,
                forced_aligner_kwargs={"dtype": dtype, "device_map": DEVICE},
                dtype="bfloat16",
                trust_remote_code=True,
            )
            log.info("Loaded with vllm backend")
            return
        except ImportError:
            if BACKEND == "vllm":
                raise
            log.info("vllm not available, falling back to transformers backend")

    _MODEL = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_ID,
        forced_aligner=ALIGNER_MODEL_ID,
        forced_aligner_kwargs={"dtype": dtype, "device_map": DEVICE},
        dtype=dtype,
        device_map=DEVICE,
    )
    log.info("Loaded with transformers backend")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield
    # nothing to clean — process exit releases GPU memory


app = FastAPI(title="Qwen3-ASR Service", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"ok": _MODEL is not None, "model": ASR_MODEL_ID}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    num_speakers: Optional[int] = Form(None),
):
    if _MODEL is None:
        raise HTTPException(503, "Model not loaded yet")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        path = f.name

    try:
        results = _MODEL.transcribe(
            audio=path,
            language=language,            # Qwen expects full names ("Russian", "English", ...)
            return_time_stamps=True,
        )
        result = results[0]

        words: list[dict] = []
        if result.time_stamps is not None and result.time_stamps.items:
            for item in result.time_stamps.items:
                words.append({
                    "word": item.text,
                    "start": float(item.start_time),
                    "end": float(item.end_time),
                })

        return {
            "detected_language": result.language or "en",
            "text": result.text or " ".join(w["word"] for w in words),
            "words": words,
        }
    except Exception as e:
        log.exception("transcribe failed")
        raise HTTPException(500, f"transcribe failed: {type(e).__name__}: {e}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
