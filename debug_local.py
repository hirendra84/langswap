"""Iterative local debug runner for the langswap pipeline.

Runs the pipeline against ``12.mp4`` (or any other local file passed via
argv) with verbose logging, and runs each pipeline stage separately so we
can see exactly which step fails.  Intermediate JSON outputs (ASR result,
translation, etc.) get cached under ``data/<id>/`` so each rerun skips
stages that already succeeded.

Usage:
    python debug_local.py                    # 12.mp4 -> english
    python debug_local.py 12.mp4 english     # explicit
    python debug_local.py 12.mp4 english russian   # different src/tgt

Env overrides honored:
    LANGSWAP_QWEN_ASR_MODEL, LANGSWAP_QWEN_ALIGNER_MODEL,
    LANGSWAP_TRANSLATEGEMMA_MODEL, LANGSWAP_OMNIVOICE_MODEL
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Loud logging from the moment we start importing.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("debug_local")


def _id_for(video_path: str) -> str:
    return hashlib.md5(str(video_path).encode("utf-8")).hexdigest()[:12]


def _pick_device(want: str) -> str:
    want = (want or "auto").lower()
    if want != "auto":
        return want
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run(video: str, target_lang: str, source_lang: str | None,
        device: str, tts_engine: str, asr_backend: str,
        translation_backend: str, skip_diarization: bool,
        stop_after: str | None) -> int:
    from langswap.file_repository import LocalOnlyFileRepository
    from langswap.pipeline_models.models import TranslationPipelineConfig
    from langswap.translation_pipeline import VideoTranslationPipeline

    video_path = Path(video).expanduser().resolve()
    if not video_path.exists():
        log.error("Video not found: %s", video_path)
        return 2

    public_id = _id_for(str(video_path))
    base_dir = os.environ.get("LANGSWAP_DATA_DIR", "data")
    repo = LocalOnlyFileRepository(public_id, base_dir)

    resolved_device = _pick_device(device)
    log.info("video=%s id=%s device=%s data_dir=%s",
             video_path, public_id, resolved_device, repo.directory)

    config = TranslationPipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        name=public_id,
        public_id=public_id,
        num_speakers=None,
        source_video_path=str(video_path),
        base_dir=base_dir,
        device=resolved_device,
        voice_conv=False,
        tts_model=tts_engine,
        dubbing_algo="speedup",
        eleven_api_token=os.environ.get("ELEVEN_API_KEY"),
        watermark=False,
        skip_diarization=skip_diarization,
        asr_backend=asr_backend,
        translation_backend=translation_backend,
    )

    pipeline = VideoTranslationPipeline(config=config, file_repository=repo)
    stages = [
        ("asr", pipeline._generate_asr),
        ("translation", pipeline._generate_translation),
        ("tts", pipeline._generate_speech),
        ("merge", lambda: setattr(pipeline, "video_translation",
                                  pipeline._merge(pipeline.config.dubbing_algo))),
        ("srt", pipeline.generate_srt_files),
    ]

    for name, fn in stages:
        log.info("===== stage start: %s =====", name)
        try:
            fn()
        except Exception:
            log.exception("Stage %s failed", name)
            return 1
        log.info("===== stage done:  %s =====", name)
        if stop_after == name:
            log.info("Stopping after %s as requested.", name)
            break

    out = pipeline.video_translation.processed_video
    if out and out.file_path:
        log.info("OUTPUT VIDEO: %s", out.file_path)
    log.info("OUTPUT DIR:   %s", repo.directory)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", nargs="?", default="12.mp4")
    p.add_argument("target_lang", nargs="?", default="english")
    p.add_argument("source_lang", nargs="?", default=None,
                   help="optional; pipeline will autodetect if omitted")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--tts", default="omnivoice")
    p.add_argument("--asr", default="qwen")
    p.add_argument("--translation", default="local")
    p.add_argument("--with-diarization", action="store_true",
                   help="enable speaker diarization (off by default for speed)")
    p.add_argument("--stop-after", default=None,
                   choices=["asr", "translation", "tts", "merge", "srt"])
    args = p.parse_args()

    try:
        return run(
            video=args.video,
            target_lang=args.target_lang,
            source_lang=args.source_lang,
            device=args.device,
            tts_engine=args.tts,
            asr_backend=args.asr,
            translation_backend=args.translation,
            skip_diarization=not args.with_diarization,
            stop_after=args.stop_after,
        )
    except Exception:
        log.exception("Fatal error")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
