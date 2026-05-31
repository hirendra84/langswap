"""Quick smoke test for the qwen-asr Docker service.

Usage:
    python scripts/test_asr_service.py [audio_or_video_path] [language]

Default: tests against http://localhost:8001 (the port docker-compose exposes)
using 12.mp4 in the project root and language "Russian".

Extracts audio with ffmpeg first if the input is a video.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


def _is_audio(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _extract_audio(video_path: Path) -> Path:
    out = Path(tempfile.gettempdir()) / f"{video_path.stem}.16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000",
        "-vn", str(out),
    ]
    print(f"[ffmpeg] extracting audio -> {out}")
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default="12.mp4")
    p.add_argument("language", nargs="?", default="Russian")
    p.add_argument("--url", default=os.environ.get("LANGSWAP_QWEN_ASR_URL", "http://localhost:8001"))
    args = p.parse_args()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        print(f"file not found: {src}", file=sys.stderr)
        return 2

    print(f"[health] GET {args.url}/healthz")
    try:
        h = requests.get(f"{args.url}/healthz", timeout=10)
        print(f"  -> {h.status_code} {h.text}")
        if not h.ok:
            return 3
    except Exception as e:
        print(f"  -> service unreachable: {e}", file=sys.stderr)
        return 3

    audio = src if _is_audio(src) else _extract_audio(src)

    print(f"[transcribe] POST {args.url}/transcribe  language={args.language}")
    t0 = time.time()
    with open(audio, "rb") as f:
        r = requests.post(
            f"{args.url}/transcribe",
            files={"audio": (audio.name, f, "application/octet-stream")},
            data={"language": args.language},
            timeout=1800,
        )
    elapsed = time.time() - t0
    print(f"  -> {r.status_code}  ({elapsed:.1f}s)")

    if not r.ok:
        print(r.text[:2000])
        return 4

    body = r.json()
    n_words = len(body.get("words") or [])
    print(f"detected_language: {body.get('detected_language')}")
    print(f"text:              {body.get('text')[:300]}...")
    print(f"words:             {n_words}")
    if n_words:
        w = body["words"][0]
        print(f"first word: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
