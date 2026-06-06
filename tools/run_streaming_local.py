#!/usr/bin/env python3
"""Run the streaming dub locally and serve the HLS output for the hls.js page.

Usage:
    python tools/run_streaming_local.py path/to/video.mp4 \
        --source-language english --target-language russian \
        [--tts-engine omnivoice] [--asr-backend vad] [--chunk-target 4.0] \
        [--serve]

Writes HLS segments to data/<public_id>/hls/ as they are produced and prints
per-chunk timing + TTFS. With --serve it also starts an HTTP server on that
directory so you can open tools/hls_preview.html and point it at
http://localhost:8000/index.m3u8 to eyeball seam/sync quality live.

This uses LocalOnlyFileRepository, so no S3 credentials are required. It DOES
require the ML stack + a GPU (it runs the real ASR/translate/TTS pipeline).
"""

import argparse
import functools
import http.server
import os
import socketserver
import threading
import time
import uuid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--source-language", default=None)
    ap.add_argument("--target-language", required=True)
    ap.add_argument("--tts-engine", default="omnivoice")
    ap.add_argument("--asr-backend", default="vad")
    ap.add_argument("--translation-backend", default="llamacpp")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-target", type=float, default=4.0)
    ap.add_argument("--skip-diarization", action="store_true")
    ap.add_argument("--serve", action="store_true", help="serve the HLS dir over HTTP")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    public_id = f"stream-{uuid.uuid4().hex[:8]}"
    job = {
        "public_id": public_id,
        "source_video_path": os.path.abspath(args.video),
        "source_language": args.source_language,
        "target_language": args.target_language,
        "tts_engine": args.tts_engine,
        "asr_backend": args.asr_backend,
        "translation_backend": args.translation_backend,
        "device": args.device,
        "chunk_target": args.chunk_target,
        "skip_diarization": args.skip_diarization,
    }

    hls_dir = os.path.join("data", public_id, "hls")
    os.makedirs(hls_dir, exist_ok=True)

    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=hls_dir)
        # CORS so hls.js can fetch from anywhere; SO_REUSEADDR to re-run fast.
        class Server(socketserver.TCPServer):
            allow_reuse_address = True
        httpd = Server(("", args.port), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"[serve] http://localhost:{args.port}/index.m3u8  (dir: {hls_dir})")
        print(f"[serve] open tools/hls_preview.html and load that URL\n")

    # Import here so --help works without the ML stack installed.
    from langswap.streaming import stream_dub

    print(f"[stream] job {public_id} → {hls_dir}")
    t0 = time.perf_counter()
    n = 0
    for ev in stream_dub(job):
        n += 1
        extra = f"  TTFS={ev['ttfs']:.1f}s" if "ttfs" in ev else ""
        print(f"[stream] seg {ev['seg_index']:>3}  "
              f"start={ev['start']:.2f}s dur={ev['duration']:.2f}s "
              f"segs={ev['n_segments']}  +{time.perf_counter()-t0:.1f}s{extra}")
    print(f"[stream] done: {n} segments in {time.perf_counter()-t0:.1f}s")
    print(f"[stream] manifest: {os.path.join(hls_dir, 'index.m3u8')}")

    if args.serve:
        print("\n[serve] still serving — Ctrl-C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
