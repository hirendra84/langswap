"""
Fast iteration test: ASR → Translation → TTS on a short clip.
Skips Demucs and uses a 15-second ffmpeg clip for speed.

Usage:
    uv run --no-sync python test_quick.py [video] [--stage asr|translate|tts|all]

Examples:
    uv run --no-sync python test_quick.py
    uv run --no-sync python test_quick.py test_videos/tanks.mp4
    uv run --no-sync python test_quick.py test_videos/standup.mp4 --stage asr
    uv run --no-sync python test_quick.py --stage translate   # re-run translation on cached ASR
    uv run --no-sync python test_quick.py --stage tts         # re-run TTS on cached translation
"""
import os, sys, json, subprocess, argparse
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dotenv import load_dotenv
load_dotenv()

CACHE_DIR = "/tmp/langswap_quick_test"
os.makedirs(CACHE_DIR, exist_ok=True)

CLIP_PATH  = os.path.join(CACHE_DIR, "clip.wav")
ASR_PATH   = os.path.join(CACHE_DIR, "asr.json")
TRANS_PATH = os.path.join(CACHE_DIR, "translated.json")
TTS_DIR    = os.path.join(CACHE_DIR, "tts")
os.makedirs(TTS_DIR, exist_ok=True)

CLIP_DURATION = 15  # seconds
SOURCE_LANG   = "russian"
TARGET_LANG   = "english"
DEVICE        = "mps"


# ── helpers ────────────────────────────────────────────────────────────────────

def extract_clip(video_path: str):
    print(f"[clip] Extracting {CLIP_DURATION}s from {video_path} → {CLIP_PATH}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-t", str(CLIP_DURATION),
        "-ar", "16000", "-ac", "1",
        "-vn", CLIP_PATH,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode())
        sys.exit(1)
    print(f"[clip] Done: {os.path.getsize(CLIP_PATH)//1024} KB")


def run_asr():
    print("\n[ASR] Loading Qwen3-ASR-0.6B …")
    from langswap.ml.speech_to_text_service.asr_qwen_client import QwenASRX

    client = QwenASRX(
        device=DEVICE,
        language=SOURCE_LANG,
        skip_diarization=True,
    )
    print("[ASR] Transcribing …")
    output = client.transcribe(CLIP_PATH)
    print(f"[ASR] detected_language={output.detected_language}")
    print(f"[ASR] transcription={output.transcription!r}")
    print(f"[ASR] segments ({len(output.segments)}):")
    for s in output.segments:
        print(f"  [{s.start:.2f}-{s.end:.2f}] {s.text!r}")

    data = [{"text": s.text, "start": s.start, "end": s.end, "speaker": s.speaker}
            for s in output.segments]
    with open(ASR_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[ASR] Saved → {ASR_PATH}")
    return data


def run_translation(segments):
    print("\n[Translation] Loading TranslateGemma …")
    from langswap.ml.translation_service.translator_client import LLMTranslationClient

    client = LLMTranslationClient(device=DEVICE)
    client.load_models()

    texts = [s["text"] for s in segments]
    print(f"[Translation] Translating {len(texts)} segment(s) …")
    translated = client.translate(
        sentences=texts,
        source_language=SOURCE_LANG,
        target_language=TARGET_LANG,
    )
    result = []
    for orig, trans, seg in zip(texts, translated, segments):
        print(f"  [{seg['start']:.2f}-{seg['end']:.2f}]")
        print(f"    src: {orig!r}")
        print(f"    tgt: {trans!r}")
        result.append({**seg, "translation": trans})

    with open(TRANS_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Translation] Saved → {TRANS_PATH}")
    return result


def run_tts(segments):
    print("\n[TTS] Loading Qwen3-TTS …")
    from langswap.ml.text_to_speech_service.tts_qwen3_client import Qwen3TTSClient

    client = Qwen3TTSClient(device=DEVICE)

    for i, seg in enumerate(segments):
        text = seg.get("translation") or seg["text"]
        out_path = os.path.join(TTS_DIR, f"seg_{i:03d}.wav")
        print(f"[TTS] seg {i}: {text!r}")
        client.generate_audio(
            text=text,
            language=TARGET_LANG,
            source_audio_file=CLIP_PATH,
            source_text=seg["text"],
            save_path=out_path,
        )
        if os.path.exists(out_path):
            print(f"  → {out_path} ({os.path.getsize(out_path)//1024} KB)")
        else:
            print(f"  → MISSING {out_path}")
    print(f"[TTS] All segments saved to {TTS_DIR}/")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="test_videos/tanks.mp4")
    parser.add_argument("--stage", choices=["asr", "translate", "tts", "all"], default="all")
    parser.add_argument("--fresh", action="store_true", help="Ignore cache, rerun everything")
    args = parser.parse_args()

    # Clip extraction
    if args.stage in ("asr", "all") or not os.path.exists(CLIP_PATH):
        if not os.path.exists(CLIP_PATH) or args.fresh:
            extract_clip(args.video)

    # ASR
    if args.stage in ("asr", "all"):
        if os.path.exists(ASR_PATH) and not args.fresh:
            print(f"[ASR] Using cache: {ASR_PATH}")
            with open(ASR_PATH) as f:
                segments = json.load(f)
            print(f"[ASR] {len(segments)} segment(s) from cache")
            for s in segments:
                print(f"  [{s['start']:.2f}-{s['end']:.2f}] {s['text']!r}")
        else:
            segments = run_asr()
    elif os.path.exists(ASR_PATH):
        with open(ASR_PATH) as f:
            segments = json.load(f)
    else:
        print("[ERROR] No ASR cache found. Run with --stage asr first.")
        sys.exit(1)

    if not segments:
        print("[WARN] No segments from ASR — nothing to translate/synthesize.")
        return

    # Translation
    if args.stage in ("translate", "all"):
        if os.path.exists(TRANS_PATH) and not args.fresh and args.stage == "all":
            print(f"[Translation] Using cache: {TRANS_PATH}")
            with open(TRANS_PATH) as f:
                segments = json.load(f)
        else:
            segments = run_translation(segments)

    # TTS
    if args.stage in ("tts", "all"):
        if args.stage == "tts" and os.path.exists(TRANS_PATH):
            with open(TRANS_PATH) as f:
                segments = json.load(f)
        run_tts(segments)

    print("\n✓ Done")


if __name__ == "__main__":
    main()
