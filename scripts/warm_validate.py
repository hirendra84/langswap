"""In-container warm + validate run.

Loads every engine the default RunPod dub path uses (faster-whisper + VAD ASR +
Gemma translation + OmniVoice TTS + demucs + silero VAD) and runs ONE real
inference each.  Purpose:
  1. Validate the baked weights load offline (whisper, gemma, omnivoice, pyannote).
  2. Fetch the still-missing public models (demucs htdemucs) into /models so
     `docker commit` bakes them.
  3. Warm the vLLM / torch.compile / inductor caches under /models/vllm_cache and
     /models/torchinductor_cache so cold starts skip the ~minute compile.

Engines are loaded sequentially and freed between to stay within 24 GB.  Each
stage is isolated: a failure is logged but does not abort the others, so a single
flaky engine still lets the rest warm.  Exit code reflects the critical trio
(ASR, translation, TTS).
"""
import gc
import os
import subprocess
import sys
import traceback

# Importing model_config first points HF/torch/vLLM caches at /models.
import langswap.model_config  # noqa: F401

WAV = "/tmp/warm_clip.wav"
RESULTS = {}
# Where weights/caches live in the image (set by the Dockerfile); falls back to
# the codebase default. All path checks below derive from this.
MWD = os.environ.get("MODEL_WEIGHTS_DIR", "/app/models_weights")


def _free():
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def stage(name, critical=False):
    def deco(fn):
        print(f"\n===== [{name}] start =====", flush=True)
        try:
            fn()
            RESULTS[name] = ("OK", critical)
            print(f"===== [{name}] OK =====", flush=True)
        except Exception as e:
            RESULTS[name] = (f"FAIL: {e}", critical)
            print(f"===== [{name}] FAIL: {e} =====", flush=True)
            traceback.print_exc()
        finally:
            _free()
        return fn
    return deco


def prep_clip():
    """Provide ~8s of mono 16k audio at WAV.

    Prefer a real speech clip (better validation), but fall back to a synthetic
    tone if the download/transcode fails — model LOADING is the critical check
    and doesn't need intelligible speech, so a flaky network must not abort the
    whole warm/validate+commit run.
    """
    url = "https://storage.yandexcloud.net/langswap-public/ru_source/1.MP4"
    mp4 = "/tmp/warm_src.mp4"
    try:
        subprocess.run(["curl", "-sSL", "--max-time", "180", "-o", mp4, url], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp4, "-t", "8", "-ac", "1", "-ar", "16000", WAV],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"clip ready (real speech): {WAV} ({os.path.getsize(WAV)} bytes)", flush=True)
        return
    except Exception as e:
        print(f"[warn] clip download/transcode failed ({e}); using synthetic tone", flush=True)
    import numpy as np
    import soundfile as sf
    sr = 16000
    t = np.linspace(0, 8, sr * 8, endpoint=False)
    tone = (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(WAV, tone, sr)
    print(f"clip ready (synthetic): {WAV} ({os.path.getsize(WAV)} bytes)", flush=True)


def cleanup_scratch():
    """Remove warm-run scratch so `docker commit` does not bake it into the image."""
    import shutil
    for p in ("/tmp/warm_src.mp4", WAV, "/tmp/warm_tts.wav"):
        try:
            os.remove(p)
        except OSError:
            pass
    shutil.rmtree("/tmp/sep", ignore_errors=True)


ASR_TEXT = {"text": "Hello, this is a short calibration sentence."}


def main():
    prep_clip()

    @stage("silero_vad")
    def _():
        from silero_vad import load_silero_vad
        load_silero_vad()

    @stage("asr_vad", critical=True)
    def _():
        from langswap.ml.speech_to_text_service.asr_vad_client import VADWhisperASR
        with VADWhisperASR(device="cuda", language="russian", skip_diarization=True) as asr:
            out = asr.transcribe(WAV)
        txt = getattr(out, "transcription", None) or getattr(out, "text", None) or ""
        print("ASR text:", repr(txt)[:200], flush=True)
        if txt:
            ASR_TEXT["text"] = txt

    @stage("translation_gemma3", critical=True)
    def _():
        from langswap.ml.translation_service.translator_llamacpp_client import LlamaCppTranslationClient
        t = LlamaCppTranslationClient(device="cuda")
        t.load_models()
        res = t.translate(["Привет, как у тебя дела сегодня?"], "russian", "english")
        print("translation:", res, flush=True)
        assert res and res[0].strip(), "empty translation"

    # OmniVoice (vLLM) needs ~0.5*total VRAM free (its stage config caps
    # gpu_memory_utilization at 0.5).  On a shared GPU that may be unavailable, so
    # this is best-effort: warm the compile cache when VRAM allows, otherwise just
    # verify the baked snapshot is present.  The image still works either way —
    # an un-warmed OmniVoice pays its one-time compile on the first RunPod call,
    # then warm-reuse holds it.
    @stage("tts_omnivoice", critical=False)
    def _():
        import glob
        import torch
        snap = glob.glob(os.path.join(MWD, "models--k2-fsa--OmniVoice/snapshots/*/config.json"))
        assert snap, f"OmniVoice snapshot missing from {MWD} — bake failed"
        print("OmniVoice baked snapshot present:", snap[0], flush=True)

        free, total = torch.cuda.mem_get_info()
        need = int(0.5 * total) + 1_500_000_000  # 0.5*total target + overhead
        print(f"VRAM free={free/1e9:.1f}G total={total/1e9:.1f}G need~={need/1e9:.1f}G", flush=True)
        if free < need:
            print("[skip] insufficient free VRAM to warm OmniVoice on this shared GPU; "
                  "snapshot validated, compile cache will populate on first RunPod call.", flush=True)
            return

        from langswap.ml.text_to_speech_service.tts_omnivoice_client import OmniVoiceClient
        tts = OmniVoiceClient(device="cuda")
        tts.generate_audio(
            text="Hello, this is a short calibration sentence for warmup.",
            source_audio_file=WAV,
            source_text=ASR_TEXT["text"],
            save_path="/tmp/warm_tts.wav",
            language="en",
            duration=3.0,
        )
        assert os.path.exists("/tmp/warm_tts.wav"), "no tts output"
        print("tts out:", os.path.getsize("/tmp/warm_tts.wav"), "bytes", flush=True)

    @stage("demucs")
    def _():
        from langswap.ml.text_to_speech_service.demucs_client import DemucsClient
        os.makedirs("/tmp/sep", exist_ok=True)
        DemucsClient().separate(WAV, "/tmp/sep")

    cleanup_scratch()

    print("\n\n========== WARM/VALIDATE SUMMARY ==========", flush=True)
    failed_critical = False
    for name, (status, critical) in RESULTS.items():
        tag = "CRITICAL" if critical else "optional"
        print(f"  {name:20s} [{tag}] {status}", flush=True)
        if critical and not status.startswith("OK"):
            failed_critical = True

    vc = os.path.join(MWD, "vllm_cache"); ic = os.path.join(MWD, "torchinductor_cache")
    print("\nvLLM cache:", os.listdir(vc) if os.path.isdir(vc) else "MISSING", flush=True)
    print("inductor cache:", os.listdir(ic) if os.path.isdir(ic) else "MISSING", flush=True)
    sys.exit(1 if failed_critical else 0)


if __name__ == "__main__":
    main()
