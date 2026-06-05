"""Isolated ASR+alignment bake-off on Modal.

Measures, on the SAME audio in one container, the wall-clock (load + inference)
of three ways to get transcript + word-level timestamps:

  A. baseline  - faster-whisper large-v3 + Silero VAD (what the pipeline uses today)
  B. sherpa    - sherpa-onnx Qwen3-ASR-0.6B-int8 (ONNX, CUDA) + Qwen3ForcedAligner (torch, CUDA)
  C. femelo    - py-qwen3-asr-cpp: ASR + aligner both GGUF q8 (llama.cpp, CPU-only)

Run:  modal run bench_asr_modal.py            # uses /tmp/job.json for the audio URL

Each block is independent and wrapped in try/except, so a failure (bad model id,
build error, OOM) is reported and the others still run. The point is the numbers,
not a finished integration — we wire up only the winner.
"""

import modal

app = modal.App("langswap-bench-asr")

# Reuse the production image (vLLM + qwen-asr already there) and add sherpa-onnx.
# py-qwen3-asr-cpp builds llama.cpp from source, so install it at RUNTIME inside
# the femelo block (wrapped) to avoid a build failure killing the whole image.
image = (
    modal.Image.from_dockerfile("Dockerfile")
    .env({"MODEL_WEIGHTS_DIR": "/models"})
    .pip_install("sherpa-onnx")
)

weights = modal.Volume.from_name("langswap-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("langswap-hf")
aws_secret = modal.Secret.from_name("langswap-aws")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/models": weights},
    secrets=[hf_secret, aws_secret],
    timeout=3000,
    cpu=8.0,
)
def bench(job_input: dict):
    import os
    import time
    import glob
    import subprocess
    import traceback
    import uuid

    # ---- fetch the same audio for all three (16k mono wav) -------------------
    from langswap.api import init_s3_client, get_file, BASE_DIR
    from langswap.file_repository import RemoteFileRepository

    s3 = init_s3_client()
    pid = "bench-" + uuid.uuid4().hex[:8]
    repo = RemoteFileRepository(pid, BASE_DIR, s3)
    video_path = get_file(repo, job_input.get("s3_video_url"))

    wav = "/tmp/bench_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav],
        check=True, capture_output=True,
    )
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", wav],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"[bench] audio={wav} duration={dur}s", flush=True)

    rows = []

    def run(name, fn):
        import torch
        print(f"\n[bench] === {name} ===", flush=True)
        try:
            res = fn()
            print(f"[bench] {name} RESULT: {res}", flush=True)
            rows.append((name, res))
        except Exception as e:
            traceback.print_exc()
            rows.append((name, {"error": repr(e)}))
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ---- A. baseline: faster-whisper + VAD (what the pipeline uses today) -----
    def a_vad():
        from langswap.ml.speech_to_text_service.asr_vad_client import VADWhisperASR
        t = time.perf_counter()
        c = VADWhisperASR(device="cuda", language="english", skip_diarization=True)
        c.__enter__()
        load = time.perf_counter() - t
        t = time.perf_counter()
        o = c.transcribe(wav)
        infer = time.perf_counter() - t
        try:
            c.__exit__(None, None, None)
        except Exception:
            pass
        return {"load_s": round(load, 1), "infer_s": round(infer, 1),
                "total_s": round(load + infer, 1), "n_seg": len(o.segments),
                "text": (o.transcription or "")[:160]}

    # ---- B. sherpa-onnx 0.6B int8 (GPU) + torch forced aligner ---------------
    def b_sherpa():
        import soundfile as sf
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        d = snapshot_download("csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25")
        conv = os.path.join(d, "conv_frontend.onnx")
        enc = sorted(glob.glob(os.path.join(d, "encoder*.onnx")))[0]
        dec = sorted(glob.glob(os.path.join(d, "decoder*.onnx")))[0]
        # tokenizer: try a tokenizer/ dir, else the repo root
        tok = os.path.join(d, "tokenizer")
        if not os.path.isdir(tok):
            tok = d
        print(f"[bench] sherpa files: conv={conv} enc={enc} dec={dec} tok={tok}", flush=True)

        t = time.perf_counter()
        rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=conv, encoder=enc, decoder=dec, tokenizer=tok, provider="cuda",
        )
        load = time.perf_counter() - t

        samples, sr = sf.read(wav, dtype="float32")
        t = time.perf_counter()
        st = rec.create_stream()
        st.accept_waveform(sr, samples)
        rec.decode_stream(st)
        text = st.result.text
        infer = time.perf_counter() - t

        # forced aligner (torch, GPU) for word timestamps
        import torch
        from qwen_asr import Qwen3ForcedAligner
        t = time.perf_counter()
        al = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda:0")
        al_load = time.perf_counter() - t
        t = time.perf_counter()
        align = al.align(audio=wav, text=text, language="English")
        al_infer = time.perf_counter() - t

        return {"asr_load_s": round(load, 1), "asr_infer_s": round(infer, 1),
                "align_load_s": round(al_load, 1), "align_infer_s": round(al_infer, 1),
                "total_s": round(load + infer + al_load + al_infer, 1),
                "n_words": len(align), "text": (text or "")[:160]}

    # ---- C. py-qwen3-asr-cpp (CPU, GGUF q8) ----------------------------------
    def c_femelo():
        # build llama.cpp wheel at runtime so an install failure stays contained
        subprocess.run(["pip", "install", "--quiet", "cmake", "py-qwen3-asr-cpp"], check=True)
        from py_qwen3_asr_cpp.model import Qwen3ASRModel

        last = None
        for asr_id, al_id in [
            ("qwen3-asr-0.6b-q8-0", "qwen3-forced-aligner-0.6b-q8-0"),
            ("OpenVoiceOS/qwen3-asr-0.6b-q8-0", "OpenVoiceOS/qwen3-forced-aligner-0.6b-q8-0"),
            ("qwen3-asr-0.6b-q4-k-m", "qwen3-forced-aligner-0.6b-q4-k-m"),
            ("OpenVoiceOS/qwen3-asr-0.6b-q4-k-m", "OpenVoiceOS/qwen3-forced-aligner-0.6b-q4-k-m"),
        ]:
            try:
                t = time.perf_counter()
                m = Qwen3ASRModel(asr_model=asr_id, align_model=al_id, n_threads=8)
                load = time.perf_counter() - t
                t = time.perf_counter()
                res, align = m.transcribe_and_align(wav)
                infer = time.perf_counter() - t
                words = getattr(align, "words", align)
                return {"asr_id": asr_id, "load_s": round(load, 1), "infer_s": round(infer, 1),
                        "total_s": round(load + infer, 1),
                        "n_words": len(words) if hasattr(words, "__len__") else None,
                        "text": (getattr(res, "text", "") or "")[:160]}
            except Exception as e:
                last = e
                print(f"[bench] femelo ids {asr_id} failed: {e!r}", flush=True)
        raise RuntimeError(f"all femelo model-id candidates failed; last={last!r}")

    run("A_vad_whisper_large-v3", a_vad)
    run("B_sherpa_0.6B_int8_gpu+torch_align", b_sherpa)
    run("C_femelo_0.6B_gguf_cpu", c_femelo)

    print("\n[bench] ================= SUMMARY =================", flush=True)
    print(f"[bench] audio duration: {dur}s", flush=True)
    for name, res in rows:
        print(f"[bench] {name}: {res}", flush=True)
    return rows


@app.local_entrypoint()
def main(input_json: str = "/tmp/job.json"):
    import json
    job = json.load(open(input_json))
    bench.remote(job.get("input", job))
