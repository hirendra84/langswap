# langswap — advanced guide

Detailed reference for running langswap from source, the model registry, all
environment variables, Docker build notes, and troubleshooting.
For the quick path see the [README](../README.md).

The pipeline: **ASR** (Qwen3-ASR + ForcedAligner) → **translation** (Gemma) →
**TTS** (OmniVoice and others) → **dubbing/merge** → muxed video + SRT subtitles.
It runs entirely on a local machine (no S3/AWS required).

---

## 1. Prerequisites

- **Python 3.12** and [`uv`](https://github.com/astral-sh/uv)
- **NVIDIA GPU** with a recent driver (CUDA 13 capable; the stack uses `torch==2.11.0+cu130`)
- **System tools:**
  - `ffmpeg` — audio/video processing
  - `rubberband-cli` — time-stretching for the `speedup` / `stretch_whole` dubbing algorithms
    ```bash
    sudo apt-get install -y ffmpeg rubberband-cli
    ```
- **Docker** + the **NVIDIA Container Toolkit** — to run the all-in-one container
- **HuggingFace token** (`HF_TOKEN`) — required for the gated models (`google/gemma-3-4b-it`, pyannote)

Create a `.env` in the project root:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
ELEVEN_API_KEY=...                     # only if using the ElevenLabs TTS backend
MODEL_WEIGHTS_DIR=./models_weights     # where model weights live (used by Docker too)
LANGSWAP_DATA_DIR=./data               # where outputs/artifacts go
```

---

## 2. Install dependencies

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[demo]"      # ".[demo]" adds gradio; drop it for headless use
```

> **Important — torch/vLLM ABI:** `vllm==0.21.0` (used by ASR and OmniVoice TTS) is built
> against `torch==2.11.0`. `requirements.txt` is unpinned, so if a fresh resolve pulls
> `torch>=2.12` you will hit `undefined symbol: ...getCurrentCUDABlasHandle`. Pin it back with:
> ```bash
> uv pip install "torch==2.11.0" "torchaudio==2.11.0" "torchvision==0.26.0" \
>   --index-url https://download.pytorch.org/whl/cu130
> ```

### Reproducible install (exact, pinned)

To recreate the exact known-good environment (the editable `-e .` flow above re-resolves and can
drift), use the fully pinned lock. It embeds the PyTorch cu130 index and must be installed with
`--no-deps` (every transitive dependency is already pinned):

```bash
# uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --no-deps -r requirements.lock.txt
uv pip install -e . --no-deps

# pip
python3.12 -m venv .venv && source .venv/bin/activate
pip install --no-deps -r requirements.lock.txt
pip install -e . --no-deps
```

Regenerate the lock after changing dependencies:
```bash
uv pip freeze | grep -vE '^-e |^langswap==| @ file://' >> requirements.lock.txt   # then re-add the header
```

---

## 3. Download model weights

Weights are fetched with the bundled CLI into `MODEL_WEIGHTS_DIR` (default `./models_weights`).

```bash
# list everything in the registry
langswap-download-models --list

# download all models (HF_TOKEN must be set for the gated ones)
langswap-download-models --all

# or download individually
langswap-download-models --model qwen3-asr
langswap-download-models --model qwen3-asr-aligner
langswap-download-models --model gemma-translate     # gated -> needs HF_TOKEN
langswap-download-models --model omnivoice
```

| Registry name                   | Repo                              | Gated | Used by              |
|----------------------------------|-----------------------------------|-------|----------------------|
| `qwen3-asr`                      | `Qwen/Qwen3-ASR-1.7B`             | No    | ASR                  |
| `qwen3-asr-aligner`              | `Qwen/Qwen3-ForcedAligner-0.6B`   | No    | ASR                  |
| `gemma-translate`                | `google/gemma-3-4b-it`            | **Yes** | Translation        |
| `omnivoice`                      | `k2-fsa/OmniVoice`                | No    | TTS                  |
| `pyannote-speaker-diarization`   | `pyannote/speaker-diarization-3.1`| **Yes** | Diarization (opt.) |
| `pyannote-segmentation`          | `pyannote/segmentation-3.0`       | **Yes** | Diarization (opt.) |

Override any repo id at runtime without editing code:
`LANGSWAP_QWEN_ASR_MODEL`, `LANGSWAP_TRANSLATEGEMMA_MODEL`, `LANGSWAP_OMNIVOICE_MODEL`.

---

## 4. Run locally for debugging (`debug_local.py`)

Runs each pipeline stage separately with verbose logging and caches intermediate JSON under
`data/<id>/`, so reruns skip stages that already succeeded.

```bash
.venv/bin/python debug_local.py 12.mp4 english russian 2>&1 | tee /tmp/langswap_debug.log
```

Positional args: `<video> [target_lang] [source_lang]` (source is auto-detected if omitted).

Useful flags:

| Flag                  | Default     | Notes                                                       |
|-----------------------|-------------|-------------------------------------------------------------|
| `--device`            | `auto`      | `auto` / `cuda` / `mps` / `cpu`                             |
| `--asr`               | `qwen`      | `qwen` (Qwen3-ASR in-process) / `whisperx` / `openai` |
| `--translation`       | `local`     | `local` (Gemma) / `vllm` / `openai`                        |
| `--tts`               | `omnivoice` | `omnivoice` / `xtts` / `f5tts` / `chatterbox` / `qwen3` / `elevenlabs` |
| `--with-diarization`  | off         | enable speaker diarization (needs pyannote weights + token) |
| `--stop-after`        | —           | stop after `asr` / `translation` / `tts` / `merge` / `srt`  |

Output: `data/<id>/resulted_video.mp4`, plus `source_transcript.srt` and `translated_transcript.srt`.

---

## 5. Run the Gradio demo (`gradio_demo.py`)

Browser UI for the full pipeline; uses `LocalOnlyFileRepository` (no AWS/S3).

```bash
.venv/bin/python gradio_demo.py          # http://localhost:7860  (add --share for a public link)
```

It loads `.env` and runs the full pipeline locally, with Qwen3-ASR loaded **in-process**
(default ASR backend `qwen`). Switch backends / languages / dubbing algorithm in the
*Models / backends* accordion.

Other flags: `--host`, `--port`, `--share`, `--data-dir`.

---

## 6. Docker build notes

The single [`Dockerfile`](../Dockerfile) builds the whole pipeline into one image: it installs the
deps on `transformers==5.9.0` + `vllm==0.21.0` (forced via [overrides.txt](../overrides.txt), which
vllm-omni's voice cloning requires — it needs `transformers>=5.3.0`), then adds `qwen-asr` and
`qwen-tts` with `--no-deps`. Those two hard-pin `transformers==4.57.x` but run fine on 5.x via the
in-process compat shims in `asr_qwen_client.py` / `tts_qwen3_client.py`, so ASR runs in the same
process — no separate service.

- Built on `ubuntu24.04` (Python 3.12 is native there; 22.04 only ships 3.10).
- `demucs` is built from a git sdist, which fetches `setuptools`/`wheel` from PyPI during build
  isolation. The Dockerfile wraps the dependency install in a retry loop and installs the editable
  package with `--no-build-isolation` to tolerate flaky outbound network.
- You will see `vLLM and vLLM-Omni appear to have mismatched major/minor versions` — this warning
  is expected for the locked `vllm 0.21.0` / `vllm-omni 0.20.0` pair and is harmless.

---

## Environment variables

| Variable                      | Purpose                                                            |
|-------------------------------|--------------------------------------------------------------------|
| `HF_TOKEN`                    | HuggingFace token for gated models (Gemma, pyannote)               |
| `ELEVEN_API_KEY`              | ElevenLabs key (only for `--tts elevenlabs`)                       |
| `MODEL_WEIGHTS_DIR`           | Where weights are stored/loaded (default `./models_weights`)       |
| `LANGSWAP_DATA_DIR`           | Where intermediate artifacts/outputs go (default `data/`)          |
| `LANGSWAP_QWEN_ASR_MODEL`     | Override the ASR repo id                                           |
| `LANGSWAP_TRANSLATEGEMMA_MODEL` | Override the translation model repo id                           |
| `LANGSWAP_OMNIVOICE_MODEL`    | Override the TTS model repo id                                     |
| `LANGSWAP_QWEN_ASR_GPU_UTIL`  | In-process ASR vLLM `gpu_memory_utilization` (default `0.5`)       |
| `LANGSWAP_QWEN_ASR_MAX_LEN`   | In-process ASR vLLM `max_model_len` (default `16384`)              |

---

## Troubleshooting

- **`undefined symbol: ...getCurrentCUDABlasHandle`** — torch is newer than vLLM expects; pin
  `torch==2.11.0+cu130` (see §2).
- **`Failed to execute rubberband`** — install `rubberband-cli` (needed by `speedup`/`stretch_whole`).
- **`Cannot re-initialize CUDA in forked subprocess`** — handled in code (vLLM uses `spawn`); if you
  hit it elsewhere, set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- **`Free memory ... less than desired GPU memory utilization`** — the GPU is shared. Lower
  `LANGSWAP_QWEN_ASR_GPU_UTIL`, or free other processes. Note Gradio is long-lived, so restart it to
  release accumulated GPU memory between runs.

---

## Project structure

- `debug_local.py` — local stage-by-stage CLI runner
- `gradio_demo.py` — local browser UI
- `Dockerfile` / `overrides.txt` — all-in-one image (full pipeline + Gradio UI on transformers 5.9 / vllm 0.21)
- `langswap/translation_pipeline.py` — pipeline orchestration
- `langswap/ml/` — ASR / translation / TTS / dubbing implementations
- `langswap/model_downloader.py` — model registry + `langswap-download-models` CLI

---

## Model licenses

This repository licenses the **pipeline code only** ([AGPL-3.0-or-later](../LICENSE)). The models it
downloads and runs at inference time are **not** covered by the AGPL and carry their own terms, some
of which restrict commercial or derivative use:

| Component | Model | License |
| --- | --- | --- |
| Translation | TranslateGemma (built on Google **Gemma**) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — *use restrictions apply; not OSI-approved* |
| ASR | Qwen3-ASR | Per-model on Hugging Face (Apache-2.0 or Qwen license) — verify before use |
| TTS | `k2-fsa/OmniVoice` | See the model's Hugging Face model card |
| ElevenLabs / OpenAI backends | Hosted APIs | Provider commercial Terms of Service |

You are responsible for complying with each model's license for your use case. The AGPL grant on this
code does **not** grant any rights to the model weights.
