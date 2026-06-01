# All-in-one container — the entire langswap pipeline in a single image.
#
#   ASR (Qwen3-ASR + ForcedAligner) → translation (Gemma) → TTS (OmniVoice /
#   Qwen3-TTS / …) → dubbing → muxed video, served through the Gradio UI.
#
# Everything runs in ONE process on transformers 5.x.  Historically ASR needed
# its own container pinned to transformers 4.57.x; the in-process compat shims
# in asr_qwen_client.py and tts_qwen3_client.py now make qwen-asr and qwen-tts
# import and run on 5.x, so the separate microservice is no longer required.
#
# Build:  docker build -t langswap:latest .
# Run:    docker run --rm --gpus all -p 7860:7860 \
#           -e HF_TOKEN=$HF_TOKEN \
#           -v "$PWD/models_weights:/models" -v "$PWD/data:/app/data" \
#           langswap:latest
#         # then open http://localhost:7860
#
# Ubuntu 24.04 (noble) ships python3.12 natively — 22.04 (jammy) only has 3.10
# and deadsnakes does not backport 3.12 to jammy, so we use the noble base.
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        git \
        curl \
        ffmpeg \
        libsndfile1 \
        libasound2-dev \
        libportaudio2 \
        portaudio19-dev \
        rubberband-cli \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# noble's python3 is 3.12.  --break-system-packages bypasses PEP 668's
# externally-managed marker so we can install into the system interpreter.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python3 -m pip install --break-system-packages uv

# Force every `uv pip install --system` below to target the system python3.12,
# and allow writing into noble's externally-managed (PEP 668) system env.
ENV UV_PYTHON=/usr/bin/python3 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_HTTP_TIMEOUT=180

WORKDIR /app

# Install Python deps from the project's requirements + extras.
#
# overrides.txt pins transformers==5.9.0 / vllm==0.21.0 — vllm-omni's voice
# cloning needs transformers>=5.3.0, and the compat shims let qwen-asr/qwen-tts
# run on 5.x even though they hard-pin 4.57.x.
#
# demucs builds from a git sdist, whose PEP 517 build isolation fetches
# setuptools from pypi.org — flaky outbound network makes this fail
# intermittently.  Retry the whole install a few times; uv's cache makes each
# retry resume from where the previous left off.
COPY requirements.txt overrides.txt pyproject.toml ./
RUN for i in 1 2 3 4 5; do \
        uv pip install -r requirements.txt --override overrides.txt --system && \
        uv pip install 'gradio>=4.0.0' fastapi 'safetensors>=0.4.3' \
                       'tokenizers>=0.22.0,<=0.23.0' accelerate sentencepiece protobuf \
                       --override overrides.txt --system && \
        exit 0; \
        echo "=== uv install attempt $i failed, retrying in 10s ==="; sleep 10; \
    done; \
    echo "=== uv install failed after 5 attempts ==="; exit 1

# qwen-asr and qwen-tts hard-pin older transformers (4.57.x), so install them
# with --no-deps to keep the transformers 5.x already installed above.  Their
# code runs fine on 5.x via the in-process compat shims in the clients.
RUN uv pip install qwen-asr qwen-tts --no-deps --system

COPY langswap/ ./langswap/
COPY gradio_demo.py main.py serverless.py ./
COPY langswap/__VERSION__ ./langswap/__VERSION__
# The editable build needs setuptools+wheel as its build backend.  Install them
# from apt (no pypi — the network is flaky here) in a layer AFTER the heavy deps
# install so that layer stays cached, then build with --no-build-isolation.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-setuptools python3-wheel \
        && rm -rf /var/lib/apt/lists/*
RUN uv pip install -e . --no-deps --no-build-isolation --system

# Models auto-download into MODEL_WEIGHTS_DIR (=/models) on first use; mount a
# volume there (see the `docker run` command above) so weights persist across
# runs — container- and runpod-friendly. model_config points the HF cache here
# too. Set HF_TOKEN at runtime for gated models. Override any single model with
# LANGSWAP_*_MODEL (a HF repo id or a local path) if you don't want auto-download.
ENV MODEL_WEIGHTS_DIR=/models \
    LANGSWAP_DATA_DIR=/app/data

EXPOSE 7860

CMD ["python", "-u", "gradio_demo.py", "--host", "0.0.0.0", "--port", "7860"]
