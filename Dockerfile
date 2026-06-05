# RunPod serverless image: the container runs the RunPod handler (serverless.py),
# not the Gradio UI.  Model weights live in MODEL_WEIGHTS_DIR (a mounted volume),
# auto-downloaded on first use if absent — they are not baked into image layers.
#
# Build:         docker build -t langswap/video-translation-pipeline:4.0-base .
# Warm the public models / vLLM + torch.compile caches and commit the final
# image:         scripts/warm_and_commit.sh
#
# There is no web server — the worker pulls jobs from the RunPod queue. To smoke
# it locally:    docker run --rm --gpus all <image>   # boots the handler

FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
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


RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python3 -m pip install --break-system-packages \
            --timeout 30 --retries 10 uv 

ENV UV_PYTHON=/usr/bin/python3 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_HTTP_TIMEOUT=180

WORKDIR /app

# Single source of truth for dependencies: pyproject.toml + uv. The ".[gpu]"
# extra pulls the whole local stack — torch/torchaudio/torchcodec from the CUDA 13
# index and llama-cpp-python from its prebuilt CPU wheel (both wired in
# pyproject's [tool.uv]); transformers/vllm/vllm-omni (OmniVoice), faster-whisper
# (VAD ASR), silero-vad, and the qwen-omni-utils OmniVoice runtime helper. The
# "runpod" extra adds the serverless worker SDK (serverless.py).
#
# nvidia-cublas-cu12 (in the gpu extra): this is a CUDA-13 image (cu130 torch/
# vLLM/OmniVoice), but ctranslate2 (faster-whisper's backend) links cuBLAS *12*
# (libcublas.so.12) — absent here, so GPU ASR would crash with "Library
# libcublas.so.12 is not found". cuDNN 9 is already present (cu13, used by torch)
# and is ABI-compatible, so we add ONLY cuBLAS-12 — NOT nvidia-cudnn-cu12, which
# would overwrite the cu13 cuDNN torch/vLLM rely on. The cu12 and cu13 cuBLAS
# coexist in nvidia/cublas/lib (distinct .so.12/.so.13); ctranslate2's RPATH
# ($ORIGIN/../nvidia/cublas/lib) finds .so.12 with no LD_LIBRARY_PATH needed.
COPY pyproject.toml README.md ./
COPY langswap/ ./langswap/
COPY gradio_demo.py main.py serverless.py ./

# Reinstall pip/setuptools/wheel via uv so they carry pip metadata (RECORD);
# the apt-packaged versions lack it and can't be upgraded/uninstalled later.
RUN uv pip install pip setuptools wheel --system
RUN uv pip install -e ".[gpu,runpod]" --system

# MODEL_WEIGHTS_DIR should be a mounted volume holding the model weights (OmniVoice,
# faster-whisper large-v3, the Gemma-4-12B GGUF under gemma-4-12b-it-GGUF/, pyannote
# configs).  Anything absent is auto-downloaded into it on first use.
ENV MODEL_WEIGHTS_DIR=/app/models_weights \
    LANGSWAP_DATA_DIR=/app/data
# ASR runs on GPU (faster-whisper large-v3, float16).  The only missing piece on
# this CUDA-13 image was cuBLAS-12, added above (nvidia-cublas-cu12); cuDNN 9 is
# already present.

# No EXPOSE: the RunPod serverless worker pulls jobs from the queue and serves no
# HTTP port (the old EXPOSE 7860 belonged to the Gradio UI).
#
# RunPod serverless handler — NOT the Gradio UI.  Running gradio_demo.py here is
# why jobs previously sat in_queue: no handler ever registered with RunPod.
CMD ["python", "-u", "serverless.py"]
