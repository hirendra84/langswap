FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    gcc \
    wget \
    git \
    build-essential \
    libpq-dev \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    rubberband-cli \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip uv

WORKDIR /app

# Install Python dependencies.
# vllm ships pre-built CUDA wheels — no custom compilation needed.
# qwen-asr and qwen-tts are installed with --no-deps because they hard-pin
# older transformers versions that conflict with vllm/vllm-omni (needs 5.x).
COPY requirements.txt overrides.txt ./
RUN uv pip install -r requirements.txt --override overrides.txt --system && \
    uv pip install qwen-asr --no-deps --system

# Copy source and install the package
COPY langswap/ ./langswap/
COPY main.py serverless.py pyproject.toml ./
RUN uv pip install -e . --no-deps --system

# ── Deploy-stack model weights ─────────────────────────────────────────────
# Models baked into the image for the production stack:
#   ASR:         Qwen3-ASR-0.6B   (qwen-asr)
#   Aligner:     Qwen3-ForcedAligner-0.6B  (loaded by qwen-asr internally)
#   Translation: translategemma-4b-it Q4_K_M GGUF  (vLLM offline)
#   TTS:         OmniVoice  (vllm-omni offline)
#
# Pre-download before building:
#   huggingface-cli download mradermacher/translategemma-4b-it-GGUF \
#     --include "translategemma-4b-it.Q4_K_M.gguf" \
#     --local-dir models_weights/translategemma-4b-it-GGUF
#   huggingface-cli download k2-fsa/OmniVoice --local-dir models_weights/OmniVoice
#   # Tokenizer only (no safetensors):
#   huggingface-cli download google/translategemma-4b-it \
#     --ignore-patterns "*.safetensors" "*.bin" \
#     --local-dir models_weights/translategemma-4b-it-tokenizer

COPY models_weights/models--Qwen--Qwen3-ASR-0.6B           ./models_weights/models--Qwen--Qwen3-ASR-0.6B
COPY models_weights/models--Qwen--Qwen3-ForcedAligner-0.6B ./models_weights/models--Qwen--Qwen3-ForcedAligner-0.6B
COPY models_weights/translategemma-4b-it-GGUF               ./models_weights/translategemma-4b-it-GGUF
COPY models_weights/translategemma-4b-it-tokenizer          ./models_weights/translategemma-4b-it-tokenizer
COPY models_weights/OmniVoice                               ./models_weights/OmniVoice

# ── Environment ────────────────────────────────────────────────────────────
ENV MODEL_WEIGHTS_DIR="/app/models_weights"
ENV HUGGINGFACE_HUB_CACHE="/app/models_weights"
ENV HF_HOME="/app/models_weights"
ENV TRANSFORMERS_CACHE="/app/models_weights"
ENV TORCH_HOME="/app/models_weights"
ENV BASE_WORKING_DIR="/app/data"
# Point each client at its baked-in weights (no runtime downloads).
ENV LANGSWAP_OMNIVOICE_MODEL="/app/models_weights/OmniVoice"
ENV LANGSWAP_TRANSLATEGEMMA_GGUF="/app/models_weights/translategemma-4b-it-GGUF/translategemma-4b-it.Q4_K_M.gguf"
ENV LANGSWAP_TRANSLATEGEMMA_TOKENIZER="/app/models_weights/translategemma-4b-it-tokenizer"

CMD ["python3", "-u", "serverless.py"]
