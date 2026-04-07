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

# Install Python dependencies
COPY requirements.txt overrides.txt ./
RUN uv pip install -r requirements.txt --override overrides.txt --system

# Copy source and install the package
COPY langswap/ ./langswap/
COPY main.py serverless.py pyproject.toml ./
RUN uv pip install -e . --no-deps --system

# ── Deploy-stack model weights ─────────────────────────────────────────────
# Only the four models needed for the production stack are baked in.
#
# Before building, download OmniVoice if you haven't yet:
#   huggingface-cli download k2-fsa/OmniVoice --local-dir models_weights/OmniVoice
#
# The other three use the HF hub cache layout written by HUGGINGFACE_HUB_CACHE:
#   models_weights/models--{owner}--{name}

COPY models_weights/models--Qwen--Qwen3-ASR-0.6B          ./models_weights/models--Qwen--Qwen3-ASR-0.6B
COPY models_weights/models--Qwen--Qwen3-ForcedAligner-0.6B ./models_weights/models--Qwen--Qwen3-ForcedAligner-0.6B
COPY models_weights/models--google--translategemma-4b-it   ./models_weights/models--google--translategemma-4b-it
COPY models_weights/OmniVoice                              ./models_weights/OmniVoice

# ── Environment ────────────────────────────────────────────────────────────
ENV MODEL_WEIGHTS_DIR="/app/models_weights"
ENV HUGGINGFACE_HUB_CACHE="/app/models_weights"
ENV HF_HOME="/app/models_weights"
ENV TRANSFORMERS_CACHE="/app/models_weights"
ENV TORCH_HOME="/app/models_weights"
ENV BASE_WORKING_DIR="/app/data"
# Tells OmniVoice client to use the baked-in weights directly (no download).
ENV LANGSWAP_OMNIVOICE_MODEL="/app/models_weights/OmniVoice"

CMD ["python3", "-u", "serverless.py"]
