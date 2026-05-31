# Isolated container for qwen-asr 0.0.6.
#
# Uses the PyTorch base image so torch is preinstalled at a version compatible
# with the transformers==4.57.6 that qwen-asr hard-pins.  This sidesteps the
# transformers-5.x incompatibilities the main app would otherwise run into.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        sox \
        curl \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY server.py ./

ENV HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
