# Build:  docker build -t langswap:latest .
# Run:    docker run --rm --gpus all -p 7860:7860 \
#           -v "$PWD/models_weights:/models" -v "$PWD/data:/app/data" \
#           langswap:latest
#         # then open http://localhost:7860

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

COPY requirements.txt overrides.txt pyproject.toml ./
RUN uv pip install -r requirements.txt --override overrides.txt --system && \
    uv pip install 'gradio>=4.0.0' fastapi 'safetensors>=0.4.3' \
                       'tokenizers>=0.22.0,<=0.23.0' accelerate sentencepiece protobuf \
                       --override overrides.txt --system

RUN uv pip install qwen-asr qwen-tts --no-deps --system && \
    uv pip install nagisa soynlp librosa qwen-omni-utils runpod sherpa-onnx \
        --override overrides.txt --system

COPY langswap/ ./langswap/
COPY gradio_demo.py main.py serverless.py ./
COPY langswap/__VERSION__ ./langswap/__VERSION__


# Reinstall pip/setuptools/wheel via uv so they carry pip metadata (RECORD);
# the apt-packaged versions lack it and can't be upgraded/uninstalled later.
RUN uv pip install pip setuptools wheel --system
RUN uv pip install -e . --no-deps --no-build-isolation --system

ENV MODEL_WEIGHTS_DIR=/models \
    LANGSWAP_DATA_DIR=/app/data

EXPOSE 7860

CMD ["python", "-u", "gradio_demo.py", "--host", "0.0.0.0", "--port", "7860"]
