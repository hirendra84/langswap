# Build stage
FROM python:3.11-slim 
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# AS builder

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3-dev \
    libpq-dev \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libtbbmalloc2 \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app


# Runtime stage
# FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    netcat-traditional \
    libsndfile1 \
    rubberband-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY freeze.txt .
RUN pip install --no-deps --no-cache-dir -r freeze.txt


# COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# COPY --from=builder /usr/local/bin /usr/local/bin
# COPY data ./data
COPY src ./src
COPY openvoice_conf /openvoice_conf

ENV PYTHONPATH="/app/:${PYTHONPATH}"
ENV LOCAL_DEBUG="True"

ENV COQUI_TOS_AGREED="1"
ENV OPENVOICE_CONF_DIR="/openvoice_conf"
ENV BASE_WORKING_DIR="/app/data"

CMD python src/local_runner.py

