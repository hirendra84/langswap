# Build stage
FROM python:3.11-slim AS builder

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

COPY freeze.txt .
RUN pip install --no-deps --no-cache-dir -r freeze.txt

# Runtime stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY data ./data

ENV PYTHONPATH="/app/:${PYTHONPATH}"

CMD ["python", "src/local_runner.py"]