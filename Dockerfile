FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && pip3 install --upgrade pip

# Install curl separately to test
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl

# Install build tools
RUN apt-get install -y --no-install-recommends \
    python3-dev \
    libpq-dev \
    build-essential \
    git

# Install sound and media dependencies
RUN apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev

RUN apt-get update && apt-get install -y rubberband-cli

WORKDIR /app

COPY requirements.txt .
RUN pip install -r  requirements.txt
RUN rm -rf /var/lib/apt/lists/*


ENV BASE_WORKING_DIR="/app/data"

COPY . .

CMD ["python3", "-u", "serverless.py"]
