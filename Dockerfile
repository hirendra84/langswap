# Build stage
# FROM python:3.11-slim AS builder
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

RUN pip install amqp
RUN pip install jmespath
RUN pip install python-dateutil

RUN rm -rf /var/lib/apt/lists/*

# COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# COPY --from=builder /usr/local/bin /usr/local/bin

# now we mount the disk without copying everything
# COPY data ./data
# COPY src ./src
# COPY coqui ./coqui
# COPY whisperX ./whisperX
# COPY models_weights ./models_weights
# COPY resemble ./resemble
# COPY voice_conv ./voice_conv
# COPY pyannote-audio ./pyannote-audio
# COPY pipeline_check.py pipeline_check.py


ENV PYTHONPATH="/app/:${PYTHONPATH}"
ENV LOCAL_DEBUG="True"
ENV COQUI_TOS_AGREED="1"
ENV OPENVOICE_CONF_DIR="/openvoice_conf"
ENV BASE_WORKING_DIR="/app/data"


# previous example of running the code:
# ENV FILE_PATH=./data/shulman.mp4
# ENV BASE_DIR=./data/shulman_example
# ENV SOURCE_LANG=russian
# ENV TARGET_LANG=english
# ENV VIDEO_NAME=shulman
# CMD ["python3", "pipeline_check.py", "--file_path", "$FILE_PATH", "--base_dir", "$BASE_DIR", "--source_lang", "$SOURCE_LANG", "--target_lang", "$TARGET_LANG", "--name", "$VIDEO_NAME"]

# run with entrypoint.sh
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["run"]
