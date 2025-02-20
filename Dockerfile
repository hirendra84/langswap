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
RUN pip install gradio
RUN pip install torchdiffeq
RUN pip install x_transformers
RUN pip install wandb
RUN pip install ema_pytorch
RUN pip install datasets
RUN pip install vocos
RUN pip install cached_path
RUN pip install elevenlabs
RUN pip install gradio_log
RUN pip install deepl
RUN pip install runpod

RUN rm -rf /var/lib/apt/lists/*



ENV PYTHONPATH="/app/:${PYTHONPATH}"
ENV LOCAL_DEBUG="True"
ENV COQUI_TOS_AGREED="1"
ENV OPENVOICE_CONF_DIR="/openvoice_conf"
ENV BASE_WORKING_DIR="/app/data"

COPY . .

#EXPOSE 4444
#COPY entrypoint.sh /entrypoint.sh
#RUN chmod +x /entrypoint.sh

CMD ["python3", "-u", "serverless.py"]
#ENTRYPOINT ["/entrypoint.sh"]
