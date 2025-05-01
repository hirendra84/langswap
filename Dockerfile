FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and upgrade pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && pip3 install --upgrade pip

# Install curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    libpq-dev 

# Sound and media dependencies
RUN apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    gcc wget git

RUN apt-get install -y rubberband-cli

# llama.cpp dependencies for CUDA build
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y build-essential \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && apt-get clean

# Install llama-cpp-python with CUDA
WORKDIR /app
RUN pip install uv
RUN uv init .
RUN export CC=/usr/bin/gcc CXX=/usr/bin/g++
RUN export LD_LIBRARY_PATH=/usr/lib/gcc/$(gcc -dumpmachine)/$(gcc -dumpversion):$LD_LIBRARY_PATH
RUN CMAKE_ARGS="-DGGML_CUDA=on \
            -DCMAKE_CUDA_ARCHITECTURES=75 \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TESTS=OFF" FORCE_CMAKE=1 \
uv pip install --system --upgrade --force-reinstall llama-cpp-python==0.3.8 \
--index-url https://pypi.org/simple \
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 \
--index-strategy unsafe-best-match

# Install Python dependencies (general)
COPY requirements.txt .
RUN uv pip install -r requirements.txt --system

# Clean up
RUN rm -rf /var/lib/apt/lists/*

ENV BASE_WORKING_DIR="/app/data"

COPY . .

CMD ["python3", "-u", "serverless.py"]