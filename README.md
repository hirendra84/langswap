# Video Translation Pipeline

This project implements a serverless video translation pipeline designed for deployment on Runpod. It processes an input video by extracting its audio, transcribing the speech, translating the text, synthesizing dubbed audio, and then reassembling the video with the new audio track. The system leverages modern machine learning models for speech-to-text, translation, and text-to-speech—and it runs in a Docker container with NVIDIA CUDA support for accelerated inference.

## Features

- **Serverless Handler:** The entry point is in `serverless.py`, which integrates with Runpod for scalable, event-driven processing.
- **Remote File Management:** Supports S3-compatible storage (e.g., Yandex Cloud) to download the source video and upload the translated output.
- **Dockerized Deployment:** The provided `Dockerfile` builds a container image with all necessary dependencies and GPU support.
  
## Installation

### Quick Start (pip install)

```bash
# Install the package
pip install -e .

# Models are downloaded automatically on first use
# Or pre-download all models:
langswap-download-models --all

# For gated models (pyannote), set your HuggingFace token first:
export HF_TOKEN=your_huggingface_token
langswap-download-models --all
```

### Model Management

Models are cached in platform-appropriate directories:
- **Linux:** `~/.cache/langswap/models`
- **macOS:** `~/Library/Caches/langswap/models`
- **Windows:** `%LOCALAPPDATA%\langswap\models`

You can override this with the `MODEL_WEIGHTS_DIR` environment variable.

```bash
# List available models
langswap-download-models --list

# Download specific models
langswap-download-models --model qwen3-tts
langswap-download-models --model faster-whisper-large-v3
langswap-download-models --model translategemma

# Download to custom directory
langswap-download-models --all --cache-dir /path/to/models
```

### Available Models

| Model | Description | Requires Token |
|-------|-------------|----------------|
| `faster-whisper-large-v3` | WhisperX ASR model | No |
| `qwen3-tts` | Qwen3 TTS with voice cloning | No |
| `xtts-v2` | Coqui XTTS v2 TTS | No |
| `translategemma` | TranslateGemma translation | No |
| `pyannote-speaker-diarization` | Speaker diarization | Yes (HF_TOKEN) |
| `pyannote-segmentation` | Speaker segmentation | Yes (HF_TOKEN) |
| `vocos-mel-24khz` | Vocos vocoder | No |
| `silero-vad` | Voice activity detection | No |

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (for serverless deployment)
- A Runpod account (for serverless deployment)
- Environment variables for AWS credentials (or compatible storage) set via a `.env` file
- A S3-compatible storage bucket (e.g., Yandex Cloud)
- (Optional) HuggingFace token for gated models

### Environment Setup

Create a `.env` file in the project root with your credentials:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
BUCKET="your_S3_bucket_name"
```

### Local Testing

You can test the pipeline locally by running:

```bash
python serverless.py
```

This will run test_input.json as an input without need to deploy the service.

```bash
python tests/runpod.py --local
```

This command will run more comprehensive tests. It will go through every video in test_videos folder on S3.

### Deploying with Runpod

The project is intended for serverless deployment on Runpod. To deploy:

1. **Build the Docker Image**

   Before building the image, download all model weights to include them in the container:

   ```bash
   # Set HF token for gated models
   export HF_TOKEN=your_huggingface_token

   # Download all models
   langswap-download-models --all --cache-dir ./models_weights
   ```

   Then build the Docker container using the provided Dockerfile:

   ```bash
   docker build -t your_dockerhub_username/video-translation-pipeline:latest .
   ```

   Now you can run the container locally:

   ```bash
   docker run --env-file .env --gpus all -it your_dockerhub_username/video-translation-pipeline:latest
   ```

2. **Push to Dockerhub**

   Push your built image to Dockerhub:

   ```bash
   docker push your_dockerhub_username/video-translation-pipeline:latest
   ```

3. **Deploy on Runpod**

   Follow Runpod’s deployment instructions to deploy your container image. The service will start using the `serverless.py` entry point.

## Project Structure

- **serverless.py:** Main entry point implementing the Runpod handler.
- **Dockerfile:** Defines the container environment with dependencies, CUDA support, and the execution command.
- **langswap/translation_pipeline:** Contains the video translation logic.
- **langswap/file_repository:** Manages remote and local file interactions.
- **langswap/ml:** Contains the machine learning models and logic.
- Other modules support speech-to-text, text-to-speech, and language translation.


## License

All rights reserved by Peace Data Inc, 2026.