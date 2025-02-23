# Video Translation Pipeline

This project implements a serverless video translation pipeline designed for deployment on Runpod. It processes an input video by extracting its audio, transcribing the speech, translating the text, synthesizing dubbed audio, and then reassembling the video with the new audio track. The system leverages modern machine learning models for speech-to-text, translation, and text-to-speech—and it runs in a Docker container with NVIDIA CUDA support for accelerated inference.

## Features

- **Serverless Handler:** The entry point is in `serverless.py`, which integrates with Runpod for scalable, event-driven processing.
- **Remote File Management:** Supports S3-compatible storage (e.g., Yandex Cloud) to download the source video and upload the translated output.
- **Dockerized Deployment:** The provided `Dockerfile` builds a container image with all necessary dependencies and GPU support.
  
## Getting Started

### Prerequisites

- Docker  
- A Runpod account (for serverless deployment)
- Environment variables for AWS credentials (or compatible storage) set via a `.env` file
- A S3-compatible storage bucket (e.g., Yandex Cloud)

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

### Deploying with Runpod

The project is intended for serverless deployment on Runpod. To deploy:

1. **Build the Docker Image**

   Before building the image, you need to download all model weights and put them in the `model_weights` folder. Otherwise, they would be downloaded during the first run of the service which isn't acceptable for the serverless deployment.


   Build the Docker container using the provided Dockerfile:

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
- **src/translation_pipeline:** Contains the video translation logic.
- **src/file_repository:** Manages remote and local file interactions.
- **src/ml:** Contains the machine learning models and logic.
- Other modules support speech-to-text, text-to-speech, and language translation.

## License

All rights reserved by Peace Data Inc, 2025.