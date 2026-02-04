#!/bin/bash
set -e

# Read version
tag=$(cat langswap/__VERSION__ | tr -d '\n')
echo "Using version: $tag"

# # Run tests (local mode)
# echo "Running tests..."
# python3 tests/runpod.py --local

echo "Building Docker image..."
docker build -t langswap/video-translation-pipeline:$tag .

echo "Running Docker image..."
docker run --gpus "device=1" langswap/video-translation-pipeline:$tag

echo "Pushing Docker image..."
docker push langswap/video-translation-pipeline:$tag

echo "Done!"
