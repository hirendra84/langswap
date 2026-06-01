#!/bin/bash
set -e

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

# Read version
tag=$(cat langswap/__VERSION__ | tr -d '\n')
echo "Using version: $tag"

# # Run integration tests (local mode)
# echo "Running tests..."
# python3 scripts/runpod_integration.py --local

echo "Building Docker image..."
docker build -t langswap/video-translation-pipeline:$tag .

echo "Running Docker image..."
docker run --gpus "device=1" langswap/video-translation-pipeline:$tag

echo "Pushing Docker image..."
docker push langswap/video-translation-pipeline:$tag

echo "Done!"
