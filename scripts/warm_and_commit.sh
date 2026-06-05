#!/bin/bash
# Run the warm/validate pass in a GPU container on the freshly built base image,
# then `docker commit` the result so the downloaded public models (sherpa-onnx
# ASR, demucs htdemucs) and the warmed vLLM/torch.compile caches under /models
# are baked into the final image.
#
# The warm GPU MUST match the RunPod worker's GPU arch (compile/cudagraph caches
# are sm-version specific) for the warming to pay off; mismatched arch just
# recompiles at runtime (no breakage, lost speedup).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_TAG=${BASE_TAG:-langswap/video-translation-pipeline:4.0-base}
FINAL_TAG=${FINAL_TAG:-langswap/video-translation-pipeline:4.0}
GPU=${GPU:-0}
NAME=langswap-warm

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "== warm/validate run on GPU $GPU =="
set +e
docker run --name "$NAME" --gpus "\"device=$GPU\"" \
    -v "$PWD/scripts:/warm:ro" \
    "$BASE_TAG" python -u /warm/warm_validate.py
warm_rc=$?
set -e
echo "== warm run exit code: $warm_rc =="

if [ "$warm_rc" -ne 0 ]; then
    echo "!! warm/validate failed (critical engine). NOT committing. Inspect with:"
    echo "   docker logs $NAME"
    exit "$warm_rc"
fi

echo "== committing warmed container -> $FINAL_TAG =="
# Re-assert the serverless CMD so the warm command does not become the image CMD.
docker commit \
    --change 'CMD ["python", "-u", "serverless.py"]' \
    "$NAME" "$FINAL_TAG"
docker tag "$FINAL_TAG" langswap/video-translation-pipeline:latest
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "== done. final image: $FINAL_TAG (+ :latest) =="
docker images langswap/video-translation-pipeline
