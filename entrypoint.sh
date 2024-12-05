#!/bin/bash
set -e

if [ "$1" = "demo" ]; then
    python3 gradio_demo.py
elif [ "$1" = "bash" ]; then
    exec /bin/bash
else
    exec "$@"
fi