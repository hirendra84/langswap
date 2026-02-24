"""
Model Configuration

This module sets up the model cache directory and environment variables.
Models are automatically downloaded on first use.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import the downloader to set up cache environment
from langswap.model_downloader import setup_cache_environment, get_default_cache_dir

# Get or create the model weights directory
MODEL_WEIGHTS_DIR = os.environ.get('MODEL_WEIGHTS_DIR')

if MODEL_WEIGHTS_DIR is None:
    # Use default cache directory
    MODEL_WEIGHTS_DIR = str(get_default_cache_dir())

# Set up cache environment (creates directory and sets env vars)
MODEL_WEIGHTS_DIR = str(setup_cache_environment(Path(MODEL_WEIGHTS_DIR)))

# Export for other modules
__all__ = ['MODEL_WEIGHTS_DIR']
