import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_WEIGHTS_DIR = os.environ.get('MODEL_WEIGHTS_DIR')

# Create the directory if it doesn't exist
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)

# Set Hugging Face cache environment variables
os.environ["TRANSFORMERS_CACHE"] = MODEL_WEIGHTS_DIR
os.environ["HF_HOME"] = MODEL_WEIGHTS_DIR
os.environ["HF_DATASETS_CACHE"] = MODEL_WEIGHTS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_WEIGHTS_DIR

# Set for specific libraries that might use their own cache
os.environ["TORCH_HOME"] = MODEL_WEIGHTS_DIR
os.environ["XDG_CACHE_HOME"] = MODEL_WEIGHTS_DIR 