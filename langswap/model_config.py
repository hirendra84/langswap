"""
Model Configuration

Resolves the model weights/cache directory, points every model library at it,
and provides a tiny helper for resolving model ids.  Models are loaded directly
from HuggingFace and auto-downloaded on first use (no custom downloader layer).
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _default_cache_dir() -> Path:
    """Default model weights/cache directory.

    Always the project's ``models_weights/`` folder unless ``MODEL_WEIGHTS_DIR``
    is set.  Keeping weights inside the project tree makes builds, containers,
    and runpod volumes predictable and self-contained.
    """
    env_dir = os.environ.get("MODEL_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir)
    # langswap/model_config.py -> project root is two parents up.
    return Path(__file__).resolve().parent.parent / "models_weights"


# Resolve the weights/cache directory and point every model library at it so
# HuggingFace / torch.hub downloads all land in the project's models_weights/.
MODEL_WEIGHTS_DIR = str(_default_cache_dir())
Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
for _var in (
    "MODEL_WEIGHTS_DIR",
    "TRANSFORMERS_CACHE",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TORCH_HOME",
):
    os.environ[_var] = MODEL_WEIGHTS_DIR


def resolve_model(env_var: str, default_repo_id: str, explicit: Optional[str] = None) -> str:
    """Resolve a model id/path to hand straight to a loader.

    Precedence: ``explicit`` arg → ``$env_var`` → ``default_repo_id``.  The
    default is a HuggingFace repo id, which loaders auto-download into
    ``MODEL_WEIGHTS_DIR`` on first use.  A local path is returned unchanged.
    """
    return explicit or os.environ.get(env_var) or default_repo_id


__all__ = ["MODEL_WEIGHTS_DIR", "resolve_model"]
