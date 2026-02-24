"""LangSwap - Video translation pipeline package."""
from pathlib import Path

# Read version from __VERSION__ file
__version__ = (Path(__file__).parent / '__VERSION__').read_text().strip()

# Optional imports: keep package importable even when heavy ML deps aren't installed.
__all__ = ["__version__"]

# Model download utilities - always available
try:
    from langswap.model_downloader import (
        download_all_models,
        ensure_model,
        list_available_models,
    )
    __all__ += ["download_all_models", "ensure_model", "list_available_models"]
except Exception:
    pass

try:
    # Import main API functions
    from langswap.api import process_translation, init_s3_client, get_file

    __all__ += ["process_translation", "init_s3_client", "get_file"]
except Exception:
    # Missing optional dependencies (e.g., whisperx) shouldn't prevent importing `langswap`.
    pass

try:
    # Import core components
    from langswap.translation_pipeline import VideoTranslationPipeline, ChangeManager
    from langswap.pipeline_models.models import (
        TranslationPipelineConfig,
        load_config_from_json,
        save_config_to_json,
    )
    from langswap.file_repository import RemoteFile, RemoteFileRepository

    __all__ += [
        "VideoTranslationPipeline",
        "ChangeManager",
        "TranslationPipelineConfig",
        "load_config_from_json",
        "save_config_to_json",
        "RemoteFile",
        "RemoteFileRepository",
    ]
except Exception:
    pass
