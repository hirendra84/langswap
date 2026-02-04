"""LangSwap - Video translation pipeline package."""
from pathlib import Path

# Read version from __VERSION__ file
__version__ = (Path(__file__).parent / '__VERSION__').read_text().strip()

# Import main API functions
from langswap.api import (
    process_translation,
    init_s3_client,
    get_file
)

# Import core components
from langswap.translation_pipeline import VideoTranslationPipeline, ChangeManager
from langswap.pipeline_models.models import (
    TranslationPipelineConfig,
    load_config_from_json,
    save_config_to_json
)
from langswap.file_repository import RemoteFile, RemoteFileRepository

__all__ = [
    '__version__',
    'process_translation',
    'init_s3_client',
    'get_file',
    'VideoTranslationPipeline',
    'ChangeManager',
    'TranslationPipelineConfig',
    'load_config_from_json',
    'save_config_to_json',
    'RemoteFile',
    'RemoteFileRepository'
]
