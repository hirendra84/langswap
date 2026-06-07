"""Pure tests for TranslationPipelineConfig serialization and defaults."""
import dataclasses

from langswap.pipeline_models.models import (
    TranslationPipelineConfig,
    save_config_to_json,
    load_config_from_json,
)


def _make_config(**overrides):
    base = dict(
        source_lang="english",
        target_lang="russian",
        source_video_path="/tmp/in.mp4",
        base_dir="/tmp/work",
        public_id="abc123",
    )
    base.update(overrides)
    return TranslationPipelineConfig(**base)


def test_config_round_trip(tmp_path):
    """save_config_to_json -> load_config_from_json preserves key fields."""
    config = _make_config(
        num_speakers=2,
        device="cpu",
        name="job",
        asr_backend="openai",
        translation_backend="openai",
    )
    path = tmp_path / "config.json"
    save_config_to_json(config, path)
    loaded = load_config_from_json(path)

    assert loaded.source_lang == config.source_lang
    assert loaded.target_lang == config.target_lang
    assert loaded.public_id == config.public_id
    assert loaded.num_speakers == config.num_speakers
    assert loaded.device == config.device
    assert loaded.asr_backend == config.asr_backend
    assert loaded.translation_backend == config.translation_backend
    # Paths round-trip back to Path objects.
    assert str(loaded.source_video_path) == str(config.source_video_path)
    assert str(loaded.base_dir) == str(config.base_dir)


def test_config_defaults():
    """New production defaults: vad ASR + llamacpp translation."""
    config = _make_config()
    assert config.asr_backend == "vad"
    assert config.translation_backend == "llamacpp"


def test_config_dropped_fields():
    """Removed backends left no stale config fields behind."""
    field_names = {f.name for f in dataclasses.fields(TranslationPipelineConfig)}
    assert "voice_conv" not in field_names
    assert "eleven_api_token" not in field_names
