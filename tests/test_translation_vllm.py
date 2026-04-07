"""
Unit tests for VLLMTranslationClient.

vllm, vllm.LLM, and transformers.AutoTokenizer are all mocked so this
runs on Mac without a GPU or vllm installed.
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────


def _install_fake_vllm(monkeypatch):
    """Inject a minimal fake vllm module into sys.modules."""
    captured = {}

    class FakeSamplingParams:
        def __init__(self, temperature=0.0, max_tokens=512):
            captured["sampling_params"] = {"temperature": temperature, "max_tokens": max_tokens}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

        def generate(self, prompts, sampling_params):
            captured["prompts"] = prompts
            captured["generate_sampling_params"] = sampling_params
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text=f"translated: {p.split('<model>')[0].strip()}")])
                for p in prompts
            ]

    vllm_module = types.ModuleType("vllm")
    vllm_module.LLM = FakeLLM
    vllm_module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    return captured, FakeLLM, FakeSamplingParams


def _install_fake_tokenizer(monkeypatch):
    """Inject a fake AutoTokenizer that returns a simple prompt string."""

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            content = messages[0]["content"][0]
            return (
                f"<model>{content['source_lang_code']}->{content['target_lang_code']}: "
                f"{content['text']}"
            )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=FakeTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return FakeTokenizer


def _make_client(monkeypatch, tmp_path):
    """Create a VLLMTranslationClient with faked dependencies."""
    captured, FakeLLM, FakeSamplingParams = _install_fake_vllm(monkeypatch)
    _install_fake_tokenizer(monkeypatch)

    gguf_file = tmp_path / "model.gguf"
    gguf_file.touch()
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir()

    # Patch model_downloader to return our fake paths
    import langswap.model_downloader as md
    monkeypatch.setattr(md, "ensure_translategemma_gguf_model", lambda p=None: gguf_file)
    monkeypatch.setattr(md, "ensure_translategemma_tokenizer", lambda p=None: tok_dir)

    from langswap.ml.translation_service.translator_vllm_client import VLLMTranslationClient

    client = VLLMTranslationClient(device="cpu")
    return client, captured, FakeLLM, FakeSamplingParams


# ── Tests ──────────────────────────────────────────────────────────────────


def test_lazy_load(monkeypatch, tmp_path):
    """LLM must not be loaded during __init__."""
    _install_fake_vllm(monkeypatch)
    _install_fake_tokenizer(monkeypatch)

    gguf_file = tmp_path / "model.gguf"
    gguf_file.touch()
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()

    import langswap.model_downloader as md
    monkeypatch.setattr(md, "ensure_translategemma_gguf_model", lambda p=None: gguf_file)
    monkeypatch.setattr(md, "ensure_translategemma_tokenizer", lambda p=None: tok_dir)

    from langswap.ml.translation_service.translator_vllm_client import VLLMTranslationClient
    client = VLLMTranslationClient(device="cpu")
    assert client._llm is None
    assert client._tokenizer is None


def test_load_models_initialises_llm(monkeypatch, tmp_path):
    client, captured, FakeLLM, _ = _make_client(monkeypatch, tmp_path)
    client.load_models()
    assert client._llm is not None
    assert client._tokenizer is not None
    assert "model" in captured["llm_kwargs"]


def test_load_models_idempotent(monkeypatch, tmp_path):
    """Calling load_models() twice must not reinitialise the LLM."""
    client, captured, FakeLLM, _ = _make_client(monkeypatch, tmp_path)
    client.load_models()
    first_llm = client._llm
    client.load_models()
    assert client._llm is first_llm


def test_translate_before_load_raises(monkeypatch, tmp_path):
    client, _, _, _ = _make_client(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="load_models"):
        client.translate(["hello"], "en", "ru")


def test_translate_basic(monkeypatch, tmp_path):
    client, captured, _, _ = _make_client(monkeypatch, tmp_path)
    client.load_models()

    results = client.translate(["Hello world", "Good morning"], "en", "ru")
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_translate_batches_all_at_once(monkeypatch, tmp_path):
    """All sentences must be sent in a single llm.generate() call."""
    client, captured, _, _ = _make_client(monkeypatch, tmp_path)
    client.load_models()

    sentences = ["one", "two", "three"]
    client.translate(sentences, "en", "ru")
    assert len(captured["prompts"]) == 3


def test_translate_greedy_sampling(monkeypatch, tmp_path):
    """Sampling must use temperature=0 (greedy decoding)."""
    client, captured, _, _ = _make_client(monkeypatch, tmp_path)
    client.load_models()
    client.translate(["test"], "en", "ru")
    assert captured["sampling_params"]["temperature"] == 0.0


def test_translate_lang_code_mapping(monkeypatch, tmp_path):
    """Full language names must be mapped to 2-letter codes."""
    client, captured, _, _ = _make_client(monkeypatch, tmp_path)
    client.load_models()
    client.translate(["hello"], "english", "russian")
    # The fake tokenizer encodes the codes into the prompt string
    prompt = captured["prompts"][0]
    assert "en->ru" in prompt


def test_translate_two_letter_codes_pass_through(monkeypatch, tmp_path):
    """2-letter codes must not be remapped."""
    client, captured, _, _ = _make_client(monkeypatch, tmp_path)
    client.load_models()
    client.translate(["hello"], "en", "de")
    prompt = captured["prompts"][0]
    assert "en->de" in prompt


def test_translate_empty_list(monkeypatch, tmp_path):
    client, _, _, _ = _make_client(monkeypatch, tmp_path)
    client.load_models()
    results = client.translate([], "en", "ru")
    assert results == []


def test_cpu_mode_sets_enforce_eager(monkeypatch, tmp_path):
    """CPU / Mac fallback must use float32 and enforce_eager."""
    client, captured, _, _ = _make_client(monkeypatch, tmp_path)
    client.load_models()
    assert captured["llm_kwargs"].get("dtype") == "float32"
    assert captured["llm_kwargs"].get("enforce_eager") is True
