"""Plain-Gemma-4 translation client backed by llama-cpp-python + a GGUF model.

Why this is the default translation backend: the dubbing image already carries
vLLM (OmniVoice's vllm-omni needs it).  Running translation through a *second*
vLLM engine would double the CUDA-graph/compile cost and split GPU memory with
OmniVoice (gpu_memory_utilization 0.6 + 0.5 > 1.0 → OOM risk).  llama-cpp-python
is a tiny in-process API that loads a Q4 GGUF in a few seconds, runs Gemma-4-12B
reliably, and offloads to the GPU when one is available.

Plain Gemma-4-12B (instruction-tuned), NOT TranslateGemma — uses the model's own
chat template via create_chat_completion.

Interface matches the other translator clients: load_models() + translate().
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from langswap.model_config import MODEL_WEIGHTS_DIR
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

logger = logging.getLogger(__name__)

# A plain gemma-4-12b-it GGUF that llama.cpp loads directly.
_GGUF_REPO = "unsloth/gemma-4-12b-it-GGUF"
_GGUF_FILE = "gemma-4-12b-it-UD-Q4_K_XL.gguf"


def _lang_name(lang: str) -> str:
    """Best-effort human-readable language name for the prompt."""
    l = (lang or "").strip()
    if not l:
        return l
    if len(l) == 2 and l.isalpha():
        try:
            return map_language_to_code(l.lower(), system="reverse_from_whisper").capitalize()
        except Exception:
            return l
    return l.capitalize()


def _ensure_gguf(model_path: Optional[str]) -> str:
    """Resolve the gemma-4 GGUF path from MODEL_WEIGHTS_DIR; download on first use."""
    if model_path and os.path.exists(model_path):
        return model_path
    local_dir = os.path.join(MODEL_WEIGHTS_DIR, "gemma-4-12b-it-GGUF")
    candidate = os.path.join(local_dir, _GGUF_FILE)
    if os.path.exists(candidate):
        return candidate
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=_GGUF_REPO,
        filename=_GGUF_FILE,
        local_dir=local_dir,
        token=os.environ.get("HF_TOKEN"),
    )


class LlamaCppTranslationClient:
    """Gemma-4-12B translation via llama-cpp-python + GGUF."""

    def __init__(self, device: str = "cuda", model_path: Optional[str] = None):
        self.device = device
        self.model_path = _ensure_gguf(model_path)
        self._llm = None

    def load_models(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `llama-cpp-python` required for LlamaCppTranslationClient. "
                "Install with: pip install llama-cpp-python"
            ) from e

        # n_gpu_layers: -1 offloads the whole model to GPU when on cuda; 0 keeps
        # it on CPU otherwise.
        n_gpu_layers = -1 if self.device.startswith("cuda") else 0
        n_threads = os.cpu_count() or 8

        logger.info("Loading gemma-4-12b GGUF via llama.cpp: %s (n_gpu_layers=%s)",
                    self.model_path, n_gpu_layers)
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            flash_attn=True,
            verbose=False,
        )

    def translate(
        self,
        sentences: List[str],
        source_language: str,
        target_language: str,
        max_new_tokens: int = 512,
    ) -> List[str]:
        if self._llm is None:
            raise RuntimeError("Call load_models() before translate().")

        src = _lang_name(source_language)
        tgt = _lang_name(target_language)

        results: List[str] = []
        for sentence in sentences:
            text = (sentence or "").strip()
            if not text:
                results.append("")
                continue
            prompt = (
                f"Translate the following text from {src} to {tgt}. "
                f"Preserve meaning and tone. Respond with ONLY the translation, "
                f"no quotes, no notes, no explanations.\n\n{text}"
            )
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_new_tokens,
            )
            results.append((out["choices"][0]["message"]["content"] or "").strip())
        logger.info("llama.cpp translated %d sentences (%s → %s)", len(sentences), src, tgt)
        return results
