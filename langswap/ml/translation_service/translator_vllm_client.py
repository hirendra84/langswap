"""
TranslateGemma translation client backed by vLLM offline inference.

Uses a GGUF-quantised model (Q4_K_M by default) so no HF-transformers model
loading is needed — only the tokenizer is loaded via transformers for
`apply_chat_template`.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from tqdm import tqdm

from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code
from langswap.model_downloader import ensure_translategemma_gguf_model, ensure_translategemma_tokenizer

logger = logging.getLogger(__name__)


class VLLMTranslationClient:
    """
    TranslateGemma via vLLM + GGUF quantised weights.

    All sentences are batched into a single vLLM call for throughput.
    Falls back to CPU (float32, enforce_eager) on non-CUDA hardware so
    local Mac debugging still works.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model_path = str(ensure_translategemma_gguf_model(model_path))
        self.tokenizer_path = str(ensure_translategemma_tokenizer(tokenizer_path))
        self.device = device
        self._llm = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self):
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams  # noqa: F401 — just validate importable
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `vllm` required for VLLMTranslationClient. "
                "Install with: pip install vllm"
            ) from e

        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `transformers` required for tokenizer. "
                "Install with: pip install transformers"
            ) from e

        import torch

        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        kwargs = dict(
            model=self.model_path,
            tokenizer=self.tokenizer_path,
            # GGUF models need trust_remote_code for the Gemma architecture
            trust_remote_code=True,
            # Limit KV-cache so the whole 4B model fits on smaller GPUs
            max_model_len=4096,
        )

        if is_cuda:
            kwargs.update(dtype="bfloat16", gpu_memory_utilization=0.6)
        else:
            # CPU / MPS fallback: float32, no CUDA graphs
            kwargs.update(dtype="float32", enforce_eager=True)
            # Tell vLLM to use the CPU backend
            os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

        from vllm import LLM
        self._llm = LLM(**kwargs)

        # Load tokenizer separately — only used for apply_chat_template.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            local_files_only=True,
        )

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate(
        self,
        sentences: List[str],
        source_language: str,
        target_language: str,
        max_new_tokens: int = 512,
    ) -> List[str]:
        if self._llm is None or self._tokenizer is None:
            raise RuntimeError("Call load_models() before translate().")

        from vllm import SamplingParams

        def _to_lang_code(lang: str) -> str:
            l = (lang or "").strip()
            if len(l) == 2 and l.isalpha():
                return l.lower()
            try:
                return map_language_to_code(l.lower(), system="whisper")
            except Exception:
                return l

        src = _to_lang_code(source_language)
        tgt = _to_lang_code(target_language)

        # Build all prompts — TranslateGemma expects a special content format.
        prompts: List[str] = []
        for sentence in sentences:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": src,
                            "target_lang_code": tgt,
                            "text": (sentence or "").strip(),
                        }
                    ],
                }
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        sampling = SamplingParams(
            temperature=0.0,   # greedy — translation doesn't benefit from sampling
            max_tokens=max_new_tokens,
        )

        # vLLM batches all prompts in one call — much faster than per-sentence loops.
        logger.info("vLLM translating %d sentences (%s → %s)", len(prompts), src, tgt)
        outputs = self._llm.generate(prompts, sampling)

        return [o.outputs[0].text.strip() for o in outputs]
