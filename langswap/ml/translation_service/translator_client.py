from __future__ import annotations

import os
from abc import ABC
from typing import List, Optional

from tqdm import tqdm

from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code
from langswap.model_downloader import ensure_translategemma_model

class TranslatorClient(ABC):

    def __init__(self, device: str):
        ...

    def translate(self, sentences: List[str], source_lang: str, target_lang: str, context: List[str]) -> list[str]:
        ...


class LLMTranslationClient(TranslatorClient):
    """
    TranslateGemma via Hugging Face Transformers.
    Uses the model's chat template (recommended by the model card).
    Models are automatically downloaded on first use.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
    ):
        super().__init__(device)
        # Auto-download model if not present
        self.model_path = str(ensure_translategemma_model(model_path))
        self.model = None
        self.tokenizer = None

    def load_models(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,  # Model already downloaded by ensure_translategemma_model
        )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        device_map = "auto" if torch.cuda.is_available() else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,  # Model already downloaded by ensure_translategemma_model
        )

        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def translate(
        self,
        sentences: List[str],
        source_language: str,
        target_language: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
    ) -> list[str]:
        import torch
        import torch._dynamo

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLMTranslationClient.load_models() must be called before translate().")

        def _to_lang_code(lang: str) -> str:
            l = (lang or "").strip()
            if len(l) == 2 and l.isalpha():
                return l.lower()
            try:
                return map_language_to_code(l.lower(), system="whisper")
            except Exception:
                return l

        source_code = _to_lang_code(source_language)
        target_code = _to_lang_code(target_language)

        translations: list[str] = []
        for sentence in tqdm(sentences):
            safe_sentence = (sentence or "").strip()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_code,
                            "target_lang_code": target_code,
                            "text": safe_sentence,
                        }
                    ],
                }
            ]

            # TranslateGemma is designed around this chat template.
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Disable TorchDynamo tracing/compilation for generation.
                # accelerate hooks use hasattr(...) and can trip Dynamo ("hasattr ConstDictVariable to").
                #
                # torch._dynamo.disable(fn) is sometimes not enough if Dynamo is already enabled upstream,
                # so we also flip the local config flags for this call.
                prev_disable = torch._dynamo.config.disable
                prev_suppress = torch._dynamo.config.suppress_errors
                torch._dynamo.config.disable = True
                torch._dynamo.config.suppress_errors = True
                try:
                    out = self.model.generate(
                        **inputs,
                        do_sample=temperature > 0,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                        pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                        **(
                            {"temperature": temperature, "top_p": top_p}
                            if temperature > 0
                            else {}
                        ),
                    )
                finally:
                    torch._dynamo.config.disable = prev_disable
                    torch._dynamo.config.suppress_errors = prev_suppress

            input_len = inputs["input_ids"].shape[-1]
            gen_tokens = out[0, input_len:]
            translations.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

        return translations
    