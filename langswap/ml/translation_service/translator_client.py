from __future__ import annotations

from abc import ABC
from typing import List, Optional

from tqdm import tqdm

from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code
from langswap.model_config import resolve_model


_LANG_CODE_TO_NAME = {
    "en": "English", "ru": "Russian", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "pl": "Polish",
    "nl": "Dutch", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "uk": "Ukrainian",
    "cs": "Czech", "sv": "Swedish", "fi": "Finnish", "no": "Norwegian",
    "da": "Danish", "el": "Greek", "he": "Hebrew", "id": "Indonesian",
    "vi": "Vietnamese", "th": "Thai", "ro": "Romanian", "hu": "Hungarian",
}


def _lang_full_name(code: str) -> str:
    if not code:
        return "the target language"
    return _LANG_CODE_TO_NAME.get(code.lower(), code)


class TranslatorClient(ABC):

    def __init__(self, device: str):
        ...

    def translate(self, sentences: List[str], source_lang: str, target_lang: str, context: List[str]) -> list[str]:
        ...


class LLMTranslationClient(TranslatorClient):
    """
    Generic HuggingFace Transformers translation client.

    Supports two prompt styles:
    - ``translategemma`` — uses the TranslateGemma-specific chat template
      that expects ``source_lang_code`` / ``target_lang_code`` keys.
    - ``instruction`` — plain instruction prompt suitable for generic
      instruction-tuned models (e.g. Gemma-4 / Gemma-3n).
    - ``auto`` (default) — picks ``translategemma`` if the model path
      contains "translategemma", otherwise ``instruction``.

    Models are auto-downloaded on first use.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        prompt_style: str = "auto",
    ):
        super().__init__(device)
        self.model_path = resolve_model(
            "LANGSWAP_TRANSLATEGEMMA_MODEL", "google/translategemma-4b-it", model_path)
        self.model = None
        self.tokenizer = None
        # For Gemma-4 (and other multimodal models) the chat template lives on
        # the processor, not the tokenizer.  When present we prefer it.
        self.processor = None
        self.prompt_style = self._resolve_prompt_style(prompt_style)

    def _resolve_prompt_style(self, style: str) -> str:
        if style != "auto":
            return style
        return "translategemma" if "translategemma" in self.model_path.lower() else "instruction"

    def load_models(self):
        # Idempotent: when this client is reused across jobs (warm pool) the
        # manager calls load_models() again — don't reload the weights.
        if self.model is not None and self.tokenizer is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=False,  # auto-download from HF into models_weights on first use
        )

        # Prefer the processor's chat template when it carries one (Gemma-4 keeps
        # the chat template on the processor, not the tokenizer).
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            if getattr(processor, "chat_template", None):
                self.processor = processor
        except Exception:
            self.processor = None

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        device_map = "auto" if torch.cuda.is_available() else None

        # Multimodal Gemma checkpoints (e.g. gemma-3-4b-it /
        # Gemma{3,4}ForConditionalGeneration) nest the language model under a
        # vision-text wrapper, so AutoModelForCausalLM would mismatch the weight
        # prefixes.  Pick the image-text-to-text class for those and the plain
        # causal-LM class for text-only Gemmas (e.g. gemma-2-2b-it).
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        architectures = getattr(config, "architectures", None) or []
        is_multimodal = any(
            ("ConditionalGeneration" in a) or ("ImageTextToText" in a) or ("MultimodalLM" in a)
            for a in architectures
        )
        model_cls = AutoModelForCausalLM
        if is_multimodal:
            try:
                from transformers import AutoModelForImageTextToText

                model_cls = AutoModelForImageTextToText
            except Exception:
                model_cls = AutoModelForCausalLM

        self.model = model_cls.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=False,  # auto-download from HF into models_weights on first use
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

        # Build the stop-token set.  Gemma instruction models terminate a turn
        # with <end_of_turn>, NOT the base <eos>.  The previous code passed only
        # tokenizer.eos_token_id, which overrode the model's generation_config and
        # meant generation never stopped — every segment ran to max_new_tokens
        # (512), ~35s of wasted decode per sentence (and trailing garbage after
        # the real translation).  Stop on both <eos> and <end_of_turn>.
        eos_ids: list[int] = []
        _base_eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(_base_eos, int):
            eos_ids.append(_base_eos)
        try:
            _eot = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if isinstance(_eot, int) and _eot >= 0 and _eot != getattr(self.tokenizer, "unk_token_id", None):
                eos_ids.append(_eot)
        except Exception:
            pass
        eos_ids = list(dict.fromkeys(eos_ids)) or None

        translations: list[str] = []
        for _seg_idx, sentence in enumerate(tqdm(sentences)):
            import time as _t
            _seg_t0 = _t.perf_counter()
            safe_sentence = (sentence or "").strip()

            if self.prompt_style == "translategemma":
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
            else:
                src_full = _lang_full_name(source_code)
                tgt_full = _lang_full_name(target_code)
                user_prompt = (
                    f"Translate the following text from {src_full} to {tgt_full}. "
                    "Reply ONLY with the translation, no preamble or quotes.\n\n"
                    f"{safe_sentence}"
                )
                # Multimodal Gemma-3/4 processors expect structured content
                # (a list of typed parts); text-only tokenizers expect a plain
                # string.  Match the format to whichever encoder we'll use below.
                if self.processor is not None:
                    content = [{"type": "text", "text": user_prompt}]
                else:
                    content = user_prompt
                messages = [{"role": "user", "content": content}]

            # Gemma-4 keeps the chat template on the processor; fall back to the
            # tokenizer for models (e.g. TranslateGemma) that template there.
            chat_encoder = self.processor or self.tokenizer
            template_kwargs = {}
            if self.prompt_style != "translategemma":
                # Suppress the reasoning block on instruction-tuned Gemma-4 so the
                # output is the bare translation.  Harmless for templates that
                # don't use the flag.
                template_kwargs["enable_thinking"] = False
            inputs = chat_encoder.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **template_kwargs,
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
                        eos_token_id=eos_ids,
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
            _n_gen = int(gen_tokens.shape[-1])
            _hit_cap = _n_gen >= max_new_tokens
            print(
                f"[timing] translate seg {_seg_idx}: {_n_gen} tok in "
                f"{_t.perf_counter() - _seg_t0:.1f}s"
                + (" <<< HIT max_new_tokens CAP (no EOS!)" if _hit_cap else "")
            )
            translations.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

        return translations
    