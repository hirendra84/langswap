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

        translations: list[str] = []
        for sentence in tqdm(sentences):
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
    