from abc import ABC
import json
import torch
from tqdm import tqdm
import os
from src.utils.ml_processing.lang2code_mapper import map_language_to_code
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class TranslatorClient(ABC):

    def __init__(self, device: str):
        ...

    def translate(self, sentences: list[str], source_lang: str, target_lang: str) -> list[str]:
        ...

class HugTranslationClient(TranslatorClient):
    def __init__(self, device="cuda", path_to_model="./models_weights/cohereforai-23/cohere/", is_quantization=None):
        super().__init__(device)

        if is_quantization is None:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.device = device
        
        self.path_to_model = os.path.abspath(path_to_model)

        self.tokenizer = None
        self.model = None

    def load_models(self):
        self.model = AutoModelForCausalLM.from_pretrained(
                self.path_to_model,
                quantization_config=self.quantization_config,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_to_model, device_map=self.device)
      

    def translate_sent(self, text: str, input_lang: str, output_lang: str, temperature=0.75, top_p=1.0, top_k=0, max_new_tokens=1024) -> str:
        input_lang = input_lang.capitalize()
        output_lang = output_lang.capitalize()
        messages = [
            {"role": "system", "content": f"You are a translation assistant. Your task is to translate the text provided in the user's input from the source language to the target language. You will respond only with the translation of the text in the target language."},    
            {"role": "user", "content": f'Translate from {input_lang} to {output_lang}: "{text}"'}
        ]
        input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
            )
        input_ids = input_ids.to(self.model.device)
        prompt_padded_len = len(input_ids[0])

        gen_tokens = self.model.generate(
                input_ids,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

        # get only generated tokens
        gen_tokens = [
            gt[prompt_padded_len:] for gt in gen_tokens
            ]

        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return gen_text[0]

    def translate(self, sentences: list[str], source_lang: str, target_lang: str) -> list[str]:
        source_lang = map_language_to_code(source_lang, "cohere")
        target_lang = map_language_to_code(target_lang, "cohere")
        
        translations = []
        for sent in tqdm(sentences):
            translated_sent = self.translate_sent(sent, input_lang=source_lang, output_lang=target_lang)

            translations.append(translated_sent)
        return translations

import deepl

class DeepLClient(TranslatorClient):
    _client: deepl.Translator

    def __init__(self, key: str):
        super().__init__(key)
        self._client = deepl.Translator(key)

    def translate(self, sentences: list[str], source_lang: str, target_lang: str) -> list[str]:
        def _fix_target_language(language: str):
            if language == 'en':
                return 'en-US'
            return language
        
        target_lang = _fix_target_language(target_lang)
        try:
            # TODO: add the languages back, add the json file
            translated_sentences = self._client.translate_text(sentences, source_lang="RU", target_lang="EN-US")
        except AttributeError:
            print()

        if isinstance(translated_sentences, deepl.TextResult):
            translated_sentences = [translated_sentences]

        return [r.text for r in translated_sentences]
