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
    def __init__(self, device="cuda", path_to_model="./models_weights/cohereforai-23/cohere/", quantization_config=None):
        super().__init__(device)

        if quantization_config is None:
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.device = device
        
        self.path_to_model = os.path.abspath(path_to_model)

        self.tokenizer = None
        self.model = None

    def load_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_to_model, local_files_only=True, device_map=self.device)
        self.model = AutoModelForCausalLM.from_pretrained(self.path_to_model, local_files_only=True, device_map=self.device,
                                                        torch_dtype=torch.float16, quantization_config=self.quantization_config)
      

    def translate_sent(self, input_text: str, input_lang: str, output_lang: str) -> str:
        messages = [
            {"role": "system", "content": f"You are a translation assistant. Your task is to translate the text provided in the user's input from the source language to the target language. You will respond only with the translation of the text in the target language in json format with, one field 'translated_text'."},
            {"role": "user", "content": f'source_language: "{input_lang}", target_language: "{output_lang}", text: "{input_text}", '}

        ]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        gen_tokens = self.model.generate(
            input_ids,
            max_new_tokens=len(input_text),
            do_sample=True,
            temperature=0.3,
        )

        gen_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).split("<|CHATBOT_TOKEN|>")[1]
        if "{" in gen_text and "}" in gen_text:
            return json.loads(gen_text)['translated_text']
        try:
            gen_text += "}"
            return json.loads(gen_text)['translated_text']
        except:
            return gen_text

    def translate(self, sentences: list[str], source_lang: str, target_lang: str) -> list[str]:
        source_lang = map_language_to_code(source_lang, "cohere")
        target_lang = map_language_to_code(target_lang, "cohere")

        translations = []
        for sent in tqdm(sentences):
            translated_sent = self.translate_sent(sent, input_lang=source_lang, output_lang=target_lang)

            translations.append(translated_sent)
        return translations