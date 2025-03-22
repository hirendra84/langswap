from abc import ABC
import json
import torch
from tqdm import tqdm
import os
from src.utils.ml_processing.lang2code_mapper import map_language_to_code
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from typing import Tuple, List
from src.model_config import MODEL_WEIGHTS_DIR

class TranslatorClient(ABC):

    def __init__(self, device: str):
        ...

    def translate(self, sentences: List[str], source_lang: str, target_lang: str, context: str) -> list[str]:
        ...



class GemmaTranslationClient(TranslatorClient):
    def __init__(self, device="cuda", path_to_model="./models_weights/gemma-3-4b-it"):
        super().__init__(device)

        self.device = device
        
        self.path_to_model = os.path.abspath(path_to_model)

        self.tokenizer = None
        self.model = None

    def load_models(self):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.path_to_model,
            device_map=self.device,
            cache_dir=MODEL_WEIGHTS_DIR
        ).eval()      
        self.processor = AutoProcessor.from_pretrained(
            self.path_to_model,
            cache_dir=MODEL_WEIGHTS_DIR
        )

    def translate_sent(self, text: str, input_lang: str, output_lang: str, context: str, temperature=0.75, top_p=1.0, top_k=0, max_new_tokens=1024) -> str:
        messages = [

            {
                "role": "system", 
                "content":  [
                    {
                        "type": "text", 
                        "text": "You are a translation assistant. Your task is to translate the text provided in the user's input (can contain syntax and semantic error, need fix them) from the source language to the target language, using context. You will respond only with the translation of the text in the target language."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Translate from "input_lang": "{input_lang}" to "target_lang": "{output_lang}", "context": "{context}",  "text": "{text}"'}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded


    def translate(self, sentences: List[str], source_lang: str, target_lang: str, context: str) -> list[str]:
        translations = []
        for sentence in tqdm(sentences):
            translated_sent = self.translate_sent(sentence, input_lang=source_lang, output_lang=target_lang, context=context)

            translations.append(translated_sent)
        return translations