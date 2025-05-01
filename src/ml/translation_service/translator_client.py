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
from llama_cpp import Llama

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
            cache_dir=MODEL_WEIGHTS_DIR,
        ).eval()      
        self.processor = AutoProcessor.from_pretrained(
            self.path_to_model,
            cache_dir=MODEL_WEIGHTS_DIR
        )

    def translate_sent(self, text: str, input_lang: str, output_lang: str, context: str, temperature=0.75, top_p=1.0, top_k=0, max_new_tokens=1024) -> str:
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": "You are a translation assistant. Your task is to translate the text provided in the user's input (can contain syntax and semantic error, need fix them) from the source language to the target language, using context. You will respond only with the translation of the text in the target language."}]
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

        raw_decoded = self.processor.decode(generation, skip_special_tokens=False)
        print("Raw decoded:", raw_decoded)
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        print("Final decoded:", decoded)
        
        if not decoded.strip():
            raise ValueError("Empty translation")
        
        return decoded


    def translate(self, sentences: List[str], source_lang: str, target_lang: str, context: str) -> list[str]:
        translations = []
        for sentence in tqdm(sentences):
            translated_sent = self.translate_sent(sentence, input_lang=source_lang, output_lang=target_lang, context=context)

            translations.append(translated_sent)
        return translations

class QuantizedGemmaTranslationClient(TranslatorClient):
    def __init__(self, device="cuda", model_path="./models_weights/gemma-3-12b-it-Q4_K_M.gguf", n_gpu_layers=-1):
        super().__init__(device)
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers  # -1 means use all available GPU layers
        self.model = None
        
    def load_models(self):
        # For GGUF models, we use llama-cpp-python
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            chat_format="gemma",
            verbose=False
        )
    
    def translate_sent(self, text: str, input_lang: str, output_lang: str, context: str, SENTINEL = "<|start_translation|>",
                       temperature=1.0, top_k=64, top_p=0.95, min_p=0.0, max_new_tokens=1024*8) -> str:
        
        # Define the messages structure for create_chat_completion
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a native speaker of {output_lang}. You have a fantastic vocabulary. You like to translate words down to their original meaning yet something that your friends would say."

                )
            },
            {"role": "user", 
             "content": f"Translate the following text from {input_lang} to {output_lang}. Input may contain ASR errors, try to fix errors depending on context."
                    "The resulting text would be used in dubbing so make text longer/shorter to match the original length."
                    f"The resulting text should extend all dates, numbers and addresses into their normal form. I.e. 22 -> twenty two. Make some notes on how to translate input text, then write {SENTINEL} token, after which include nothing but translation. : ```{text}```"}
        ]
        
        # Generate response using create_chat_completion
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            stop=["<end_of_turn>", "<eos>"] # Common stop tokens for Gemma
        )
        
        # Extract the generated message content
        decoded = response["choices"][0]["message"]["content"].strip()
        print("Final decoded:", decoded)
        
        if SENTINEL not in decoded:
            raise ValueError("Text splitter for translation is missing")
        
        decoded = decoded.split(SENTINEL)[-1].strip()
        
        if not decoded.strip():
            raise ValueError("Empty translation")
        
        return decoded

    def translate(self, sentences: List[str], source_lang: str, target_lang: str, context: str) -> list[str]:
        translations = []
        for sentence in tqdm(sentences):
            translated_sent = self.translate_sent(
                sentence, 
                input_lang=source_lang, 
                output_lang=target_lang, 
                context=context
            )
            translations.append(translated_sent)
        return translations