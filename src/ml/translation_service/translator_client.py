from abc import ABC
from tqdm import tqdm
import os
from typing import Tuple, List
from llama_cpp import Llama
import re

MODEL_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "../../../models_weights")

class TranslatorClient(ABC):

    def __init__(self, device: str):
        ...

    def translate(self, sentences: List[str], source_lang: str, target_lang: str, context: List[str]) -> list[str]:
        ...

class LLMTranslationClient(TranslatorClient):
    def __init__(self, device="cuda", model_path="/media/beijing/checkpoints/gpt-oss-20b-Q5_K_M.gguf", n_gpu_layers=-1):
        super().__init__(device)
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers  # -1 means use all available GPU layers
        self.model = None
        
    def load_models(self):
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            chat_format="chatml",
            n_ctx=8096,
            verbose=False
        )

    def extract_final_output(self, text: str) -> str:
        # Match content inside <|channel|>final ... <|end|>
        match = re.search(r"<\|channel\|>final<\|message\|>(.*?)$", text, re.S)
        if match:
            return match.group(1).strip()
        return ""  # or None if not found

    def translate(self, sentences: List[str], source_language: str, target_language: str, temperature=1.0, top_k=64, top_p=0.95, min_p=0.0, max_new_tokens=1024*8) -> list[str]:
        '''
        Translates a list of sentences using LLM.
        '''
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a native speaker of {target_language}. You have a fantastic vocabulary. You like to translate words down to their original meaning yet something that your friends would say."
                    "Never use a metaphor, simile, or other figure of speech which you are used to seeing in print."
                    "Never use a long word where a short one will do."
                    "If it is possible to cut a word out, always cut it out."
                    "Never use the passive where you can use the active."
                    f"Never use a foreign phrase, a scientific word, or a jargon word if you can think of an everyday {target_language} equivalent."
                    "Break any of these rules sooner than say anything outright barbarous."
                )
            },
            {"role": "user", 
            "content": f"Translate the following text from {source_language} to {target_language}. Input may contain ASR errors, try to fix errors depending on context."
                        "The resulting text would be used in dubbing so make text longer/shorter to match the original length."
                        f"The resulting text should extend all dates, numbers and addresses into their normal form. I.e. 22 -> twenty two. In your answer, include nothing but translation."},
        ]
        
        translations = []
        for sentence in tqdm(sentences):
            if len(messages) > 22:
                messages = messages[:2] + messages[-20:]

            messages.append({"role": "user", "content": sentence})
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                stop=["<|im_end|>", "<|endoftext|>"]  # Changed to ChatML stop tokens
            )

            decoded = self.extract_final_output(response["choices"][0]["message"]["content"].strip())
            
            messages.append(response["choices"][0]['message'])
        
            translations.append(decoded)
        
        return translations
    

if __name__ == "__main__":
    client = LLMTranslationClient()
    client.load_models()
    print(client.translate(["Hello, how are you?"], "en", "it", temperature=1.0))