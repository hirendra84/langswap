from abc import ABC
import deepl
import torch
from tqdm import tqdm
from seamless_communication.inference import Translator

from src.utils.ml_processing.lang2code_mapper import map_language_to_code

class TranslatorClient(ABC):

    def __init__(self, key: str):
        ...

    def translate(self, sentences: list[str], source_lang: str, target_lang: str) -> list[str]:
        ...


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
            translated_sentences = self._client.translate_text(sentences, source_lang=source_lang, target_lang=target_lang)
        except AttributeError:
            print()
        if isinstance(translated_sentences, deepl.TextResult):
            translated_sentences = [translated_sentences]

        return [r.text for r in translated_sentences]

class SeamlessClient(TranslatorClient):
    def __init__(self, key: str):
        super().__init__(key)
        model_name = "seamlessM4T_v2_large"
        vocoder_name = "vocoder_v2"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.translator = Translator(
            model_name,
            vocoder_name,
            device=device,
            dtype=torch.float16,
        )
    
    def translate(self, sentences: list[str], source_lang: str, target_lang: str) -> list[str]:
        source_lang = map_language_to_code(source_lang, "seamless")
        target_lang = map_language_to_code(target_lang, "seamless")
        
        translations = []
        for sent in tqdm(sentences):
            translated_sent = ""
            translated_s, _ = self.translator.predict(
                    input=sent,
                    task_str="t2st",
                    tgt_lang=target_lang,
                    src_lang=source_lang,
                )

            translated_sent = str(translated_s[0])

            translations.append(translated_sent)
        return translations
