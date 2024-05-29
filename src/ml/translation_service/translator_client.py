from abc import ABC
import deepl


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
