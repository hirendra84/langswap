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
        # self._client = deepl.Translator(key)

    def translate(self, sentences: list[str], source_lang: str, target_lang: str) -> list[str]:
        def _fix_target_language(language: str):
            if language == 'en':
                return 'en-US'
            return language
        target_lang = _fix_target_language(target_lang)
        # translated_sentences = self._client.translate_text(sentences, source_lang=source_lang, target_lang=target_lang)
        translated_sentences = [' When Kakashi first met Zabuza in battle, it took the swordsman a very long time to fold the seals in order to apply the water dragon technique.',
 ' It looked rather odd, considering that usually any other shinobi would fold far fewer seals to use their abilities.',
 " God forbid someone add up 12, but there's 44 of them.",
 " And it would be okay if it was a weak shinobi who couldn't reduce the number of seals to use a B rank technique, but here we have the demon of the hidden fog himself, what's the meaning of this?",
 ' The whole point is that Zabuza did it on purpose.',
 " At that moment, he wasn't fully familiar with Sharingan yet.",
 ' And it honestly seemed to him that if you put together such a vast number of seals, it would',
 " Kakashi won't be able to keep up with him.",
 " What Swordsman didn't know was that a Sharingan user could copy his actions in real time.",
 ' And to watch the video about',
 ' Black Clover, click on the link in the description.']
        
        return translated_sentences

        # if isinstance(translated_sentences, deepl.TextResult):
        #     translated_sentences = [translated_sentences]

        # return [r.text for r in translated_sentences]
