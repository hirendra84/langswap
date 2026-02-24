from logging import getLogger
import os
import json
from langswap.pipeline_models.models import TranslatedTextedSegment, VideoTranslation
from langswap.file_repository import FileRepository


from langswap.ml.translation_service.translator_client import TranslatorClient, LLMTranslationClient


logger = getLogger(__name__)


class TranslationManager:
    public_id: str

    _translator_client: TranslatorClient
    _file_repository: FileRepository

    def __init__(self, public_id: str, file_repository: FileRepository, device: str, logger):
        self.public_id = public_id
        self._file_repository = file_repository

        self.device = device
        self.logger = logger
        model_path = os.getenv("LANGSWAP_TRANSLATEGEMMA_MODEL")
        self._translator_client = LLMTranslationClient(self.device, model_path=model_path)

    def translate(self, video_translation: VideoTranslation, source_lang: str, target_lang: str) -> VideoTranslation:
        segments = video_translation.recognized_texts
        sentences_texts = [s.text for s in segments]
        
        source_sentence_collection = [{'speaker': s.speaker, 'text': s.text} for s in segments]


        video_translation
        self.logger.file_logger.info(f'Step: Translate the segments')

        file_name = "translations.json"
        log_text = os.path.join(self._file_repository.directory, file_name)
        if os.path.exists(log_text):
            self.logger.file_logger.info(f'Getting info from already translated samples')
            with open(log_text, encoding="utf-8") as f:
                json_segments = json.load(f)
                translated_segments = []
                for s, t in zip(segments, json_segments):
                    translated_segments.append(
                        TranslatedTextedSegment(
                                    text=s.text,
                                    start=s.start,
                                    end=s.end,
                                    translation=t["translation"],
                                    source_file=None,
                                    generated_file=None,
                                    speaker=s.speaker
                                )
                            )

        else:
            self._translator_client.load_models()
            translations = self._translator_client.translate(sentences=sentences_texts,
                                                         source_language=source_lang,
                                                         target_language=target_lang)

            translated_segments = []
            for s, t in zip(segments, translations):
                    translated_segments.append(
                        TranslatedTextedSegment(
                            text=s.text,
                            start=s.start,
                            end=s.end,
                            translation=t,
                            source_file=None,
                            generated_file=None,
                            speaker=s.speaker
                        )
                    )
            json_segments = [{"translation": seg.translation, "text": seg.text} for seg in translated_segments]
            self.logger.log_json(file_name="translations.json", data=json_segments)
            local_log_text = self._file_repository.get_file("translations.json")
            self._file_repository.save_file(local_log_text)

        new_video_translation = VideoTranslation(
            public_id=video_translation.public_id,
            source_file=video_translation.source_file,
            extracted_audio=video_translation.extracted_audio,
            vad_filtered_audio=video_translation.vad_filtered_audio,
            background_audio=video_translation.background_audio,
            recognized_texts=segments,
            translated_texts=translated_segments,
            processed_video=video_translation.processed_video,
        )
        return new_video_translation
