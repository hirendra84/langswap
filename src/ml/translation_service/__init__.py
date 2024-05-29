from logging import getLogger

from src.ml.api_client import APIClient
from src.pipeline_models.enums import ProcessStatus
from src.pipeline_models import TranslatedTextedSegment, VideoTranslation

from src.ml.translation_service.translator_client import TranslatorClient, DeepLClient


logger = getLogger(__name__)


class TranslationManager:
    public_id: str
    _api_client: APIClient

    _translator_client: TranslatorClient

    def __init__(self, public_id: str, api_client: APIClient):
        self.public_id = public_id
        self._api_client = api_client
        self._translator_client = DeepLClient('b95266dc-1675-4c76-86f4-c36dd6ab9a76:fx')

    def translate(self, video_translation: VideoTranslation) -> VideoTranslation:

        segments = video_translation.recognized_texts

        sentences_texts = [s.text for s in segments]

        translations = self._translator_client.translate(sentences_texts,
                                                         source_lang='ru',
                                                         target_lang='en')
        translated_segments = []
        for s, t in zip(segments, translations):
            translated_segments.append(
                TranslatedTextedSegment(
                    text=s.text,
                    start=s.start,
                    end=s.end,
                    translation=t,
                )
            )

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
        self._api_client.update_video(self.public_id,
                                      new_video_translation,
                                      progress=60,
                                      status=ProcessStatus.translation_ready)
        return new_video_translation
