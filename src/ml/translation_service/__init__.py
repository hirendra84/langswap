from logging import getLogger
import pandas as pd
import os
import json
from src.ml.api_client import APIClient
from src.pipeline_models.enums import ProcessStatus
from src.pipeline_models.models import TranslatedTextedSegment, VideoTranslation
from src.file_repository import FileRepository

from src.ml.translation_service.translator_client import TranslatorClient, GemmaTranslationClient


logger = getLogger(__name__)


class TranslationManager:
    public_id: str

    _translator_client: TranslatorClient
    _api_client: APIClient
    _file_repository: FileRepository

    def __init__(self, public_id: str, api_client: APIClient, file_repository: FileRepository, device: str, logger):
        self.public_id = public_id
        self._api_client = api_client
        self._file_repository = file_repository

        self.device = device
        self.logger = logger
        self._translator_client = GemmaTranslationClient(self.device)

    def translate(self, video_translation: VideoTranslation, source_lang: str, target_lang: str) -> VideoTranslation:
        segments = video_translation.recognized_texts
        sentences_texts = [s.text for s in segments]
        context = ''.join([f"speaker: {s.speaker}:\n {s.text}\n" for s in segments])
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
                                                         source_lang=source_lang,
                                                         target_lang=target_lang,
                                                         context=context)

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
