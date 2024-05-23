from time import sleep

import requests

from src.enums import ProcessStatus
from src.pipeline_models import VideoTranslation
from src.settings import BACKEND_URL


class APIClient:
    api_url: str = BACKEND_URL

    def __init__(self, api_url: str = None):
        if api_url is not None:
            self.api_url = api_url

    @staticmethod
    def _translation_to_api_data(video_translation: VideoTranslation,
                                 progress: int,
                                 status: ProcessStatus):
        data = {
            'status': status,
            'progress': progress,
        }
        if video_translation.recognized_texts:
            data['translated'] = [segment.text for segment in video_translation.recognized_texts]

        if video_translation.translated_texts:
            data['translated'] = [segment.translation for segment in video_translation.translated_texts]

        if video_translation.processed_video:
            data['prepared_link'] = video_translation.processed_video.s3_url
        return data

    def update_video(self, public_id, video_translation: VideoTranslation, progress: int, status: ProcessStatus):
        exception = None
        for i in range(10):
            r = requests.put(
                f'{BACKEND_URL}/video/{public_id}',
                json=self._translation_to_api_data(video_translation, progress, status)
            )
            try:
                r.raise_for_status()
                return
            except Exception as e:
                exception = e
            sleep(5)
        raise exception


class MockAPIClient(APIClient):

    def __init__(self, api_url: str = None):
        super().__init__(api_url)

    def update_video(self, public_id, video_translation: VideoTranslation, progress: int, status: ProcessStatus):
        ...
