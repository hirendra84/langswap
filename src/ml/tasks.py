import os

import requests
from celery import Celery

from src.ml.api_client import MockAPIClient, APIClient
from src.file_repository import RemoteFile, FileRepository, file_repo_klass
from src.pipeline_models import VideoTranslation
from src.settings import DEBUG, BACKEND_URL, BASE_WORKING_DIR
from src.ml.speech_to_text_service import SpeechToTextManager
from src.ml.text_to_speech_service import TextToSpeechManager
from src.ml.translation_service import TranslationManager
from src.utils.s3_client import get_s3_client

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost')

app = Celery('api-ml', broker=CELERY_BROKER_URL)

api_client_klass = DEBUG and MockAPIClient or APIClient

api_client = api_client_klass(BACKEND_URL)


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def speech_to_text(public_id: str, file: RemoteFile, file_repository: FileRepository | None = None):
    if file_repository is None:
        file_repository = file_repo_klass(
            public_id,
            base_directory=BASE_WORKING_DIR,
            s3_client=get_s3_client()
        )
    manager = SpeechToTextManager(public_id, api_client, file_repository)
    video_translation = manager.extract_and_transcribe(
        VideoTranslation(
            public_id=public_id,
            source_file=file
        )
    )
    return video_translation


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def translate(video_translation: VideoTranslation):
    manager = TranslationManager(video_translation.public_id, api_client)
    video_translation = manager.translate(video_translation)
    return video_translation


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def text_to_speech(video_translation: VideoTranslation):

    file_repository = file_repo_klass(
            video_translation.public_id,
            base_directory=BASE_WORKING_DIR,
            s3_client=get_s3_client()
        )
    manager = TextToSpeechManager(video_translation.public_id, api_client, file_repository)
    video_translation = manager.synthesize(video_translation)
    return video_translation




