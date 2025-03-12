import os

import requests
from celery import Celery


from src.pipeline_models.models import VideoTranslation
from src.settings import DEBUG, BACKEND_URL, BASE_WORKING_DIR
from src.ml.api_client import MockAPIClient, APIClient


CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost')

app = Celery('api-ml', broker=CELERY_BROKER_URL)

app.conf.update(
    task_serializer='attrs_json',
    accept_content=['attrs_json', 'json'],  # Allow both our custom serializer and the default JSON serializer
    result_serializer='attrs_json'
)

api_client_klass = DEBUG and MockAPIClient or APIClient

api_client = api_client_klass(BACKEND_URL)


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def speech_to_text(video_translation: VideoTranslation):
    from src.file_repository import file_repo_klass
    from src.ml.speech_to_text_service import SpeechToTextManager
    from src.utils.s3_client import get_s3_client
    file_repository = file_repo_klass(
        video_translation.public_id,
        base_directory=BASE_WORKING_DIR,
        s3_client=get_s3_client()
    )
    manager = SpeechToTextManager(video_translation.public_id, api_client, file_repository)
    video_translation = manager.extract_and_transcribe(
        VideoTranslation(
            public_id=video_translation.public_id,
            source_file=video_translation.source_file
        )
    )
    return video_translation


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def translate(video_translation: VideoTranslation):
    from src.ml.translation_service import TranslationManager
    manager = TranslationManager(video_translation.public_id, api_client)
    video_translation = manager.translate(video_translation)
    return video_translation


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def text_to_speech(video_translation: VideoTranslation):
    from src.file_repository import file_repo_klass
    from src.ml.text_to_speech_service import TextToSpeechManager
    from src.utils.s3_client import get_s3_client

    file_repository = file_repo_klass(
            video_translation.public_id,
            base_directory=BASE_WORKING_DIR,
            s3_client=get_s3_client()
        )
    manager = TextToSpeechManager(video_translation.public_id, api_client, file_repository)
    video_translation = manager.synthesize(video_translation)
    return video_translation




