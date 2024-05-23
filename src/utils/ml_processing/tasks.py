import os

import requests
from celery import Celery

from src.api_client import MockAPIClient, APIClient
from src.pipeline_models import VideoTranslation
from src.settings import DEBUG, BACKEND_URL
from src.speech_to_text_service import SpeechToTextManager
from src.text_to_speech_service import TextToSpeechManager
from src.translation_service import TranslationManager

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost')

app = Celery('api-ml', broker=CELERY_BROKER_URL)

api_client_klass = DEBUG and MockAPIClient or APIClient

api_client = api_client_klass(BACKEND_URL)


# @app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
#           retry_kwargs={'max_retries': 5},
#           default_retry_delay=3)  # 3 sec
# def speech_to_text(public_id: str, file_link: str):
#     pass
#     print('AMAMLBACKEND')
#     print(file_link)
#     manager = SpeechToTextManager(public_id)
#     manager.extract_audio(file_link)
#     # sleep(10)
#     # requests.get('ilya_back')
#     # _i_call_my_ml_back_processing()
#     r = requests.put(f'{BACKEND_URL}/video/{public_id}',
#                      json={
#                         'status': ProcessStatus.in_progress,
#                         'prepared_link': 'https://shluhi.com/ssilka_na_fotki_twoyey_tyan',
#                         'public_id': public_id
#                         }
#                      )
#     r.raise_for_status()
#     return 'my_sp_to_text_result'


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def speech_to_text(public_id: str, file_link: str):
    pass
    print('AMAMLBACKEND')
    print(file_link)
    manager = SpeechToTextManager(public_id, api_client)
    video_translation = manager.extract_and_transcribe(VideoTranslation(source_url=file_link))
    manager = TranslationManager(public_id, api_client)
    video_translation = manager.translate(video_translation)
    manager = TextToSpeechManager(public_id, api_client)
    video_translation = manager.synthesize(video_translation)
    print(video_translation)


    # sleep(10)
    # requests.get('ilya_back')
    # _i_call_my_ml_back_processing()
    # r = requests.put(f'{BACKEND_URL}/video/{public_id}',
    #                  json={
    #                     'status': ProcessStatus.in_progress,
    #                     'prepared_link': 'https://shluhi.com/ssilka_na_fotki_twoyey_tyan',
    #                     'public_id': public_id
    #                     }
    #                  )
    # r.raise_for_status()
    return 'my_sp_to_text_result'


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def speaker_encoder(speech_to_text_result, public_id: str, file_link: str):

    print('AMAMLBACKEND')
    print(speech_to_text_result)
    # sleep(10)
    # requests.get('ilya_back')
    # _i_call_my_ml_back_processing()
    # r = requests.put(f'{BACKEND_URL}/video/{public_id}',
    #                  json={
    #                      'status': ProcessStatus.in_progress,
    #                      'prepared_link': 'https://shluhi.com/ssilka_na_fotki_twoyey_tyan',
    #                      'public_id': public_id
    #                  }
    #                  )
    # r.raise_for_status()
    return 'my_speaker_encoder_result'


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def text_to_speech(speaker_encoder_result, public_id: str, file_link: str):
    # print('AMAMLBACKEND')
    # print(speaker_encoder_result)
    # sleep(10)
    # # requests.get('ilya_back')
    # # _i_call_my_ml_back_processing()
    # r = requests.put(f'{BACKEND_URL}/video/{public_id}',
    #                  json={
    #                      'status': ProcessStatus.done,
    #                      'prepared_link': 'https://shluhi.com/ssilka_na_fotki_twoyey_tyan',
    #                      'public_id': public_id,
    #                      'progress': 3,
    #                      'translated': ['xyi', 'xyi', 'xyi'],
    #                      'recognized': ['xyi', 'xyi', 'xyi'],
    #                  }
    #                  )
    # r.raise_for_status()
    pass
