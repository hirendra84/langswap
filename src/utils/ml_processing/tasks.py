import os
from time import sleep

import requests
from celery import Celery

from src.enums import ProcessStatus

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost')
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:8000')

app = Celery('api-ml', broker=CELERY_BROKER_URL)


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def speech_to_text(public_id: str, file_link: str):
    pass
    print('AMAMLBACKEND')
    print(file_link)
    sleep(10)
    # requests.get('ilya_back')
    # _i_call_my_ml_back_processing()
    r = requests.put(f'{BACKEND_URL}/video/{public_id}',
                     json={
                        'status': ProcessStatus.in_progress,
                        'prepared_link': 'https://shluhi.com/ssilka_na_fotki_twoyey_tyan',
                        'public_id': public_id
                        }
                     )
    r.raise_for_status()
    return 'my_sp_to_text_result'


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def speaker_encoder(speech_to_text_result, public_id: str, file_link: str):

    print('AMAMLBACKEND')
    print(speech_to_text_result)
    sleep(10)
    # requests.get('ilya_back')
    # _i_call_my_ml_back_processing()
    r = requests.put(f'{BACKEND_URL}/video/{public_id}',
                     json={
                         'status': ProcessStatus.in_progress,
                         'prepared_link': 'https://shluhi.com/ssilka_na_fotki_twoyey_tyan',
                         'public_id': public_id
                     }
                     )
    r.raise_for_status()
    return 'my_speaker_encoder_result'


@app.task(autoretry_for=(requests.HTTPError, requests.ConnectionError),
          retry_kwargs={'max_retries': 5},
          default_retry_delay=3)  # 3 sec
def text_to_speech(speaker_encoder_result, public_id: str, file_link: str):
    print('AMAMLBACKEND')
    print(speaker_encoder_result)
    sleep(10)
    # requests.get('ilya_back')
    # _i_call_my_ml_back_processing()
    r = requests.put(f'{BACKEND_URL}/video/{public_id}',
                     json={
                         'status': ProcessStatus.done,
                         'prepared_link': 'https://shluhi.com/ssilka_na_fotki_twoyey_tyan',
                         'public_id': public_id
                     }
                     )
    r.raise_for_status()
