from time import sleep

import cattrs
import requests


import attr


@attr.s(auto_attribs=True)
class WordTimestamp:
    start: float
    end: float
    word: str


@attr.s(auto_attribs=True)
class Segment:
    avg_logprob: float
    compression_ratio: float
    end: float
    id: int
    no_speech_prob: float
    seek: int
    start: float
    temperature: int
    text: str
    tokens: list[int]
    word_timestamps: list[WordTimestamp] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class Output:
    detected_language: str
    device: str
    model: str
    transcription: str
    translation: str = None  # Optional, since it might be null
    segments: list[Segment] = attr.ib(factory=list)
    word_timestamps: list[WordTimestamp] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class TranscriptionData:
    delayTime: int
    executionTime: int
    id: str
    output: Output
    status: str


class ASRClient:

    token: str

    def __init__(self, api_key: str):
        self.token = api_key

    def transcribe(self, source_url: str) -> Output:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": self.token,
        }

        url = "https://api.runpod.ai/v2/faster-whisper/runsync"

        payload = {
            "input": {
                "audio": source_url,
                "model": "large-v2",
                "transcription": "plain_text",
                "translate": False,
                "language": "ru",
                "temperature": 0,
                "best_of": 5,
                "beam_size": 5,
                "patience": 1,
                "suppress_tokens": "-1",
                "condition_on_previous_text": False,
                "temperature_increment_on_fallback": 0.2,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1,
                "no_speech_threshold": 0.6,
                "word_timestamps": True
            },
            "enable_vad": False
        }

        response = requests.post(url, json=payload, headers=headers)

        # STEP 4: Print the response
        response = response.json()

        try:
            return cattrs.structure(response, TranscriptionData).output
        except Exception:
            print(response)
            if 'error' in response and response['error'] == 'failed to add to queue':
                print('Retryable error, retrying...')
                sleep(5)
                self.transcribe(source_url)
            raise


