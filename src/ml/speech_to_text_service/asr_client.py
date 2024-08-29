from time import sleep

import cattrs
import requests
import whisper
import whisperx
import attr

from src.utils.ml_processing.lang2code_mapper import map_language_to_code

# @attr.s(auto_attribs=True)
# class WordTimestamp:
#     start: float
#     end: float
#     word: str

# @attr.s(auto_attribs=True)
# class Segment:
#     avg_logprob: float
#     compression_ratio: float
#     end: float
#     id: int
#     no_speech_prob: float
#     seek: int
#     start: float
#     temperature: int
#     text: str
#     tokens: list[int]
#     word_timestamps: list[WordTimestamp] = attr.ib(factory=list)
#     speaker: str = None 

@attr.s(auto_attribs=True)
class Segment:
    # avg_logprob: float
    # compression_ratio: float
    end: float
    # id: int
    # no_speech_prob: float
    # seek: int
    start: float
    # temperature: int
    text: str
    words: list[dict]
    # word_timestamps: list[WordTimestamp] = attr.ib(factory=list)
    speaker: str = None 

@attr.s(auto_attribs=True)
class Output:
    detected_language: str
    device: str
    model: str
    transcription: str
    translation: str = None  # Optional, since it might be null
    segments: list[Segment] = attr.ib(factory=list)
    # word_timestamps: list[WordTimestamp] = attr.ib(factory=list)

@attr.s(auto_attribs=True)
class TranscriptionData:
    # delayTime: int
    # executionTime: int
    # id: str
    output: Output
    # status: str

@attr.s(auto_attribs=True)
class TranscriptionDataLocal:
    output: Output

class ASRX:

    token: str

    def __init__(self, device) -> None:
        compute_type = "float32" if device == "cpu" else "float16"
        self.model = whisperx.load_model("large-v2", device, compute_type=compute_type)

        token = "***REDACTED-HF-TOKEN***"
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=token, device=device)

        self.device = device

    def transcribe(self, source_file: str, lang='en') -> Output:
        audio = whisperx.load_audio(source_file)
        response = self.model.transcribe(audio, batch_size=8)

        language = response['language']

        model_a, metadata = whisperx.load_align_model(language_code=response["language"], device=self.device)

        response = whisperx.align(response["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        diarize_segments = self.diarize_model(audio)
        response = whisperx.assign_word_speakers(diarize_segments, response)

        all_words = []
        result = {}

        final_response = {}

        for i in range(len(response['segments'])):
            all_words.extend([i for i in response['segments'][i]['words']])
        
        full_text = " ".join([i['word'] for i in all_words])

        final_response['detected_language'] = language
        final_response['device'] = 'cuda'
        final_response['model'] = 'base'
        final_response['transcription'] = full_text
        final_response['translation'] = ''
        final_response['segments'] = response['segments']
        final_response['word_timestamps'] = all_words

        # result['delayTime'] = 1
        # result['executionTime'] = 1
        # result['id'] = 3
        result['status'] = 'finished'

        result['output'] = final_response
        return cattrs.structure(result, TranscriptionData).output

class ASRClientFaster:

    token: str

    def __init__(self, api_key: str):
        self.token = api_key

        self.model = whisper.load_model("medium")

    def transcribe(self, source_file: str, lang: str) -> Output:
        response = self.model.transcribe(source_file, word_timestamps=True)
        # response = self.model.transcribe(source_file, word_timestamps=True)

        all_words = []

        result = {}

        for i in range(len(response['segments'])):
            all_words.extend(response['segments'][i]['words'])

        response['detected_language'] = lang
        response['device'] = 'cuda'
        response['model'] = 'base'
        response['transcription'] = response["text"]
        response['translation'] = ''
        response['segments'] = response['segments']
        response['word_timestamps'] = all_words

        result['delayTime'] = 1
        result['executionTime'] = 1
        result['id'] = 3
        result['status'] = 'finished'

        result['output'] = response
        return cattrs.structure(result, TranscriptionData).output


class ASRClient:

    token: str

    def __init__(self, api_key: str):
        self.token = api_key

    def transcribe(self, source_url: str, lang: str) -> Output:
        lang = map_language_to_code(lang, "whisper")
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
                "language": lang,
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


