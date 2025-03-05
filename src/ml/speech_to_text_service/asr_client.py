import cattrs

import sys

sys.path.append("/app")
sys.path.append("/app/whisperX")
sys.path.append("/app/src")

from whisperX import whisperx
import attr
import torch
import os
import requests
from time import sleep
from src.utils.ml_processing.lang2code_mapper import map_language_to_code

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@attr.s(auto_attribs=True)
class Segment:
    end: float
    start: float
    text: str
    words: list[dict]
    speaker: str = None


@attr.s(auto_attribs=True)
class Output:
    detected_language: str
    device: str
    model: str
    transcription: str
    translation: str = None  # Optional, since it might be null
    segments: list[Segment] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class TranscriptionData:
    output: Output


@attr.s(auto_attribs=True)
class TranscriptionDataLocal:
    output: Output


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
                "word_timestamps": True,
            },
            "enable_vad": False,
        }

        response = requests.post(url, json=payload, headers=headers)
        response = response.json()

        try:
            return cattrs.structure(response, TranscriptionData).output
        except Exception:
            print(response)
            if "error" in response and response["error"] == "failed to add to queue":
                print("Retryable error, retrying...")
                sleep(5)
                self.transcribe(source_url)
            raise


class ASRX:

    token: str

    def __init__(self, device) -> None:
        model_path = (
            "./models_weights/whisper-large-v2/f0fe81560cb8b68660e564f55dd99207059c092e"
        )
        self.model_path_whisper = os.path.abspath(model_path)

        self.model = None
        self.diarize_model = None

        self.token = "***REDACTED-HF-TOKEN***"  # secret, move somewhere
        model_path_diarization = "./models_weights/pyannote/models--pyannote--speaker-diarization-3.1/snapshots/84fd25912480287da0247647c3d2b4853cb3ee5d/config.yaml"
        self.model_path_diarization = os.path.abspath(model_path_diarization)

        self.device = device
    
    
    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model = None
        self.diarize_model = None

    def load_models(self):
        compute_type = "float32" if self.device == "cpu" else "float16"

        self.model = whisperx.load_model(
            self.model_path_whisper, device=self.device, compute_type=compute_type, local_files_only=True
        )

        self.diarize_model = whisperx.DiarizationPipeline(
            self.model_path_diarization, use_auth_token=self.token, device=self.device
        )


    def transcribe(self, source_file: str, lang=None, num_speakers=None) -> Output:
        language = None
        if lang != None:
            language = map_language_to_code(lang, system="whisper")
        
        audio = whisperx.load_audio(source_file)
        
        response = self.model.transcribe(audio, language=language, batch_size=8)

        language = response["language"]
        lang = language
        model_a, metadata = whisperx.load_align_model(
            language_code=response["language"], device=self.device
        )

        response = whisperx.align(
            response["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        diarize_segments = self.diarize_model(audio, num_speakers=num_speakers)
        response = whisperx.assign_word_speakers(diarize_segments, response)

        all_words = []
        result = {}

        final_response = {}

        for i in range(len(response["segments"])):
            all_words.extend([i for i in response["segments"][i]["words"]])

        full_text = " ".join([i["word"] for i in all_words])

        final_response["detected_language"] = language
        final_response["device"] = "cuda"
        final_response["model"] = "base"
        final_response["transcription"] = full_text
        final_response["translation"] = ""
        final_response["segments"] = response["segments"]
        final_response["word_timestamps"] = all_words

        result["status"] = "finished"

        result["output"] = final_response
        return cattrs.structure(result, TranscriptionData).output
