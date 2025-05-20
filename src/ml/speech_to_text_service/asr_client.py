import cattrs
import logging
import sys
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Set HuggingFace cache directory to models_weights/
current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_file_dir.parents[2]
os.environ["HF_HOME"] = str(project_root / "models_weights")
os.environ["TRANSFORMERS_CACHE"] = str(project_root / "models_weights")

import whisperx
import attr
import torch
import requests
from time import sleep
from src.utils.ml_processing.lang2code_mapper import map_language_to_code
from src.model_config import MODEL_WEIGHTS_DIR

# Load environment variables from .env file
load_dotenv()

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


class ASRX:

    def __init__(self, device, language) -> None:
        # Get the project root directory (3 levels up from current file)
        # This assumes the file is at src/ml/speech_to_text_service/asr_client.py
        current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = current_file_dir.parents[2]  # Go up 3 levels to reach project root
        
        # Define base directory for models relative to project root
        models_base_dir = project_root / "models_weights"
        
        # Whisper model path
        # whisper_model_dir = models_base_dir / "whisper-large-v3"
        self.model_path_whisper = "large-v3"

        self.model = None
        if language is not None:
            self.language = map_language_to_code(language, system="whisper")
        else:
            self.language = None
        self.diarize_model = None

        # Diarization model path
        diarize_model_dir = models_base_dir / "pyannote/pyannote_diarization_config.yaml"#models_base_dir / "pyannote" / "models--pyannote--speaker-diarization-3.1" / "snapshots" / "84fd25912480287da0247647c3d2b4853cb3ee5d" / "config.yaml"
        self.model_path_diarization = str(diarize_model_dir.resolve())
        
        # Verify model paths exist
        # if not os.path.exists(self.model_path_whisper):
        #     raise FileNotFoundError(f"Whisper model not found at: {self.model_path_whisper}")
        
        if not os.path.exists(self.model_path_diarization):
            raise FileNotFoundError(f"Diarization model not found at: {self.model_path_diarization}")

        self.device = device
        self.load_models()
    
    
    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model = None
        self.diarize_model = None

    def load_models(self):
        # For int8 models
        compute_type = "int8" if self.device != "cpu" else "float32"
        
        # Consider medium-int8 for good balance of speed and accuracy
        self.model = whisperx.load_model(
            self.model_path_whisper, 
            device=self.device, 
            compute_type=compute_type, 
            local_files_only=False,
            language=self.language
        )
        cwd = Path.cwd().resolve() 
        cd_to = Path(self.model_path_diarization).parent.parent.resolve()
        os.chdir(cd_to)
        self.diarize_model = whisperx.DiarizationPipeline(
            self.model_path_diarization, device=self.device
        )
        os.chdir(cwd)


    def get_cache_dir(self):
        """Returns the models_weights directory path"""
        return MODEL_WEIGHTS_DIR  # Go up to models_weights

    def transcribe(self, source_file: str, num_speakers=None) -> Output:
        audio = whisperx.load_audio(source_file)
        
        response = self.model.transcribe(
            audio, 
            language=self.language, 
            batch_size=8
        )

        language = response["language"]
        
        model_a, metadata = whisperx.load_align_model(
            language_code=response["language"], 
            device=self.device,
            model_dir=self.get_cache_dir()
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
