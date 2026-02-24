import cattrs
import logging
import sys
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Import model config first to set up cache environment
from langswap.model_config import MODEL_WEIGHTS_DIR
from langswap.model_downloader import ensure_whisperx_model

import whisperx
import attr
import torch
import requests
from time import sleep
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

# WhisperX API compatibility:
# Some forks expose DiarizationPipeline at top-level (whisperx.DiarizationPipeline),
# others expose it under whisperx.diarize.DiarizationPipeline.
try:
    DiarizationPipeline = whisperx.DiarizationPipeline  # type: ignore[attr-defined]
except Exception:
    from whisperx.diarize import DiarizationPipeline  # type: ignore

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

    def __init__(self, device, language, skip_diarization: bool = False) -> None:
        # Auto-download whisper model if not present
        self.model_path_whisper = ensure_whisperx_model()

        self.model = None
        if language is not None:
            self.language = map_language_to_code(language, system="whisper")
        else:
            self.language = None
        self.diarize_model = None
        self.skip_diarization = skip_diarization

        # Diarization model path - uses pyannote config in cache directory
        # Note: pyannote models require HF_TOKEN for gated model access
        models_base_dir = Path(MODEL_WEIGHTS_DIR)
        diarize_model_dir = models_base_dir / "pyannote/pyannote_diarization_config.yaml"
        self.model_path_diarization = str(diarize_model_dir.resolve())

        if not skip_diarization and not os.path.exists(self.model_path_diarization):
            raise FileNotFoundError(
                f"Diarization model not found at: {self.model_path_diarization}\n"
                "Please set HF_TOKEN environment variable and run: langswap-download-models --model pyannote-speaker-diarization"
            )

        self.device = device
        self.load_models()
    
    
    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model = None
        self.diarize_model = None

    def load_models(self):
        compute_type = "int8" if self.device != "cpu" else "float32"

        self.model = whisperx.load_model(
            str(self.model_path_whisper),
            device=self.device,
            compute_type=compute_type,
            local_files_only=True,
            language=self.language
        )

        if not self.skip_diarization:
            cwd = Path.cwd().resolve()
            cd_to = Path(self.model_path_diarization).parent.parent.resolve()
            os.chdir(cd_to)
            self.diarize_model = DiarizationPipeline(
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

        if not self.skip_diarization and self.diarize_model is not None:
            diarize_segments = self.diarize_model(audio, num_speakers=num_speakers)
            response = whisperx.assign_word_speakers(diarize_segments, response)
        else:
            # Assign default speaker when diarization is skipped
            for segment in response["segments"]:
                segment["speaker"] = "SPEAKER_00"
                for word in segment.get("words", []):
                    word["speaker"] = "SPEAKER_00"

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
