import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from langswap.model_config import MODEL_WEIGHTS_DIR
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

import attr
import torch
import cattrs

load_dotenv()

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def ensure_punctuation(text: str) -> str:
    """Ensure text ends with sentence-ending punctuation for better forced alignment."""
    text = text.strip()
    if text and text[-1] not in ".!?,;:":
        text += "."
    return text


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
    translation: str = None
    segments: list[Segment] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class TranscriptionData:
    output: Output


def _group_words_into_segments(words: list[dict], pause_threshold: float = 0.5) -> list[dict]:
    """
    Group word-level timestamps into segments by pause length and speaker changes.
    Returns list of dicts: {text, start, end, words, speaker}
    """
    if not words:
        return []

    segments = []
    current = [words[0]]

    for word in words[1:]:
        prev_end = current[-1].get("end", 0)
        cur_start = word.get("start", 0)
        split = (
            (cur_start - prev_end) > pause_threshold
            or word.get("speaker", "SPEAKER_00") != current[-1].get("speaker", "SPEAKER_00")
        )
        if split:
            segments.append(_make_segment(current))
            current = [word]
        else:
            current.append(word)

    segments.append(_make_segment(current))
    return segments


def _make_segment(words: list[dict]) -> dict:
    return {
        "text": " ".join(w["word"] for w in words),
        "start": words[0].get("start", 0.0),
        "end": words[-1].get("end", 0.0),
        "words": words,
        "speaker": words[0].get("speaker", "SPEAKER_00"),
    }


class QwenASRX:
    """
    ASR client using Qwen3-ASR for transcription and Qwen3-ForcedAligner for
    word-level alignment, both from the `qwen-asr` package.

    Speaker diarization is handled by pyannote via the whisperx
    DiarizationPipeline wrapper.

    Drop-in replacement for ASRX: same constructor signature and transcribe()
    return type.
    """

    def __init__(
        self,
        device: str,
        language: str,
        skip_diarization: bool = False,
        asr_model_id: str = "Qwen/Qwen3-ASR-0.6B",
        aligner_model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    ) -> None:
        self.device = device
        self.skip_diarization = skip_diarization
        self.asr_model_id = asr_model_id
        self.aligner_model_id = aligner_model_id

        if language is not None:
            self.language = map_language_to_code(language, system="whisper")
            # Qwen models expect full names: "Russian", "English", etc.
            self.language_full = map_language_to_code(language, system="cohere")
        else:
            self.language = None
            self.language_full = None

        self.asr_model = None
        self.diarize_model = None

        models_base_dir = Path(MODEL_WEIGHTS_DIR)
        diarize_config = models_base_dir / "pyannote/pyannote_diarization_config.yaml"
        self.model_path_diarization = str(diarize_config.resolve())

        if not skip_diarization and not os.path.exists(self.model_path_diarization):
            raise FileNotFoundError(
                f"Diarization model not found at: {self.model_path_diarization}\n"
                "Please set HF_TOKEN and run: langswap-download-models --model pyannote-speaker-diarization"
            )

        self.load_models()

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.asr_model = None
        self.diarize_model = None

    def _best_device(self) -> tuple[torch.dtype, str]:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16, self.device
        if torch.backends.mps.is_available():
            return torch.float32, "mps"
        return torch.float32, "cpu"

    def load_models(self):
        self._load_asr_model()
        if not self.skip_diarization:
            self._load_diarize_model()

    def _load_asr_model(self):
        try:
            from qwen_asr import Qwen3ASRModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `qwen-asr`. Install with: pip install -U qwen-asr"
            ) from e

        model_dtype, device_map = self._best_device()
        self.asr_model = Qwen3ASRModel.from_pretrained(
            self.asr_model_id,
            forced_aligner=self.aligner_model_id,
            forced_aligner_kwargs={"dtype": model_dtype, "device_map": device_map},
            dtype=model_dtype,
            device_map=device_map,
        )

    def _load_diarize_model(self):
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            try:
                import whisperx
                DiarizationPipeline = whisperx.DiarizationPipeline  # type: ignore[attr-defined]
            except Exception as e:
                raise ImportError(
                    "pyannote diarization requires whisperx. Install: pip install whisperx"
                ) from e

        cwd = Path.cwd().resolve()
        os.chdir(Path(self.model_path_diarization).parent.parent.resolve())
        self.diarize_model = DiarizationPipeline(
            self.model_path_diarization, device=self.device
        )
        os.chdir(cwd)

    def transcribe(self, source_file: str, num_speakers=None) -> Output:
        # 1. Transcribe + forced alignment in one call
        results = self.asr_model.transcribe(
            audio=str(source_file),
            language=self.language_full,
            return_time_stamps=True,
        )
        result = results[0]

        raw_lang = result.language or self.language_full or "en"
        # Normalize to whisper code (e.g. "Russian" -> "ru", "ru" stays "ru")
        try:
            detected_language = map_language_to_code(raw_lang.lower(), system="whisper")
        except (AssertionError, KeyError):
            detected_language = self.language or raw_lang

        if result.time_stamps is None or not result.time_stamps.items:
            logger.warning("ASR returned no word-level timestamps. text=%r", result.text)
            words = []
        else:
            words = [
                {
                    "word": item.text,
                    "start": float(item.start_time),
                    "end": float(item.end_time),
                }
                for item in result.time_stamps.items
            ]

        # 2. Build single-segment transcript_result for speaker assignment
        audio_end = words[-1]["end"] if words else 0.0
        transcript_result = {
            "segments": [
                {"text": result.text, "start": 0.0, "end": audio_end, "words": words}
            ]
        }

        # 3. Diarize + assign speakers per word
        if not self.skip_diarization and self.diarize_model is not None:
            import whisperx
            audio = whisperx.load_audio(source_file)
            diarize_df = self.diarize_model(audio, num_speakers=num_speakers)
            transcript_result = whisperx.assign_word_speakers(diarize_df, transcript_result)
        else:
            for seg in transcript_result["segments"]:
                seg["speaker"] = "SPEAKER_00"
                for w in seg["words"]:
                    w["speaker"] = "SPEAKER_00"

        # 4. Group words (with speaker info) into segments
        words_with_speakers = transcript_result["segments"][0]["words"]
        segments = _group_words_into_segments(words_with_speakers)

        full_text = " ".join(w["word"] for w in words_with_speakers)

        final_response = {
            "detected_language": detected_language,
            "device": "cuda" if self.device.startswith("cuda") else self.device,
            "model": "qwen3-asr",
            "transcription": full_text,
            "translation": "",
            "segments": segments,
        }
        result_data = {"status": "finished", "output": final_response}
        return cattrs.structure(result_data, TranscriptionData).output
