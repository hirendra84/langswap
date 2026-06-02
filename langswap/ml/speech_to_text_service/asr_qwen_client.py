import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langswap.model_config import MODEL_WEIGHTS_DIR, resolve_model
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

import attr
import torch
import cattrs

load_dotenv()

logger = logging.getLogger(__name__)

# vLLM launches its EngineCore in a child process.  When the parent has already
# initialized a CUDA context (e.g. the Gradio app probed torch.cuda for device
# selection), a forked child cannot re-initialize CUDA and crashes with
# "Cannot re-initialize CUDA in forked subprocess".  Force the spawn start
# method so the worker gets a fresh process.  setdefault keeps any user override.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# vLLM's FlashInfer top-k/top-p sampler JIT-compiles a CUDA kernel (needs ninja
# + a matching nvcc) on first use.  ASR decodes greedily (temperature=0), so the
# sampler is unnecessary — disable it to skip the fragile JIT toolchain.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

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
    translation: str = None
    segments: list[Segment] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class TranscriptionData:
    output: Output


# Minimum silence (seconds) that counts as a real, dubbing-relevant pause.
# 0.25 s is the standard prosodic-phrase boundary: above the ~0.18 s stop-closure
# of voiceless consonants (so we don't split on articulation) yet low enough to
# keep every linguistically meaningful pause. Splitting here makes the silence a
# gap *between* segments, which is the only pause the downstream merge reinserts.
# NOTE: keep this in sync with the _remap_pauses default in __init__.py.
PAUSE_THRESHOLD_SECONDS = 0.25


def _group_words_into_segments(words: list[dict], pause_threshold: float = PAUSE_THRESHOLD_SECONDS) -> list[dict]:
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
        asr_model_id: Optional[str] = None,
        aligner_model_id: Optional[str] = None,
    ) -> None:
        self.device = device
        self.skip_diarization = skip_diarization

        # Resolve to a local cache path when a model name is not explicitly
        # provided.  An explicit HF repo id (e.g. "Qwen/Qwen3-ASR-1.7B") is
        # respected and passed straight to the loader, which will fetch via
        # the standard HF cache.
        self.asr_model_id = resolve_model(
            "LANGSWAP_QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B", asr_model_id)
        self.aligner_model_id = resolve_model(
            "LANGSWAP_QWEN_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B", aligner_model_id)

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
                "Please set HF_TOKEN — the pyannote diarization model downloads automatically on first use."
            )

        self.load_models()

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # When the client is reused across jobs (warm pool) keep the loaded
        # engine resident; otherwise free it so the next stage has VRAM.
        from langswap.model_pool import warm_reuse_enabled
        if warm_reuse_enabled():
            return False
        self.asr_model = None
        self.diarize_model = None

    def _best_device(self) -> tuple[torch.dtype, str]:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16, self.device
        if torch.backends.mps.is_available():
            return torch.float32, "mps"
        return torch.float32, "cpu"

    def load_models(self):
        # Idempotent.  This client is both constructed (──> __init__ calls
        # load_models) AND used as a context manager (──> __enter__ calls it
        # again) by the pipeline, so without this guard the ASR model loads
        # TWICE — spinning up a second vLLM engine that collides with the first
        # on GPU memory and aborts with "Free memory ... less than desired GPU
        # memory utilization".  Load once; subsequent calls are no-ops.
        if getattr(self, "asr_model", None) is not None:
            return
        self._load_asr_model()
        if not self.skip_diarization:
            self._load_diarize_model()

    def _load_asr_model(self):
        # ── transformers 5.x compatibility shims for qwen-asr 0.0.6 ──────────
        # qwen-asr was built against transformers 4.57.6.  Several internals it
        # relies on were removed or renamed in 5.x.  We patch them back in-process
        # before importing qwen_asr so no code changes to the upstream package are
        # needed.

        # 1. check_model_inputs — its signature flipped between transformers
        # versions.  qwen-asr 0.0.6 uses `@check_model_inputs()` (factory form,
        # returns a decorator), but transformers 5.x defines it as a direct
        # decorator `check_model_inputs(func)`.  Force a no-op shim that
        # accepts BOTH call styles.
        import transformers.utils.generic as _tug

        def _check_model_inputs_compat(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]            # @check_model_inputs (direct)
            return lambda fn: fn          # @check_model_inputs(...) (factory)

        _tug.check_model_inputs = _check_model_inputs_compat
        # Some modules import the symbol directly from transformers.utils too:
        try:
            import transformers.utils as _tu
            _tu.check_model_inputs = _check_model_inputs_compat
        except Exception:
            pass

        # 1b. create_causal_mask - transformers 5.x renamed the input_embeds
        # argument to inputs_embeds and dropped cache_position.  qwen-asr 0.0.6
        # forced-aligner forward still calls it the old way, so wrap it to
        # translate the kwarg and swallow the removed one.  Patched on the
        # source module before qwen_asr imports the symbol from it.
        try:
            import transformers.masking_utils as _mu

            _orig_create_causal_mask = _mu.create_causal_mask

            def _create_causal_mask_compat(*args, **kwargs):
                if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                kwargs.pop("cache_position", None)  # removed in transformers 5.x
                return _orig_create_causal_mask(*args, **kwargs)

            _mu.create_causal_mask = _create_causal_mask_compat
        except Exception:
            pass

        # 2. ROPE_INIT_FUNCTIONS['default'] — the plain RoPE variant was removed
        #    from the registry in 5.x.  Re-add it using the standard formula:
        #    inv_freq = 1 / (base ** (2i / dim)).
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            import torch

            def _default_rope_init(config, device=None, **kwargs):
                base = getattr(config, "rope_theta", 10000)
                head_dim = getattr(config, "head_dim", None) or (
                    config.hidden_size // config.num_attention_heads
                )
                # Create on CPU; register_buffer in the caller handles device
                # placement.  Avoid .to(device) here — during device_map loading
                # transformers uses meta-tensor context that intercepts .to() and
                # raises "Cannot copy out of meta tensor".
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, head_dim, 2).float() / head_dim)
                )
                return inv_freq, 1.0

            ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

        try:
            from qwen_asr import Qwen3ASRModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `qwen-asr`. Install with: pip install qwen-asr --no-deps"
            ) from e

        # 3. huggingface_hub 1.x runs `__class_validators__` inside
        # PretrainedConfig.__init__ via @strict_dataclass.  qwen-asr 0.0.6
        # expects validation to run AFTER it has populated `thinker_config`,
        # and its Qwen3ASRThinkerConfig doesn't expose `pad_token_id` at all.
        # Strip the `validate_token_ids` class-validator from every relevant
        # config class so init goes through.
        def _strip_token_validator(cls):
            vlist = getattr(cls, "__class_validators__", None)
            if not vlist:
                return
            cls.__class_validators__ = [
                v for v in vlist
                if getattr(v, "__name__", "") != "validate_token_ids"
            ]

        try:
            from transformers.configuration_utils import PretrainedConfig as _PC
            _strip_token_validator(_PC)
        except Exception:
            pass
        # 4. transformers 5.x _init_weights expects every RotaryEmbedding
        # module with rope_type=="default" to expose compute_default_rope_parameters.
        # qwen-asr's RoPE modules don't.  Inject a method that follows the
        # standard inv_freq formula.
        try:
            import torch as _t
            import qwen_asr.core.transformers_backend.modeling_qwen3_asr as _qmod

            def _compute_default_rope_parameters(self, config, device=None, **kwargs):
                base = getattr(config, "rope_theta", 10000)
                head_dim = getattr(config, "head_dim", None) or (
                    config.hidden_size // config.num_attention_heads
                )
                inv_freq = 1.0 / (
                    base ** (_t.arange(0, head_dim, 2).float() / head_dim)
                )
                return inv_freq, 1.0

            for _name in dir(_qmod):
                _cls = getattr(_qmod, _name, None)
                if (
                    isinstance(_cls, type)
                    and "RotaryEmbedding" in _cls.__name__
                    and not hasattr(_cls, "compute_default_rope_parameters")
                ):
                    _cls.compute_default_rope_parameters = _compute_default_rope_parameters
        except Exception:
            pass

        try:
            from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
                Qwen3ASRConfig as _QASRConfig,
            )
            _strip_token_validator(_QASRConfig)

            # In Qwen3-ASR-1.7B the per-sub-config schemas no longer expose
            # token-id fields that the (older) qwen-asr modeling code reads
            # directly (config.pad_token_id, etc.).  Backfill safe defaults
            # at the class level so attribute access returns None instead of
            # raising AttributeError.
            _TOKEN_DEFAULTS = {
                "pad_token_id": None,
                "bos_token_id": None,
                "eos_token_id": None,
                "decoder_start_token_id": None,
            }
            for _sub in (
                "Qwen3ASRThinkerConfig",
                "Qwen3ASRAudioConfig",
                "Qwen3ASRTalkerConfig",
            ):
                _cls = getattr(
                    __import__(
                        "qwen_asr.core.transformers_backend.configuration_qwen3_asr",
                        fromlist=[_sub],
                    ),
                    _sub,
                    None,
                )
                if _cls is None:
                    continue
                _strip_token_validator(_cls)
                for _attr, _default in _TOKEN_DEFAULTS.items():
                    if not hasattr(_cls, _attr):
                        setattr(_cls, _attr, _default)

            # Also harden get_text_config for the case where thinker_config is
            # not yet set when an outer validator runs.
            _orig_get_text_config = _QASRConfig.get_text_config

            def _patched_get_text_config(self, *args, **kwargs):
                if not hasattr(self, "thinker_config") or self.thinker_config is None:
                    return self
                return _orig_get_text_config(self, *args, **kwargs)

            _QASRConfig.get_text_config = _patched_get_text_config
        except Exception:
            pass

        model_dtype, device_map = self._best_device()
        is_cuda = isinstance(device_map, str) and device_map.startswith("cuda")

        # On CUDA, prefer the vllm backend — it bypasses the transformers
        # model-loading code that conflicts with transformers 5.x.
        # The vllm backend uses Qwen3ASRModel.LLM() (not from_pretrained).
        # On Mac/CPU we fall back to the transformers backend.
        if is_cuda:
            try:
                import vllm  # noqa: F401 — just check availability
                # vLLM defaults to gpu_memory_utilization=0.9+, which fails when
                # the GPU is shared (other pipeline models, a remote ASR service,
                # etc.).  Cap it (overridable) so ASR fits alongside translation/
                # TTS on a single GPU.
                gpu_util = float(os.getenv("LANGSWAP_QWEN_ASR_GPU_UTIL", "0.5"))
                # Cap context length too: the default (65536) demands ~7 GiB of
                # KV cache, which won't fit under a modest gpu_memory_utilization.
                # ASR segments are short, so a smaller window is plenty.
                max_model_len = int(os.getenv("LANGSWAP_QWEN_ASR_MAX_LEN", "16384"))
                self.asr_model = Qwen3ASRModel.LLM(
                    model=self.asr_model_id,
                    forced_aligner=self.aligner_model_id,
                    # forced_aligner is still loaded via transformers, so dtype/device_map are fine
                    forced_aligner_kwargs={"dtype": model_dtype, "device_map": device_map},
                    # vllm.LLM kwargs: dtype as string, no device_map
                    dtype="bfloat16",
                    trust_remote_code=True,
                    gpu_memory_utilization=gpu_util,
                    max_model_len=max_model_len,
                )
                logger.info(
                    "qwen-asr loaded with vllm backend (gpu_memory_utilization=%s, max_model_len=%s)",
                    gpu_util, max_model_len,
                )
                return
            except ImportError:
                logger.info("vllm not available, falling back to transformers backend for qwen-asr")

        # Transformers backend (Mac/CPU or when vllm is not installed)
        self.asr_model = Qwen3ASRModel.from_pretrained(
            self.asr_model_id,
            forced_aligner=self.aligner_model_id,
            forced_aligner_kwargs={"dtype": model_dtype, "device_map": device_map},
            dtype=model_dtype,
            device_map=device_map,
        )
        logger.info("qwen-asr loaded with transformers backend")

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
