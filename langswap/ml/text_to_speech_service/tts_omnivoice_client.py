import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from langswap.model_downloader import ensure_omnivoice_model
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code


logger = logging.getLogger(__name__)


class OmniVoiceClient:
    """
    OmniVoice TTS client backed by the offline `vllm-omni` Python API.

    Supports auto voice and voice cloning without starting a server.
    """

    def __init__(
        self,
        model_id: str = "k2-fsa/OmniVoice",
        model_path: Optional[str] = None,
        device: str = "cuda",
        stage_config_path: Optional[str] = None,
        stage_init_timeout: Optional[int] = None,
        log_stats: Optional[bool] = None,
    ):
        self.model_id = str(ensure_omnivoice_model(model_path))
        self.device = device
        self.sample_rate = 24000
        self.stage_config_path = self._resolve_stage_config_path(stage_config_path)
        self.stage_init_timeout = int(
            stage_init_timeout
            if stage_init_timeout is not None
            else os.environ.get("LANGSWAP_OMNIVOICE_STAGE_INIT_TIMEOUT", 600)
        )
        self.log_stats = self._env_bool("LANGSWAP_OMNIVOICE_LOG_STATS", default=False) if log_stats is None else log_stats
        self.model = None
        self._sampling_params_cls = None
        self._warned_duration_unsupported = False
        self.load_models()

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_stage_config_path(self, stage_config_path: Optional[str]) -> str:
        if stage_config_path:
            return str(Path(stage_config_path).expanduser().resolve())

        env_path = os.environ.get("LANGSWAP_OMNIVOICE_STAGE_CONFIG")
        if env_path:
            return str(Path(env_path).expanduser().resolve())

        bundled = Path(__file__).resolve().parent / "configs" / "omnivoice_vllm_stage.yaml"
        return str(bundled)

    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        if not language:
            return None

        normalized = language.strip().lower()
        if len(normalized) <= 3:
            return normalized

        try:
            return map_language_to_code(normalized, system="whisper")
        except Exception:
            return language

    def _load_reference_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, int(sample_rate)

    def _extract_audio(self, outputs) -> tuple[np.ndarray, int]:
        for output in outputs:
            candidate_payloads = []

            request_output = getattr(output, "request_output", None)
            if request_output is not None:
                candidate_payloads.append(getattr(request_output, "multimodal_output", None))
                for sub_output in getattr(request_output, "outputs", []) or []:
                    candidate_payloads.append(getattr(sub_output, "multimodal_output", None))

            candidate_payloads.append(getattr(output, "multimodal_output", None))

            for payload in candidate_payloads:
                if not payload or "audio" not in payload:
                    continue

                audio = payload["audio"]
                sample_rate = int(payload.get("sr", self.sample_rate))

                if hasattr(audio, "detach"):
                    audio = audio.detach().float().cpu().numpy()
                else:
                    audio = np.asarray(audio)

                audio = np.asarray(audio, dtype=np.float32).squeeze()
                if audio.ndim > 1:
                    audio = audio[0]

                return audio, sample_rate

        raise RuntimeError("vllm-omni returned no audio payload for OmniVoice generation.")

    def load_models(self):
        try:
            from vllm_omni.entrypoints.omni import Omni
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `vllm-omni` required for OmniVoice TTS. "
                "Install it with `pip install vllm-omni`."
            ) from e

        if self.model is not None:
            return

        self.model = Omni(
            model=self.model_id,
            stage_configs_path=self.stage_config_path,
            trust_remote_code=True,
            log_stats=self.log_stats,
            stage_init_timeout=self.stage_init_timeout,
        )
        self._sampling_params_cls = OmniDiffusionSamplingParams

    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.model is not None and hasattr(self.model, "close"):
            self.model.close()
        self.model = None

    def generate_audio(
        self,
        text: str,
        source_audio_file: str,
        source_text: str,
        save_path: str,
        language: str,
        duration: Optional[float] = None,
    ):
        """Generate speech with OmniVoice through the offline `vllm-omni` API."""
        if self.model is None:
            self.load_models()

        multi_modal_data = {}
        mm_processor_kwargs = {}

        if source_audio_file:
            reference_audio, reference_sr = self._load_reference_audio(source_audio_file)
            multi_modal_data["audio"] = (reference_audio, reference_sr)
            mm_processor_kwargs["ref_text"] = (source_text or "").strip()
            mm_processor_kwargs["sample_rate"] = reference_sr

        normalized_language = self._normalize_language(language)
        if normalized_language:
            mm_processor_kwargs["lang"] = normalized_language

        if duration is not None:
            mm_processor_kwargs["duration"] = float(duration)
            if not self._warned_duration_unsupported:
                logger.warning(
                    "The current vllm-omni OmniVoice offline pipeline does not enforce explicit duration yet; "
                    "the downstream dubbing stage may still retime segments as needed."
                )
                self._warned_duration_unsupported = True

        prompt = {"prompt": text}
        if multi_modal_data:
            prompt["multi_modal_data"] = multi_modal_data
        if mm_processor_kwargs:
            prompt["mm_processor_kwargs"] = mm_processor_kwargs

        outputs = self.model.generate(
            prompt,
            sampling_params_list=[self._sampling_params_cls()],
            use_tqdm=False,
        )
        wav, sample_rate = self._extract_audio(outputs)
        self.sample_rate = sample_rate

        save_dir = os.path.dirname(save_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        sf.write(save_path, wav, sample_rate)

    def tts_pipeline(self, video_translation, temp_folder, language="en"):
        for idx, segment in enumerate(
            tqdm(
                video_translation.translated_texts,
                desc="OmniVoice generation.",
                leave=True,
            )
        ):
            file_path = os.path.join(temp_folder, f"{segment.start}_{segment.end}.wav")
            if not os.path.exists(file_path):
                self.generate_audio(
                    text=segment.translation,
                    source_audio_file=segment.source_file,
                    source_text=segment.text,
                    save_path=file_path,
                    language=language,
                    duration=(segment.end - segment.start),
                )
            video_translation.translated_texts[idx].generated_file = file_path
        return video_translation
