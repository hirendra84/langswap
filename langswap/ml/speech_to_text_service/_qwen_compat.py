"""transformers 5.x compatibility shims for the `qwen-asr` package (0.0.6).

qwen-asr was built against transformers 4.57.x; several internals it relies on
were renamed/removed in 5.x.  These shims patch them back in-process before
qwen_asr classes are imported/loaded.  Idempotent and side-effecting (they patch
global transformers modules), so calling more than once is harmless.

NOTE: `asr_qwen_client.QwenASRX._load_asr_model` currently carries an inline copy
of this logic for the vLLM path.  New code (the ONNX backend) uses this shared
function; the inline copy can be migrated to call this in a later cleanup.
"""

import logging

logger = logging.getLogger(__name__)


def apply_transformers5_compat_shims() -> None:
    # 1. check_model_inputs — signature flipped between versions.  Accept both
    #    the direct-decorator and factory call styles with a no-op.
    import transformers.utils.generic as _tug

    def _check_model_inputs_compat(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

    _tug.check_model_inputs = _check_model_inputs_compat
    try:
        import transformers.utils as _tu
        _tu.check_model_inputs = _check_model_inputs_compat
    except Exception:
        pass

    # 1b. create_causal_mask — 5.x renamed input_embeds->inputs_embeds and
    #     dropped cache_position; the qwen-asr forced-aligner forward still calls
    #     it the old way.
    try:
        import transformers.masking_utils as _mu

        _orig_create_causal_mask = _mu.create_causal_mask

        def _create_causal_mask_compat(*args, **kwargs):
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)
            return _orig_create_causal_mask(*args, **kwargs)

        _mu.create_causal_mask = _create_causal_mask_compat
    except Exception:
        pass

    # 2. ROPE_INIT_FUNCTIONS['default'] — re-add the plain RoPE variant.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        import torch

        def _default_rope_init(config, device=None, **kwargs):
            base = getattr(config, "rope_theta", 10000)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

    try:
        import qwen_asr  # noqa: F401 — ensure the package is importable
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `qwen-asr`. Install with: pip install qwen-asr --no-deps"
        ) from e

    # 3. Strip the `validate_token_ids` class-validator (huggingface_hub 1.x runs
    #    validators in __init__ before qwen-asr populates thinker_config).
    def _strip_token_validator(cls):
        vlist = getattr(cls, "__class_validators__", None)
        if not vlist:
            return
        cls.__class_validators__ = [
            v for v in vlist if getattr(v, "__name__", "") != "validate_token_ids"
        ]

    try:
        from transformers.configuration_utils import PretrainedConfig as _PC
        _strip_token_validator(_PC)
    except Exception:
        pass

    # 4. Inject compute_default_rope_parameters onto qwen-asr RotaryEmbedding mods.
    try:
        import torch as _t
        import qwen_asr.core.transformers_backend.modeling_qwen3_asr as _qmod

        def _compute_default_rope_parameters(self, config, device=None, **kwargs):
            base = getattr(config, "rope_theta", 10000)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            inv_freq = 1.0 / (base ** (_t.arange(0, head_dim, 2).float() / head_dim))
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

    # 5. Backfill token-id defaults and harden get_text_config on the configs.
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRConfig as _QASRConfig,
        )
        _strip_token_validator(_QASRConfig)

        _TOKEN_DEFAULTS = {
            "pad_token_id": None,
            "bos_token_id": None,
            "eos_token_id": None,
            "decoder_start_token_id": None,
        }
        for _sub in ("Qwen3ASRThinkerConfig", "Qwen3ASRAudioConfig", "Qwen3ASRTalkerConfig"):
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

        _orig_get_text_config = _QASRConfig.get_text_config

        def _patched_get_text_config(self, *args, **kwargs):
            if not hasattr(self, "thinker_config") or self.thinker_config is None:
                return self
            return _orig_get_text_config(self, *args, **kwargs)

        _QASRConfig.get_text_config = _patched_get_text_config
    except Exception:
        pass
