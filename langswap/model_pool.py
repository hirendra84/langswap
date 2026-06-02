"""Process-global model cache for warm reuse across jobs.

By default this is a no-op: every caller gets a freshly constructed client, so
local / Gradio behaviour is unchanged.  When ``LANGSWAP_WARM_REUSE`` is set
(e.g. on the long-lived Modal container), constructed model holders are cached
by key and reused on every subsequent job — the expensive vLLM engine init /
weight load is paid once per container instead of once per job.

Caveat: with reuse on, models are NOT freed between pipeline stages, so the ASR,
translation and TTS engines stay co-resident in VRAM.  On a 24 GB GPU that
requires capping each engine's footprint (see LANGSWAP_QWEN_ASR_GPU_UTIL); on a
smaller card it can OOM.  That's why this is opt-in.
"""

import os

_POOL: dict = {}


def warm_reuse_enabled() -> bool:
    return os.getenv("LANGSWAP_WARM_REUSE", "").strip().lower() in {"1", "true", "yes", "on"}


def get_or_create(key, factory):
    """Return a cached instance for ``key`` (creating it via ``factory`` once),
    or — when warm reuse is disabled — just call ``factory()`` every time."""
    if not warm_reuse_enabled():
        return factory()
    inst = _POOL.get(key)
    if inst is None:
        inst = factory()
        _POOL[key] = inst
    return inst
