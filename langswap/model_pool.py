"""Process-global model cache: warm reuse is always on.

Constructed model holders are cached by key and reused on every subsequent job —
the expensive vLLM engine init / weight load is paid once per process instead of
once per job, so the ASR, translation and TTS engines stay resident across jobs.

Caveat: models are NOT freed between pipeline stages, so the ASR, translation and
TTS engines stay co-resident in VRAM.  On a 24 GB GPU that requires capping each
engine's footprint; on a smaller card it can OOM.
"""

_POOL: dict = {}


def get_or_create(key, factory):
    """Return a cached instance for ``key``, creating it via ``factory`` once."""
    inst = _POOL.get(key)
    if inst is None:
        inst = factory()
        _POOL[key] = inst
    return inst
