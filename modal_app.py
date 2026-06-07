"""
Modal deployment for the langswap dubbing pipeline.

This file is OPT-IN and additive: it imports the same `process_translation`
entrypoint used by `serverless.py` (RunPod) and `gradio_demo.py` (local), so
there is a single source of truth. Local / OSS users are unaffected — they keep
running `gradio_demo.py`. People who want serverless run:

    modal deploy modal_app.py        # deploy the HTTPS endpoint
    modal run modal_app.py::smoke    # sanity-check the image + GPU

Secrets (create once):
    modal secret create langswap-aws \
        AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... BUCKET=...
    modal secret create langswap-hf HF_TOKEN=...   # for gated gemma download

Weights live in a persistent Volume mounted at /models, so they download once
(first real job) and are reused on every subsequent cold start.
"""

import modal

app = modal.App("langswap-dub")

# Build the image from the repo Dockerfile — no dependency on a private
# registry, so buyers/OSS users can deploy it too. One source of truth (the
# same Dockerfile that local Docker + RunPod use). Modal injects its own
# entrypoint, so the image's gradio CMD is irrelevant here.
# The Dockerfile bakes PIP_BREAK_SYSTEM_PACKAGES=1 into its ENV, so Modal's
# client-install step (which runs directly on this image) can pip-install into
# Ubuntu 24.04's PEP-668 "externally managed" system Python.
image = modal.Image.from_dockerfile("Dockerfile").env({
    "MODEL_WEIGHTS_DIR": "/models",
})

# Persistent weights cache: HF/torch downloads land here on first use and stick.
weights = modal.Volume.from_name("langswap-weights", create_if_missing=True)

aws_secret = modal.Secret.from_name("langswap-aws")
hf_secret = modal.Secret.from_name("langswap-hf")

GPU = "L4"  # 24 GB; bump to L40S/A100 if VRAM is tight with all models resident


@app.function(image=image, gpu=GPU, timeout=120)
def smoke():
    """Cheap sanity check: image runs, GPU is visible, package imports."""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    # Confirm the import chain that the legacy builder used to break
    # (cattrs->attrs, pydantic_core->typing_extensions, fastapi, vllm).
    from importlib.metadata import version

    import typing_extensions  # noqa: F401
    import attrs  # noqa: F401
    import pydantic_core  # noqa: F401
    import fastapi  # noqa: F401
    import vllm  # noqa: F401
    import langswap.api  # noqa: F401

    info["typing_extensions"] = version("typing_extensions")
    info["attrs"] = version("attrs")
    info["imports"] = "fastapi+pydantic_core+vllm+langswap ok"
    print(info)
    return info


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": weights},
    secrets=[aws_secret, hf_secret],
    scaledown_window=300,  # stay warm 5 min after a job, then scale to zero
    timeout=1800,
)
class Dubber:
    @modal.method()
    def dub(self, job_input: dict) -> dict:
        from langswap.api import process_translation

        result = process_translation(job_input)
        # Persist any newly downloaded weights to the Volume.
        weights.commit()
        return result

    @modal.method()
    def dub_twice(self, job_input: dict) -> dict:
        """Run two jobs in ONE container to measure warm-reuse (job2) vs cold
        (job1) and surface any VRAM OOM from co-resident models."""
        import time
        from langswap.api import process_translation

        t = time.perf_counter()
        process_translation(job_input)
        job1 = time.perf_counter() - t
        t = time.perf_counter()
        process_translation(job_input)
        job2 = time.perf_counter() - t
        weights.commit()
        out = {"job1_s": round(job1, 1), "job2_s": round(job2, 1)}
        print(f"[warm-reuse] {out}", flush=True)
        return out


# A dub takes minutes, far longer than an HTTP request should stay open, so the
# endpoint is async: POST /run submits and returns a call_id immediately; the
# caller polls GET /status?call_id=... until the result (S3 URLs) is ready.
@app.function(image=image, secrets=[aws_secret], timeout=60)
@modal.fastapi_endpoint(method="POST")
def run(job: dict):
    """Submit a dub. Body is the same `{"input": {...}}` job you send RunPod."""
    call = Dubber().dub.spawn(job.get("input", job))
    return {"call_id": call.object_id}


@app.function(image=image, timeout=60)
@modal.fastapi_endpoint(method="GET")
def status(call_id: str):
    """Poll a submitted dub. Returns running / completed (+result) / failed."""
    fc = modal.FunctionCall.from_id(call_id)
    try:
        return {"status": "completed", "result": fc.get(timeout=0)}
    except TimeoutError:
        return {"status": "running"}
    except Exception as exc:  # surfaced pipeline error
        return {"status": "failed", "error": str(exc)}


@app.local_entrypoint()
def main(input_json: str = "/tmp/job.json"):
    """End-to-end test: `modal run modal_app.py` blocks until the dub returns."""
    import json

    job = json.load(open(input_json))
    result = Dubber().dub.remote(job.get("input", job))
    print("RESULT:", json.dumps(result, ensure_ascii=False))


@app.local_entrypoint()
def bench_warm(input_json: str = "/tmp/job_onnx.json"):
    """Measure warm-reuse: two jobs in one container (job1 cold, job2 warm)."""
    import json

    job = json.load(open(input_json))
    result = Dubber().dub_twice.remote(job.get("input", job))
    print("WARM_RESULT:", json.dumps(result, ensure_ascii=False))
