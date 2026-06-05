# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Coding Agent Engineering Contract

You are working in this repository as a careful senior software engineer, not a code generator. Your job is to make the smallest correct change that solves the task while preserving architecture, maintainability, and a working main branch.

## Principles

1. Keep main always working.

   Do not leave broken, half-migrated, experimental, temporary, debug, dead, commented-out, unused, or intermediate code behind.

2. Minimize blast radius.

   Modify only files necessary for the task. Do not rewrite unrelated code, reformat whole files, rename APIs, change schemas, alter configs, or touch CI unless required.

3. Understand before editing.

   Check git status. Inspect relevant files, tests, existing abstractions, and project conventions. Search for existing helpers and similar implementations before adding code.

4. Reuse before adding.

   Do not duplicate logic or create parallel implementations. Prefer existing utilities, standard library, and existing dependencies.

5. Avoid bloat.

   Prefer boring, direct, idiomatic code. Do not add speculative features, generic frameworks, extra layers, adapters, factories, registries, or future-proof abstractions.

6. Keep feature work and refactoring separate.

   Refactor only when necessary for the task or when it directly reduces risk. Mention larger cleanup separately.

7. Handle errors honestly.

   Do not silently swallow errors, fake success, use placeholder data, guessed env vars, fake API responses, or hardcoded credentials.

8. Treat tests as evidence.

   Do not delete, weaken, skip, or rewrite valid tests just to pass. Add focused regression tests when behavior changes.

9. Validate before claiming success.

   Run relevant tests/checks. If validation could not be run, say exactly what was not run and why.

10. Respect user changes.

    Never overwrite unrelated user edits. Keep the final diff limited to intentional changes.

## Required workflow

1. Inspect task, repo state, relevant files, existing patterns, and tests.

2. Plan the minimal change and define what proves it works.

3. Implement the smallest coherent fix.

4. Self-review the diff and remove unnecessary code.

5. Run relevant validation.

6. Report summary, files changed, validation, and risks.

## Stop and ask before

- Deleting or overwriting unrelated files.

- Changing public APIs, schemas, auth, payments, security, secrets, deployment, CI, or production configs.

- Adding dependencies.

- Performing large refactors.

- Weakening tests.

- Continuing after repeated failed attempts without a clear hypothesis.

## Final response format

- Summary: what changed and why.

- Files changed: concise list.

- Validation: commands run and results.

- Risks/notes: anything the reviewer should check.

Do not claim completion unless the code is implemented, cleaned up, and validated.

---

# Repository guide

langswap dubs a video into another language end to end: **ASR → translation → voice-cloned TTS → dubbing/merge → muxed video + SRT**, running locally on one GPU. The authoritative operational reference (exact pinned install, model registry, every env var, Docker notes, troubleshooting) is `docs/advanced.md` — read it before changing install/deps/Docker, and update it instead of duplicating it here.

## Commands

```bash
# Install (GPU)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[gpu]"          # full GPU stack
# or ".[api]" on Mac/CPU (hosted APIs only — OpenAI ASR + ElevenLabs TTS)

# Run
python main.py local in.mp4 russian         # dev CLI — target_lang is positional arg 2
python main.py local in.mp4 russian english # target_lang=russian source_lang=english
python main.py local in.mp4 russian --stop-after asr   # stop after one stage
python gradio_demo.py                        # browser UI on :7860 (no S3 needed)
python main.py runpod --input-file tests/fixtures/test_input.json  # S3/RunPod job

# Test (heavy: needs the gpu/api venv active)
pytest                           # whole suite
pytest tests/test_asr.py         # one file
pytest tests/test_asr.py::test_name  # one test

# Lint/format
ruff check . && ruff format .

# Build the RunPod serverless image (weights are NOT baked — mount MODEL_WEIGHTS_DIR
# as a volume; missing weights auto-download from HuggingFace on first job)
docker build -t langswap/video-translation-pipeline:4.0-base .
```

## Architecture (the parts that span multiple files)

- **Pipeline orchestration** — `langswap/translation_pipeline.py::VideoTranslationPipeline` runs the five stages (`_generate_asr` → `_generate_translation` → `_generate_speech` → `_merge` → `generate_srt_files`). All state flows through a single immutable-ish `VideoTranslation` dataclass (`langswap/pipeline_models/models.py`): each stage takes one and returns a freshly-constructed copy with more fields populated. `ChangeManager` re-runs TTS+merge for edited segments only.

- **Library entry vs. surfaces** — `langswap/api.py` (`process_translation` / `process_update_translation`) is the shared library API. Three deployment surfaces call it: `gradio_demo.py` (local UI), `serverless.py` (RunPod handler — its lazy import of `langswap.api` *inside* the handler is load-bearing; a module-level import delays handler registration past RunPod's init window), and `modal_app.py`. `main.py local` is the dev runner with per-stage JSON caching.

- **Backend dispatch** — each ML stage is a `*Manager` in `langswap/ml/<service>/` (`SpeechToTextManager` in `speech_to_text_manager.py`, `TranslationManager` in `translation_manager.py`, `TextToSpeechManager` in `tts_manager.py`; each service `__init__.py` is a one-line re-export). Each Manager picks a concrete `*_client.py` by a backend **string** via a hardcoded `if/elif` chain (unknown value → `ValueError`) and exposes one verb (transcribe / translate / synthesize). Every stage has exactly two backends: one local (GPU) and one hosted-API (no GPU). **Gotcha:** `langswap/backends.json` (via `backends.py`) only feeds the Gradio UI's option lists — it does **not** drive instantiation, so adding a backend means editing both the JSON and the Manager. Its `default` fields now match the real defaults (`vad` / `llamacpp` / `omnivoice`), which are also set in the Manager constructors and `TranslationPipelineConfig`.

- **Default backends** — each stage is local-by-default with an API fallback for no-GPU machines:
  - ASR: `vad` (local) — faster-whisper large-v3 + Silero VAD segmentation, no forced aligner, no per-language model. API fallback: `openai` (Whisper API). Chosen over Qwen-ASR after benchmarking: equal boundary accuracy, far lower weight.
  - Translation: `llamacpp` (local) — Gemma-4-12B-IT GGUF (`unsloth/gemma-4-12b-it-GGUF`) via llama-cpp-python; offloads to GPU when `device` is cuda, else CPU. A GGUF in-process engine avoids running a second vLLM alongside OmniVoice. API fallback: `openai`.
  - TTS: `omnivoice` (local) — OmniVoice via vllm-omni. API fallback: `elevenlabs`.

- **File repository** — every artifact (downloaded video, intermediate audio, SRT, result) round-trips through a `FileRepository` (`langswap/file_repository/`). `get_file(name)` returns a handle inside the job dir `data/<public_id>/`; `save_file()` persists it. `RemoteFileRepository` syncs to S3 (Yandex Cloud endpoint); `LocalOnlyFileRepository` stays on disk (`file://` URLs) and is what the CLI/Gradio use.

- **Model resolution & cache redirection** — importing `langswap/model_config.py` (transitively, importing `langswap.api`) has **import-time side effects**: it points `HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, vLLM and torch-inductor caches at `MODEL_WEIGHTS_DIR` (default `models_weights/`) so downloads and warmed compile caches are self-contained. Each client hardcodes its own model id/repo and auto-downloads it into `MODEL_WEIGHTS_DIR` on first use (no `resolve_model` indirection, no per-model env override).

- **Warm reuse** — `langswap/model_pool.py::get_or_create(key, factory)` caches constructed model holders process-globally and **always reuses** them, so ASR/translation/TTS engines stay co-resident across jobs and each pays its (slow) init once per process. This matters most on RunPod serverless. They share VRAM, so on a small GPU the local stack can OOM — use the API backends there.

- **Operator env vars (the whole set)** — `MODEL_WEIGHTS_DIR` (weights/cache dir), `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `BUCKET` (S3, for RunPod/S3 jobs), `HF_TOKEN` (pyannote diarization download), and optional `OPENAI_API_KEY` / `ELEVEN_API_KEY` (API backends) and `RUNPOD_API_KEY` (deploy). There are intentionally **no `LANGSWAP_*` tuning env vars** — model ids, GPU-offload, whisper device/dtype, and warm-reuse are all hardcoded to sensible defaults. Don't reintroduce env-var knobs; change the constant.

- **ctranslate2 / cuBLAS compatibility** — the Docker image is CUDA 13 (`cu130` torch/vLLM/OmniVoice), but ctranslate2 (faster-whisper's backend) links cuBLAS *12* (`libcublas.so.12`). The Dockerfile installs `nvidia-cublas-cu12` alongside cu13 to resolve this without overwriting the cu13 cuDNN that torch/vLLM rely on. When adding or upgrading ASR/ctranslate2 deps, check cuBLAS version compatibility.

- **No-GPU path** — for machines without a CUDA GPU, install `.[api]` and select the hosted backends (`openai` ASR + `openai` translation + `elevenlabs` TTS). They share the same Manager dispatch and `transcribe`/`translate`/`synthesize` verbs as the local clients.
