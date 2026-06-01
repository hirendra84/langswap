<div align="center">

# 🎬 langswap

**Dub any video into another language — on your own GPU, end to end.**

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](#-quick-start)
[![Stars](https://img.shields.io/github/stars/langswap-app/langswap?style=social)](https://github.com/langswap-app/langswap)

<!-- TODO: drop a short screen capture at docs/demo.gif and uncomment the line below.
     Best result: source video on the left, dubbed video on the right, ~10s loop. -->
<!-- ![langswap demo](docs/demo.gif) -->
_📽️ demo GIF coming soon_

</div>

---

## ✨ What it does

Speech recognition → translation → voice-cloned text-to-speech → dubbing → a finished video with subtitles. No cloud services, no per-minute fees.

```
🎙️ ASR  →  🌐 LLM Tranlation  →  🗣️ TTS  →  🎚️ dubbing  →  🎬 video + SRT
```

- 🌍 **Any language pair** — auto-detects the source, dubs into your target
- 🗣️ **Voice cloning** keeps the original speaker's voice (OmniVoice; XTTS / F5-TTS / Qwen3-TTS / ElevenLabs also supported)
- 🎚️ **Lip-/timing-aware dubbing** with `speedup` and `stretch_whole` algorithms
- 🖥️ **One container, one GPU** — runs entirely local, browser UI included
- 📝 Outputs the dubbed video **plus source & translated SRT** subtitles

---

## 🚀 Quick start

```bash
git clone https://github.com/langswap-app/langswap
cd langswap

docker build -t langswap .
docker run --rm --gpus all -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$PWD/models_weights:/models" \
  -v "$PWD/data:/app/data" \
  langswap
```

Open **http://localhost:7860**, drop in a video, pick a target language. Done.

> **Needs:** an NVIDIA GPU + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
> **First run:** models auto-download into `./models_weights` on first use (set `HF_TOKEN` for gated models like Gemma). Pre-place weights there to skip the download.

---

## 🛠️ Run from source

```bash
uv venv --python 3.12 && source .venv/bin/activate

uv pip install -e ".[gpu]"     # full local-model stack (needs an NVIDIA GPU)
# or, on a Mac / no GPU — relies on hosted APIs, far fewer deps:
uv pip install -e ".[api]"

python gradio_demo.py                          # browser UI  → http://localhost:7860
python main.py local in.mp4 english russian    # CLI, stage-by-stage
```

The full GPU install runs everything in one process on transformers 5.x — ASR included.
See [docs/advanced.md](docs/advanced.md) for the exact GPU install (torch cu130 + `qwen-asr`/`qwen-tts`).

---

## 📚 Documentation

**[docs/advanced.md](docs/advanced.md)** — model list & overrides, exact pinned install, every env var, Docker build notes, and troubleshooting.

---

## 📄 License

Code is **AGPL-3.0-or-later** (see [`LICENSE`](LICENSE)) — run a modified version as a network service and you must publish your source. A **commercial license** is available: ilya@langswap.app.

**Model weights are licensed separately** and some (Gemma, Qwen, OmniVoice) restrict commercial use — you're responsible for complying with each model's terms. The AGPL grant on this code does **not** cover the weights. Contributions are governed by [`CONTRIBUTING.md`](CONTRIBUTING.md).
