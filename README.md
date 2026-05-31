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
🎙️ Qwen3-ASR  →  🌐 Gemma  →  🗣️ OmniVoice  →  🎚️ dubbing  →  🎬 video + SRT
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
> **First run:** download weights into `./models_weights` with `langswap-download-models --all` (set `HF_TOKEN` for gated models like Gemma). Missing models are fetched on demand if `HF_TOKEN` is set.

---

## ⚙️ Configuration

Create a `.env` in the project root — it's loaded automatically:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx            # gated models (Gemma, pyannote)
ELEVEN_API_KEY=...                       # only for the ElevenLabs TTS backend
MODEL_WEIGHTS_DIR=./models_weights       # where weights live (mounted into Docker)
LANGSWAP_DATA_DIR=./data                 # where outputs/artifacts go
```

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | HuggingFace token for gated models (Gemma, pyannote) |
| `ELEVEN_API_KEY` | ElevenLabs key — only for the `elevenlabs` TTS backend |
| `MODEL_WEIGHTS_DIR` | Where weights are stored/loaded (default `./models_weights`) |
| `LANGSWAP_DATA_DIR` | Where outputs and intermediate artifacts go (default `data/`) |

Full list — model overrides, GPU tuning, alternative backends — is in **[docs/ADVANCED.md](docs/ADVANCED.md)**.

---

## 🛠️ Run from source

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[demo]"

python gradio_demo.py                          # browser UI  → http://localhost:7860
python debug_local.py in.mp4 english russian   # CLI, stage-by-stage
```

Everything runs in one process on transformers 5.x — ASR included.

---

## 📚 Documentation

**[docs/ADVANCED.md](docs/ADVANCED.md)** — model registry, exact pinned install, every env var, the optional multi-container (Compose) stack, and troubleshooting.

---

## 📄 License

Code is **AGPL-3.0-or-later** (see [`LICENSE`](LICENSE)) — run a modified version as a network service and you must publish your source. A **commercial license** is available: ilya@langswap.app.

**Model weights are licensed separately** and some (Gemma, Qwen, OmniVoice) restrict commercial use — you're responsible for complying with each model's terms. The AGPL grant on this code does **not** cover the weights. Contributions are governed by [`CONTRIBUTING.md`](CONTRIBUTING.md).
