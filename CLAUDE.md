# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RunPod serverless worker that transcribes audio using [faster-whisper](https://github.com/guillaumekln/faster-whisper). Runs on RunPod GPU infrastructure. Docker image downloads all model weights at build time.

## Commands

**Install dependencies (local dev):**
```bash
pip install -r builder/requirements.txt
```

**Download model weights (required before running handler locally):**
```bash
python builder/fetch_models.py
```

**Run handler with test input:**
```bash
python src/rp_handler.py --test_input='{"input": {"audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"}}'
```

**Run with debug timing:**
```bash
python src/rp_handler.py --rp_debugger --test_input='{"input": {"audio": "..."}}'
```

**Build Docker image:**
```bash
docker build -t worker-faster-whisper .
```

## Architecture

```
builder/
  fetch_models.py   # Downloads all model weights at Docker build time
  requirements.txt  # runpod~=1.9.0, faster-whisper==1.1.0

src/
  rp_handler.py     # RunPod entrypoint — validates input, downloads audio, calls Predictor
  predict.py        # Predictor class — lazy model loading/unloading, transcription logic
  rp_schema.py      # INPUT_VALIDATIONS dict for runpod validator
```

**Request flow:** RunPod calls `run_whisper_job()` → validates via `rp_schema.INPUT_VALIDATIONS` → downloads audio (YouTube/Vimeo via yt-dlp, direct URLs via runpod downloader, or base64 decode) → calls `MODEL.predict()` → returns segments + transcription + optional translation.

**Model loading strategy:** `Predictor` lazy-loads models on first use. Only one model held in memory at a time — loading a different model evicts the current one via `gc.collect()`. Models run `float16` on CUDA, `int8` on CPU.

**Transcription vs translation:** `transcription` field formats the transcribed text (`plain_text`, `formatted_text`, `srt`, `vtt`). `translation` field formats English translation output (only produced when `translate=True`).

## Key constraints

- Python 3.10 only (pinned in CI and Dockerfile)
- Requires CUDA 12.3 + cuDNN 9 in production (Docker base image: `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`)
- `fetch_models.py` downloads **all** models at build time — adding a new model requires updating both `AVAILABLE_MODELS` in `predict.py` and `model_names` in `fetch_models.py`
- CI runs on actual RunPod GPU via self-hosted runner (secrets: `RUNPOD_ENDPOINT`, `RUNPOD_API_KEY`, `GH_PAT`, `GH_ORG`)
