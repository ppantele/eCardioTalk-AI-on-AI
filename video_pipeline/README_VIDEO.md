# eCardio Vodcast — Video Pipeline

Turns each model's interview answer into a **1080p talking-head clip** in the visual
style of the per-brand templates `external-templates/model-styles/{claude,chatgpt,gemini}.mp4`
— a dark metallic android bust on an animated sci-fi grid — restyled per brand and
lip-synced to the model's TTS narration.

Per `(question, model)` the chain is:

```
TTS (Kokoro)  ─┐
               ├─►  RealESRGAN 1440p base  ─►  LatentSync (512px sync)  ─►  1080p compose
brand style ───┘       (upscale_video.py)        (env: latentsync)          (compose_final.py)
template
```

Output: **12 answer clips** (`q{1-4}_{claude,openai,gemini}.mp4`) in `final_clips/`,
plus 3 idle loops. Brand colours: Claude amber `#D4763B`,
ChatGPT emerald `#10A37F`, Gemini blue `#4285F4`.

---

## Prerequisites

This stage is **not** `pip install && run`. You must build two conda environments and download
several multi-GB model weights that are **not** in this repo (they are third-party and gitignored).
Budget time for it.

- **NVIDIA GPU** with CUDA 12.4-class drivers. Developed on an **H100 80 GB**; LatentSync at 1440p
  also peaks at **~100–140 GB system RAM per render**, so renders run strictly one at a time.
- **conda** (Miniconda/Anaconda). The helper scripts assume conda is installed at `/opt/conda` and
  that the env is named `vodcast_video` (see "Path assumptions" below if yours differ).
- **ffmpeg** on PATH (and inside the `vodcast_video` env). This build is *not* compiled with
  libsoxr — see the ffmpeg gotchas at the bottom.
- **Two conda environments**:
  - **`vodcast_video`** — Kokoro TTS, RealESRGAN, OpenCV/MediaPipe, ffmpeg (torch cu124).
  - **`latentsync`** — the LatentSync lip-sync engine and its own dependencies.
- Tens of GB of free disk for the model weights and rendered clips.

---

## Setup & installation

Run everything from the **repo root** (`cd` into your clone first).

### A. `vodcast_video` env — TTS, upscale, compose
```bash
conda create -n vodcast_video python=3.11 -y
conda activate vodcast_video
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r video_pipeline/requirements_video.txt
# ffmpeg must be on PATH inside this env:
conda install -c conda-forge ffmpeg -y      # or: sudo apt-get install ffmpeg
conda deactivate
```
Kokoro's voice weights are fetched automatically by the `kokoro` pip package on first use — no
manual download.

### B. RealESRGAN x2plus weights (the upscaler)
`upscale_video.py` loads `RealESRGANer` from a **vendored** `basicsr` (the pip `basicsr` lacks
`utils.realesrgan_utils`) and expects the weight at an exact path:
```
video_pipeline/hallo2/basicsr/                                  ← vendored basicsr (has utils/realesrgan_utils.py)
video_pipeline/hallo2/pretrained_models/realesrgan/RealESRGAN_x2plus.pth
```
Get the vendored `basicsr` from the **hallo2** repo (https://github.com/fudan-generative-vision/hallo2)
and the weight `RealESRGAN_x2plus.pth` from **Real-ESRGAN** releases
(https://github.com/xinntao/Real-ESRGAN). Place them at the two paths above.

### C. `latentsync` env + engine + checkpoints
Clone the engine into `video_pipeline/latentsync/` and install it per its own instructions:
```bash
git clone https://github.com/bytedance/LatentSync video_pipeline/latentsync
cd video_pipeline/latentsync
# follow LatentSync's README to create its conda env and download checkpoints:
#   - create the env and name it `latentsync`   (e.g. `conda create -n latentsync python=3.10`)
#   - install its requirements
#   - download its checkpoints (HuggingFace) into video_pipeline/latentsync/checkpoints/
cd ../..
```
The render driver invokes LatentSync as `conda run -n latentsync python -m scripts.inference` and
expects, relative to `video_pipeline/latentsync/`:
```
configs/unet/stage2_512.yaml          ← shipped with the LatentSync clone
checkpoints/latentsync_unet.pt        ← downloaded checkpoint (plus its auxiliary models, e.g. whisper)
```

### D. Path assumptions (adjust if your setup differs)
- `render_clips.sh` defaults to `CONDA=/opt/conda/bin/conda` and
  `PY=/opt/conda/envs/vodcast_video/bin/python`. Override per-run with env vars, e.g.
  `CONDA=/your/conda PY=/your/envs/vodcast_video/bin/python bash video_pipeline/render_clips.sh`.
- `upscale_video.py` hardcodes `FFMPEG = "/opt/conda/envs/vodcast_video/bin/ffmpeg"`. If your env
  lives elsewhere, edit that one line (or symlink) so it points at a real ffmpeg.

---

## Run (after setup is complete)

```bash
cd /path/to/wg-vodcast   # the repo root

# 1. TTS narration → video_pipeline/audio/q{1-4}_{model}.wav  (env: vodcast_video)
#    Reads the answer text from outputs/interview.json.
conda run -n vodcast_video python video_pipeline/generate_tts.py

# 2. Render the talking-head clips (TTS → 1440p base → LatentSync → 1080p)
bash video_pipeline/render_clips.sh          # all questions × 3 models
bash video_pipeline/render_clips.sh q1       # just Q1, 3 models
```

Outputs → `video_pipeline/final_clips/q{1-4}_{claude,openai,gemini}.mp4`.

`render_clips.sh` runs **sequentially** by design: each LatentSync render peaks at
~100-140 GB RAM at 1440p, so two in parallel OOM-kill each other. On a fresh clone it builds each
1440p brand base once (cached in `bases/`), then lip-syncs and composes.

> **Reproducibility note:** the **audio** is effectively reproducible — Kokoro TTS is deterministic
> for the same text/voice/version, and it reads the committed `outputs/interview.json`. The **final
> videos are not bit-exact**: LatentSync is a diffusion model and `render_clips.sh` sets no fixed
> seed, so each render is *visually equivalent* but not identical. The `bases/` are a deterministic
> RealESRGAN upscale of the committed brand templates and regenerate automatically when missing.

---

## File structure

```
video_pipeline/
├── config_video.py     ← brand colours, voice IDs, paths, encode settings
├── generate_tts.py     ← Kokoro TTS  → audio/
├── upscale_video.py    ← RealESRGAN x2plus: style template 720p → 1440p base
├── compose_final.py    ← 1440p→1080p, 48k stereo audio, ✦ removal, optional legend
├── composite_grid.py   ← freeze a sealed-lip frame + animated grid (idle loops)
├── render_clips.sh     ← end-to-end driver (base → LatentSync → compose)
├── requirements_video.txt
└── README_VIDEO.md

video_pipeline/bases/        ← cached 1440p brand bases (gitignored)
video_pipeline/audio/        ← TTS WAVs (gitignored)
video_pipeline/final_clips/  ← delivered MP4s (gitignored — heavy)
external-templates/model-styles/*.mp4     ← per-brand restyled templates (the base source)
```

Third-party engines (`latentsync/`, `hallo2/` for RealESRGAN weights) are installed
locally and gitignored.

---

## Pipeline details

- **`upscale_video.py`** — RealESRGAN x2plus brings the 720p brand style template to
  1440p so LatentSync's 512px face crop has a high-resolution face to work from (the
  synthesised mouth then matches the sharpness of the shoulders/grid). RealESRGAN's
  `RealESRGANer` is loaded from the vendored `hallo2/basicsr` (the pip `basicsr` lacks
  `utils.realesrgan_utils`).
- **LatentSync** (ByteDance) — `stage2_512`, `inference_steps=20`, `guidance_scale=1.0`
  (lower CFG = calmer mouth, less hallucination). It boomerang-loops the base to the
  audio length and repaints only the mouth, preserving the metallic head.
- **`compose_final.py`** — supersamples 1440p→1080p (lanczos), upgrades the 24 kHz mono
  TTS to 48 kHz stereo AAC 192k, removes the templates' decorative ✦ star (`--remove-sparkle`),
  and encodes H.264 High / yuv420p / CRF 16 / aq-mode=3 + psy-rd / +faststart. The brand
  legend is left off (`--no-lower-third`) for manual lower-thirds in the edit.

### Idle clips & intro/outro

- **Idle loops** — `composite_grid.py` freezes a sealed-lip, level frame of the brand base
  and composites the animated grid back behind it (MediaPipe matte), darkened, then
  boomerang-looped so the last frame equals the first (seamless repeat).
- **Intro / outro** — the host's opening and closing are entered by hand as `<<EDIT>>`
  placeholders in `outputs/interview.json` (`intro`, `closing_remarks`); they are not
  auto-generated by this pipeline.

---

## Tuning

| What | Where |
|------|-------|
| Voice per model | `BRAND[…]["voice"]` + `["lang_code"]` in `config_video.py` |
| TTS speed | `BRAND[…]["speed"]` (fallback `TTS_SPEED`) |
| Brand colour / label | `BRAND[…]` in `config_video.py` |
| Lip-sync strength | `--guidance_scale` / `--inference_steps` in `render_clips.sh` |
| Final encode quality | `VIDEO_CRF_FINAL` in `compose_final.py` |

Current voices: **Claude `bf_emma`** (British female), **ChatGPT `bm_lewis`** (British
male), **Gemini `am_onyx`** (American male, deep).

---

## ffmpeg gotchas (this build)

- Not built with libsoxr → use plain `aresample=48000` (never `resampler=soxr`).
- `loudnorm` emits a double sample-fmt → follow with `aresample,aformat=sample_fmts=fltp`
  or the native AAC encoder fails with err -22.
- With `-filter_complex`, put audio filters **inside** it (`[1:a]…[a]`), not `-af`.

---

## Citation

How to cite: **TBA**.
