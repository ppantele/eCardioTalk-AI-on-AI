# eCardio Vodcast — AI Panel Simulation

---

## Citation

How to cite: **TBA**.

---

A reproducible Python pipeline that interviews **Claude** (Anthropic), **ChatGPT** (OpenAI),
and **Gemini** (Google) on the future of AI in cardiovascular medicine, runs a 100-iteration
Monte Carlo simulation over each model's Likert (0–10) ratings, and (optionally) renders the
answers as lip-synced talking-head video.

There are **two stages**, with very different setup costs:

| Stage | What it does | Setup | Reproducible? |
|-------|--------------|-------|---------------|
| **1. Interview** (`run_ai_panel.py`) | LLM answers + Likert distributions + charts | `pip install` + API keys | Re-runs give *new* answers (LLMs are non-deterministic) — the committed results are the canonical record |
| **2. Video** (`video_pipeline/`) | TTS + lip-synced 1080p clips | Manual: 2 conda envs + multi-GB model weights + a GPU | Audio ≈ reproducible; video is diffusion-based and *not* bit-exact |

---

## What's already in this repo (and what isn't)

This repo **ships the finished study results** so the findings are preserved exactly and can be
cited without re-running anything:

- `outputs/interview.json` — every text answer + Likert stats + the raw 100-sample arrays
- `outputs/distributions/*.png` — the 16 histograms
- `outputs/raw/*.json` — raw Likert samples
- `external-templates/model-styles/*.mp4` — the three brand video templates

**Deliberately *not* committed** (gitignored — heavy, regenerable, or secret): API keys, all
audio/`*.wav`, all rendered `final_clips/*.mp4`, the upscaled `bases/`, and the third-party
video engines + model weights (LatentSync, RealESRGAN, etc.). See [Stage 2](#stage-2--video-pipeline-optional)
for how to install those.

> **Reproducibility:**
> The LLM APIs are non-deterministic (qualitative runs at temperature 0.7, Likert at 1.0–1.5
> *by design* to build a distribution), and the pinned model snapshots drift over time. The point
> of committing `outputs/` is that the **data is frozen**; you reproduce the *method*, not the exact
> tokens. Stage 2 audio is effectively reproducible (deterministic TTS); the final videos use a
> diffusion lip-sync model with no fixed seed, so they come out *equivalent*.

---

## Stage 1 — Interview pipeline

This is a fully `pip`-installable part.

### Prerequisites
- Python **3.10+**
- API keys for Anthropic, OpenAI, and Google (Gemini)
- Network access (no GPU needed)

### Setup
```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# add your API keys
cp secrets/api_keys.env.example secrets/api_keys.env
#   then edit secrets/api_keys.env and paste in your three keys
```

`secrets/api_keys.env` must contain (this file is gitignored and never committed):
```ini
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AI...
```

Model IDs, pricing, iteration count, and the hard cost ceiling live in `config.py`. **Verify the
model IDs and prices against your provider dashboards before a real run** — they change often, and
the pinned snapshots from the original run may no longer be available.

### Run
```bash
python run_ai_panel.py                 # dry-run: prints the cost estimate, no API calls
python run_ai_panel.py --mock          # full pipeline with fake responses ($0) — plumbing test
python run_ai_panel.py --execute       # real run (shows estimate, asks to confirm)
python run_ai_panel.py --execute --yes # skip the confirmation prompt
```
Useful flags: `--iterations N` (Likert samples per question/model, default 100),
`--questions-limit N` (first N questions only), `--max-cost USD` (abort ceiling).

### Outputs
`outputs/interview.json`, `outputs/distributions/*.png`, `outputs/raw/*.json`.

> ⚠️ `--mock` and `--execute` **overwrite** `outputs/`. The committed `outputs/` is the final
> collected run — copy it aside first if you want to keep it.

---

## Stage 2 — Video pipeline (optional)

Turns each answer into a 1080p lip-synced talking-head clip. This stage is **GPU-bound and not
one-command** — you must install two conda environments and several multi-GB model weights that
are *not* in this repo. The full, step-by-step setup, install, and run guide is in:

➡️ **[`video_pipeline/README_VIDEO.md`](video_pipeline/README_VIDEO.md)**

In short, you will need: an NVIDIA GPU; a `vodcast_video` conda env (PyTorch cu124, Kokoro TTS,
ffmpeg); a separate `latentsync` conda env with the LatentSync engine + checkpoints; and the
RealESRGAN x2plus weights. Then `bash video_pipeline/render_clips.sh` drives TTS → upscale →
lip-sync → compose.

---

## Repo layout

```
config.py                 ← models, pricing, iteration counts, paths
questions.txt             ← the 4 interview questions + Likert sub-questions
run_ai_panel.py           ← entry point: interview + Monte Carlo
ai_panel/                 ← pipeline modules (providers, montecarlo, viz, cost, io)
requirements.txt          ← Stage-1 Python deps
secrets/api_keys.env.example  ← copy to api_keys.env and fill in (gitignored)
outputs/                  ← interview.json, distributions/*.png, raw/*.json  (committed results)
reports/                  ← interview_qa.md, methodology.md (write-ups)
external-templates/model-styles/{claude,chatgpt,gemini}.mp4  ← brand video templates
video_pipeline/           ← Stage-2 talking-head video (see README_VIDEO.md)
```

Heavy/local artifacts (rendered videos, audio, model weights, third-party engines, secrets) are
gitignored — the repo tracks only code, config, the small brand templates, and the frozen results.

---

Code generation assisted by Claude Code.