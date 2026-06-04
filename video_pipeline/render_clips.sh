#!/bin/bash
# Render the talking-head answer clips: TTS audio + brand style template
# → RealESRGAN 1440p base → LatentSync → 1080p compose.
#
#   bash video_pipeline/render_clips.sh            # all questions × 3 models
#   bash video_pipeline/render_clips.sh q1         # one question, 3 models
#
# Sequential by design — each LatentSync render peaks ~100-140 GB RAM at 1440p;
# never run two in parallel (they OOM-kill each other).
set -u
# Repo root, derived from this script's location (video_pipeline/render_clips.sh).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Conda + the vodcast_video python; override via env if your install differs.
CONDA="${CONDA:-/opt/conda/bin/conda}"
PY="${PY:-/opt/conda/envs/vodcast_video/bin/python}"
LS=$ROOT/video_pipeline/latentsync
cd "$ROOT"

# model -> style template basename in external-templates/model-styles/
declare -A STYLE=( [claude]=claude [openai]=chatgpt [gemini]=gemini )
QUESTIONS=( "${@:-q1 q2 q3 q4}" )
OUT_DIR=$ROOT/video_pipeline/final_clips
mkdir -p "$OUT_DIR"
pkill -9 -f scripts.inference 2>/dev/null; sleep 3
echo "RENDER_START $(date +%H:%M:%S)"

for Q in ${QUESTIONS[@]}; do
  for M in claude openai gemini; do
    S=${STYLE[$M]}
    BASE=$ROOT/video_pipeline/bases/${M}_styled_base_1440.mp4
    AUD=$ROOT/video_pipeline/audio/${Q}_${M}.wav
    RAW=/tmp/${Q}_${M}_raw.mp4
    OUT=$OUT_DIR/${Q}_${M}.mp4
    echo "=== $Q $M (style=$S, free_GB=$(free -g | awk '/Mem/{print $7}')) $(date +%H:%M:%S) ==="

    # Build the 1440p brand base once (RealESRGAN upscale of the style template).
    if [ ! -s "$BASE" ]; then
      $PY video_pipeline/upscale_video.py --in external-templates/model-styles/$S.mp4 \
          --out "$BASE" --enhance realesrgan >/tmp/up_${M}.log 2>&1 \
          && echo "$M base ok" || { echo "$M BASE_FAIL"; tail -4 /tmp/up_${M}.log; continue; }
    fi

    # LatentSync (its own env) boomerang-loops the base to the audio length.
    ( cd "$LS" && export LATENTSYNC_LM_SMOOTH=1 && $CONDA run --no-capture-output -n latentsync \
        python -m scripts.inference --unet_config_path configs/unet/stage2_512.yaml \
        --inference_ckpt_path checkpoints/latentsync_unet.pt --inference_steps 20 --guidance_scale 1.0 \
        --video_path "$BASE" --audio_path "$AUD" --video_out_path "$RAW" ) >/tmp/ls_${Q}_${M}.log 2>&1 \
        && echo "$M ls ok" || { echo "${Q}_${M}_LS_FAIL"; tail -4 /tmp/ls_${Q}_${M}.log; rm -f "$RAW"; continue; }

    # 1440p → 1080p, 48k stereo audio, ✦ sparkle removed, no baked-in legend.
    $PY video_pipeline/compose_final.py --synced "$RAW" --audio "$AUD" --model "$M" \
        --no-lower-third --remove-sparkle --out "$OUT" >/dev/null 2>&1 \
        && echo "${Q}_${M}_DELIVERED" || echo "${Q}_${M}_COMPOSE_FAIL"
    rm -f "$RAW"
  done
  echo "QUESTION_${Q}_DONE $(date +%H:%M:%S)"
done
echo "RENDER_DONE $(date +%H:%M:%S)"
