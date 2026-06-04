"""
eCardio Vodcast — Video Pipeline Config
=====================================
Central configuration for the talking-head pipeline (TTS → LatentSync → compose).
Edit BRAND[model]["voice"] to swap the TTS voice identity per model.
"""

import pathlib

# ---------------------------------------------------------------------------
# Paths — relative to the project root (wg-vodcast/)
# ---------------------------------------------------------------------------

BASE_DIR       = pathlib.Path(__file__).parent.parent   # wg-vodcast/
PIPELINE_DIR   = BASE_DIR / "video_pipeline"
AUDIO_DIR      = PIPELINE_DIR / "audio"
CLIPS_DIR      = PIPELINE_DIR / "final_clips"
INTERVIEW_JSON = BASE_DIR / "outputs" / "interview.json"

# ---------------------------------------------------------------------------
# Brand identity per AI panellist
#   color  — brand hue used for the lower-third and the style template
#   voice  — Kokoro voice id (+ lang_code: 'b' = British, 'a' = American)
# ---------------------------------------------------------------------------

BRAND: dict[str, dict] = {
    "claude": {
        "label":     "Claude",
        "provider":  "Anthropic  ·  claude-opus-4-7",
        "color_hex": "#D4763B",
        "color_rgb": (212, 118, 59),
        "voice":     "bf_emma",     # British female
        "lang_code": "b",
        "speed":     0.90,
        "gender":    "f",
    },
    "openai": {
        "label":     "ChatGPT",
        "provider":  "OpenAI  ·  gpt-5.5",
        "color_hex": "#10A37F",
        "color_rgb": (16, 163, 127),
        "voice":     "bm_lewis",    # British male
        "lang_code": "b",
        "speed":     0.90,
        "gender":    "m",
    },
    "gemini": {
        "label":     "Gemini",
        "provider":  "Google  ·  gemini-3.5-flash",
        "color_hex": "#4285F4",
        "color_rgb": (66, 133, 244),
        "voice":     "am_onyx",     # American male — deep
        "lang_code": "a",
        "speed":     0.90,
        "gender":    "m",
    },
}

# ---------------------------------------------------------------------------
# TTS settings (Kokoro)
# ---------------------------------------------------------------------------

TTS_SAMPLE_RATE   = 24_000      # Kokoro output sample rate (Hz)
TTS_SPEED         = 0.90        # fallback; per-model speed is BRAND[model]["speed"]

# ---------------------------------------------------------------------------
# Composition / encode settings
# ---------------------------------------------------------------------------

CLIP_FPS          = 24          # final output fps
RESOLUTION        = (1920, 1080)  # (width, height)
LOWER_THIRD_H     = 150         # pixel height of the lower-third bar
LOWER_THIRD_ALPHA = 190         # 0-255 transparency of the bar (190 ≈ 75% opaque)
AUDIO_BITRATE     = "192k"
