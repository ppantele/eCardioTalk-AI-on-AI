#!/usr/bin/env python3
"""
Final 1080p compose/encode.
===========================
Takes the 1440p lip-synced video from LatentSync and produces the delivery clip:
  • supersample-downscale 2560×1440 → 1920×1080 (lanczos)  — crisper than a
    direct 720p→1080p upscale; this is the main image-quality win.
  • optionally overlay the brand lower-third (1920×150 PNG, no rescale).
  • upgrade the 24kHz mono TTS → 48kHz stereo AAC 192k (resample + loudnorm).
  • H.264 High / yuv420p / 24fps, CRF 16 with aq-mode=3 + psy-rd to protect the
    thin grid lines and the dark gradient background, +faststart.

Usage:
    python video_pipeline/compose_final.py \
        --synced /tmp/q1_claude_raw.mp4 \
        --audio  video_pipeline/audio/q1_claude.wav \
        --model  claude --no-lower-third --remove-sparkle \
        --out    video_pipeline/final_clips/q1_claude.mp4
"""

import argparse
import pathlib
import subprocess
import sys
from typing import Sequence

import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from video_pipeline.config_video import (
    BRAND, RESOLUTION, CLIP_FPS, LOWER_THIRD_H, LOWER_THIRD_ALPHA, AUDIO_BITRATE,
)

FFMPEG = "/opt/conda/envs/vodcast_video/bin/ffmpeg"
WIDTH, HEIGHT = RESOLUTION
LOWER_THIRD_DIR = pathlib.Path(__file__).parent / "lower_thirds"


def wav_duration(path: pathlib.Path) -> float:
    data, sr = sf.read(str(path))
    return len(data) / sr


def _find_font(size: int, bold: bool):
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf"]
             if bold else
             ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf"])
    roots = ["/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/liberation/",
             "/usr/share/fonts/truetype/freefont/"]
    for r in roots:
        for n in names:
            p = pathlib.Path(r) / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def build_lower_third(model_key: str, out_path: pathlib.Path) -> None:
    """Render a 1920 × LOWER_THIRD_H RGBA bar: [● dot] [LABEL] [provider]."""
    cfg = BRAND[model_key]
    cr, cg, cb = cfg["color_rgb"]
    img = Image.new("RGBA", (WIDTH, LOWER_THIRD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, WIDTH, LOWER_THIRD_H], fill=(10, 10, 12, LOWER_THIRD_ALPHA))
    dot_x, dot_y, dot_r = 60, LOWER_THIRD_H // 2, 22
    draw.ellipse([dot_x - dot_r, dot_y - dot_r - 4, dot_x + dot_r, dot_y + dot_r - 4],
                 fill=(cr, cg, cb, 230))
    text_x = dot_x + dot_r + 22
    draw.text((text_x, LOWER_THIRD_H // 2 - 38), cfg["label"],
              font=_find_font(52, True), fill=(255, 255, 255, 245))
    draw.text((text_x, LOWER_THIRD_H // 2 + 12), cfg["provider"],
              font=_find_font(26, False), fill=(190, 190, 195, 210))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))

# Delivery encode — exceeds the reference (720p ~1Mbps) at 1080p.
VIDEO_CRF_FINAL = 16
AUDIO_SR = 48000

# The model-style templates carry a decorative ✦ star in the bottom-right; this
# delogo box (in the 1440p synced-input coords, before the 1080p downscale)
# removes it. Position is consistent across the claude/chatgpt/gemini templates.
SPARKLE_BOX = "x=2300:y=1156:w=162:h=168"


def compose_final(synced: pathlib.Path, audio: pathlib.Path, model_key: str,
                  out_path: pathlib.Path, duration: float | None = None,
                  lower_third: bool = True, remove_sparkle: bool = False) -> None:
    if duration is None:
        duration = wav_duration(audio)
    delogo = f"delogo={SPARKLE_BOX}," if remove_sparkle else ""

    audio_fc = (
        # audio: 24k mono -> 48k stereo + loudnorm. loudnorm emits a double
        # sample-fmt, so aresample+aformat restore fltp/48k for the AAC encoder
        # (avoids err -22). Kept inside filter_complex to coexist with [v].
        "[1:a]pan=stereo|c0=c0|c1=c0,"
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        f"aresample={AUDIO_SR},"
        "aformat=sample_fmts=fltp[a]")
    vbase = f"[0:v]{delogo}scale={WIDTH}:{HEIGHT}:flags=lanczos,fps={CLIP_FPS},format=yuv420p"

    inputs = ["-i", str(synced), "-i", str(audio)]
    if lower_third:
        lt_png = LOWER_THIRD_DIR / f"{model_key}_lt.png"
        if not lt_png.exists():
            build_lower_third(model_key, lt_png)
        lt_y = HEIGHT - LOWER_THIRD_H        # 1080 - 150 = 930
        inputs += ["-i", str(lt_png)]        # input 2
        fc = (f"{vbase}[scaled];[scaled][2:v]overlay=0:{lt_y}[v];{audio_fc}")
    else:
        fc = f"{vbase}[v];{audio_fc}"

    cmd = [
        FFMPEG, "-y", *inputs,
        "-filter_complex", fc,
        "-map", "[v]", "-map", "[a]",
        # video
        "-c:v", "libx264", "-profile:v", "high", "-crf", str(VIDEO_CRF_FINAL),
        "-preset", "slow", "-pix_fmt", "yuv420p",
        "-x264-params", "aq-mode=3:psy-rd=1.0",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        str(out_path),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"final encode failed for {out_path.name}:\n{r.stderr[-1800:]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synced", required=True, type=pathlib.Path)
    p.add_argument("--audio", required=True, type=pathlib.Path)
    p.add_argument("--model", required=True, choices=list(BRAND))
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--no-lower-third", action="store_true",
                   help="Skip the brand legend overlay (the host adds it later).")
    p.add_argument("--remove-sparkle", action="store_true",
                   help="Delogo the bottom-right ✦ star from the style templates.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    print(f"\nStage 3b — Final 1080p compose: {a.out.name}")
    compose_final(a.synced, a.audio, a.model, a.out, lower_third=not a.no_lower_third,
                  remove_sparkle=a.remove_sparkle)
    mb = a.out.stat().st_size / 1_048_576
    print(f"  ✓ {a.out}  ({mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
