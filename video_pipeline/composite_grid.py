#!/usr/bin/env python3
"""
Frozen-figure + animated-grid composite — used to build the idle loops.
=======================================================================
For the "waiting" idle clips the figure should be still (sealed mouth, no
talking) but the scene should stay alive:
  • matte a frozen, sealed-lip android frame with MediaPipe selfie-segmentation;
  • composite that static figure over the ANIMATED brand base, so the grid/tunnel
    keeps shimmering behind the still figure.
The matte is computed once (the figure is static) and dilated to cover the
android's range in the grid source, then feathered for a soft seam. The result
is darkened and boomerang-looped (last frame = first) for seamless repeats.

Usage:
    python video_pipeline/composite_grid.py --frozen frozen_sealed.mp4 \
        --grid bases/gemini_styled_base_1440.mp4 --out idle_composite.mp4
"""
import argparse
import pathlib
import subprocess
import sys

import cv2
import numpy as np

FFMPEG = "/opt/conda/envs/vodcast_video/bin/ffmpeg"
DILATE = 71          # px kernel — must exceed the android's drift in the grid source
FEATHER = 41         # gaussian feather of the matte edge


def _matte(frame_bgr: np.ndarray) -> np.ndarray:
    import mediapipe as mp
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    m = seg.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).segmentation_mask
    mask = (m > 0.5).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((DILATE, DILATE), np.uint8))
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (FEATHER, FEATHER), 0)
    return np.clip(alpha, 0, 1)[..., None]


def composite(frozen: pathlib.Path, grid: pathlib.Path, out_path: pathlib.Path) -> None:
    fc = cv2.VideoCapture(str(frozen))
    fps = fc.get(cv2.CAP_PROP_FPS) or 25
    ok, f0 = fc.read()
    if not ok:
        raise RuntimeError(f"no frames in {frozen}")
    H, W = f0.shape[:2]
    alpha = _matte(f0)                       # figure is static → one matte

    # grid frames (animated base) cycled to the frozen length
    gc = cv2.VideoCapture(str(grid))
    grid_frames = []
    while True:
        ok, g = gc.read()
        if not ok:
            break
        grid_frames.append(g if (g.shape[0] == H and g.shape[1] == W)
                           else cv2.resize(g, (W, H)))
    gc.release()
    ng = len(grid_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "12",
         "-preset", "medium", "-pix_fmt", "yuv420p", str(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    i = 0
    fr = f0
    while True:
        g = grid_frames[i % ng].astype(np.float32)
        comp = (fr.astype(np.float32) * alpha + g * (1 - alpha)).astype(np.uint8)
        enc.stdin.write(comp.tobytes())
        i += 1
        ok, fr = fc.read()
        if not ok:
            break
    fc.release()
    enc.stdin.close()
    err = enc.stderr.read().decode("utf-8", "ignore")
    if enc.wait() != 0:
        raise RuntimeError(f"encode failed:\n{err[-1200:]}")
    print(f"  ✓ composite {out_path}  ({W}x{H}, {i} frames)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frozen", required=True, type=pathlib.Path)
    p.add_argument("--grid", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path)
    a = p.parse_args(argv)
    print(f"\nComposite (frozen figure + animated grid): {a.out.name}")
    composite(a.frozen, a.grid, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
