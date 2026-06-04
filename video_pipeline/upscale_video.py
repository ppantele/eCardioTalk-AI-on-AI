#!/usr/bin/env python3
"""
Upscale a video with RealESRGAN x2plus (content-agnostic).

Brings each brand's 720p style template
(external-templates/model-styles/{style}.mp4) up to 1440p so LatentSync's 512px
face processing works from a high-resolution face region — the synthesised
mouth then matches the resolution of the (already-sharp) shoulders/background.

Usage:
    python video_pipeline/upscale_video.py --in style.mp4 --out base_1440.mp4
    python video_pipeline/upscale_video.py --in a.mp4 --out b.mp4 --enhance lanczos
"""
import argparse
import os
import pathlib
import subprocess
import sys

import cv2
import numpy as np

HALLO2_DIR = pathlib.Path(__file__).parent / "hallo2"
FFMPEG = "/opt/conda/envs/vodcast_video/bin/ffmpeg"


class Upscaler:
    """RealESRGAN x2plus (content-agnostic) or a lanczos fallback.

    RealESRGAN's RealESRGANer lives in the vendored hallo2/basicsr (the pip
    basicsr lacks utils.realesrgan_utils), so we chdir + insert that dir.
    """

    def __init__(self, mode: str, scale: int = 2):
        self.mode = mode
        self.scale = scale
        self.engine = None
        if mode == "realesrgan":
            cwd = os.getcwd()
            os.chdir(str(HALLO2_DIR))            # vendored basicsr has realesrgan_utils
            sys.path.insert(0, str(HALLO2_DIR))
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from basicsr.utils.realesrgan_utils import RealESRGANer
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=2)
                self.engine = RealESRGANer(
                    scale=2,
                    model_path="./pretrained_models/realesrgan/RealESRGAN_x2plus.pth",
                    model=net, tile=0, tile_pad=10, pre_pad=0, half=True, device="cuda")
            finally:
                os.chdir(cwd)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == "realesrgan":
            out, _ = self.engine.enhance(frame, outscale=self.scale)
            return out
        if self.mode == "lanczos":
            h, w = frame.shape[:2]
            return cv2.resize(frame, (w * self.scale, h * self.scale),
                              interpolation=cv2.INTER_LANCZOS4)
        return frame  # none


def upscale(in_path: pathlib.Path, out_path: pathlib.Path, enhance: str, fps: float) -> None:
    cap = cv2.VideoCapture(str(in_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"no frames in {in_path}")
    up = Upscaler(enhance)
    scale = up.scale if enhance != "none" else 1
    h, w = first.shape[:2]
    out_w, out_h = w * scale, h * scale
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{out_h}", "-r", str(src_fps), "-i", "-",
         "-an", "-c:v", "libx264", "-crf", "12", "-preset", "medium",
         "-pix_fmt", "yuv420p", str(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    n = 0
    frame = first
    while True:
        hd = up(frame) if enhance != "none" else frame
        enc.stdin.write(hd.astype(np.uint8).tobytes())
        n += 1
        ok, frame = cap.read()
        if not ok:
            break
        if n % 50 == 0:
            print(f"\r  upscaling… {n}", end="", flush=True)
    cap.release()
    print()
    enc.stdin.close()
    err = enc.stderr.read().decode("utf-8", "ignore")
    if enc.wait() != 0:
        raise RuntimeError(f"encoder failed:\n{err[-1500:]}")
    print(f"  ✓ {out_path}  ({out_w}x{out_h}, {n} frames)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--enhance", choices=["realesrgan", "lanczos", "none"], default="realesrgan")
    p.add_argument("--fps", type=float, default=24.0)
    a = p.parse_args(argv)
    print(f"\nUpscale ({a.enhance}): {a.inp.name}")
    upscale(a.inp, a.out, a.enhance, a.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
