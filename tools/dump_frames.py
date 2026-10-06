#!/usr/bin/env python3
"""按时刻导出整帧（方向按 --rotate 处理），交给视觉模型看。

用法:
  python tools/dump_frames.py "video.mp4" --rotate 270 --at 16.2,40,80 --out out/frames_bd
  python tools/dump_frames.py "video.mp4" --rotate 0 --at 12,24 --scale 1600 --out out/frames_ce
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_candidates import apply_rotate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--at", required=True, help="逗号分隔的时刻，单位秒")
    ap.add_argument("--scale", type=int, default=1400, help="导出宽度（0=原分辨率）")
    ap.add_argument("--out", default="out/frames")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"打不开: {args.src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for t in [float(v) for v in args.at.split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            print(f"  t={t}s 取帧失败")
            continue
        fr = apply_rotate(fr, args.rotate)
        if args.scale:
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (args.scale, int(h * args.scale / w)), interpolation=cv2.INTER_AREA)
        p = out / f"frame_t{t:07.2f}.png"
        cv2.imwrite(str(p), fr)
        print(f"  {p}  {fr.shape[1]}x{fr.shape[0]}")
    cap.release()


if __name__ == "__main__":
    main()
