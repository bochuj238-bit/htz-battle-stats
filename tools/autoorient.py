#!/usr/bin/env python3
"""自动判定视频/图片的正确朝向。

原理：用「第 N 回合」这个必然存在的 UI 文本当基准 —— 四个旋转角都试一遍，
哪个能 OCR 出「第…回合」哪个就是对的。顺带报告每个角度的得分，便于排查。

用法:
  python tools/autoorient.py video.mp4
  python tools/autoorient.py video.mp4 --at 30 --dump out/orient
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
ROUND_RE = re.compile(r"第\s*(\d+)\s*回合")


def rotate(frame: np.ndarray, deg: int) -> np.ndarray:
    op = ROTATIONS[deg]
    return frame if op is None else cv2.rotate(frame, op)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--at", type=float, default=10.0, help="用哪个时刻的帧判定")
    ap.add_argument("--samples", type=int, default=3, help="多取几帧投票")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    eng = RapidOCR()
    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"打不开: {args.src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    for k in range(args.samples):
        t = args.at + k * 2.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        raise SystemExit("没取到帧")
    print(f"原始尺寸（cv2 读数，不含旋转元数据）: {frames[0].shape[1]}x{frames[0].shape[0]}")

    best_deg, best_score = 0, -1
    for deg in (0, 90, 180, 270):
        score, hits = 0, []
        for fr in frames:
            rot = rotate(fr, deg)
            # 回合号在左上角，只 OCR 那一片，快且准
            h, w = rot.shape[:2]
            crop = rot[0 : int(h * 0.20), 0 : int(w * 0.40)]
            res = eng(crop)
            if res is not None and res.txts:
                for txt in res.txts:
                    m = ROUND_RE.search(txt.replace(" ", ""))
                    if m:
                        score += 1
                        hits.append(m.group(0))
        print(f"  rotate={deg:3d} → 命中 {score}/{len(frames)}  {hits}")
        if score > best_score:
            best_deg, best_score = deg, score

    print(f"\n=> 正确旋转: --rotate {best_deg}   (命中率 {best_score}/{len(frames)})")
    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"rot{best_deg}.png"), rotate(frames[0], best_deg))


if __name__ == "__main__":
    main()
