#!/usr/bin/env python3
"""素材体检：看看这批录屏/截图能不能用来做识别。

用法:
  python tools/probe.py inbox/battle.mp4
  python tools/probe.py inbox/battle.mp4 --fps 1 --roi 0,0,1280,600
  python tools/probe.py "inbox/*.png"

输出:
  out/probe/frame_XXXX.png   采样帧（肉眼检查用）
  out/probe/roi_preview.png  ROI 预览（如果指定了 --roi）
  终端打印: 分辨率/帧率/时长 + 颜色分布 + 亮色像素占比
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_roi(s: str | None):
    if not s:
        return None
    parts = [int(v) for v in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise SystemExit("--roi 格式应为 x,y,w,h")
    return tuple(parts)


def apply_rotate(frame: np.ndarray, deg: int) -> np.ndarray:
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def apply_roi(frame: np.ndarray, roi) -> np.ndarray:
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def video_frames(path: Path, fps: float, max_frames: int, roi, rotate: int = 0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  分辨率 : {w}x{h}")
    print(f"  帧率   : {src_fps:.2f} fps")
    print(f"  时长   : {total / src_fps:.1f} s ({total} 帧)")

    step = max(1, int(round(src_fps / fps))) if fps > 0 else 1
    out, idx = [], 0
    while len(out) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out.append((idx, idx / src_fps, apply_roi(apply_rotate(frame, rotate), roi)))
        idx += 1
    cap.release()
    return out


def image_frames(pattern: str, roi, rotate: int = 0):
    files = sorted(p for p in glob.glob(pattern) if Path(p).suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"没找到图片: {pattern}")
    out = []
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        out.append((i, float(i), apply_roi(apply_rotate(img, rotate), roi)))
    return out


def palette_report(frames: list[np.ndarray], top_n: int = 12) -> None:
    """量化到 16 级/通道，统计出现最多的颜色 —— 用来确认伤害/治疗数字的色系。"""
    px = np.concatenate([f.reshape(-1, 3) for f in frames], axis=0).astype(np.uint16)
    q = (px // 16) * 16 + 8
    keys = (q[:, 0].astype(np.uint32) << 16) | (q[:, 1].astype(np.uint32) << 8) | q[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    order = np.argsort(counts)[::-1][:top_n]
    total = keys.size
    print(f"\n颜色分布 (BGR, 16级量化, 采样 {total:,} 像素):")
    for i in order:
        k = int(uniq[i])
        b, g, r = (k >> 16) & 255, (k >> 8) & 255, k & 255
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
        print(
            f"  BGR({b:3d},{g:3d},{r:3d})  HSV(H{hsv[0]:3d},S{hsv[1]:3d},V{hsv[2]:3d})  "
            f"{counts[i] / total * 100:5.2f}%"
        )

    bright = np.concatenate(
        [(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 2] >= 225).reshape(-1) for f in frames]
    )
    print(f"\n亮色像素(V>=225)占比: {bright.mean() * 100:.2f}%  ← 数字候选的可用信号量")
    if bright.mean() < 0.002:
        print("  ⚠️  占比过低，可能画质压缩太狠或数字偏暗，需要调阈值")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="视频路径 或 图片 glob")
    ap.add_argument("--fps", type=float, default=1.0, help="每秒采样多少帧 (默认1)")
    ap.add_argument("--max-frames", type=int, default=12, help="最多采样多少帧")
    ap.add_argument("--roi", default=None, help="x,y,w,h 只看战斗区")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="顺时针旋转度数（竖屏录制的素材需要 270）")
    ap.add_argument("--out", default="out/probe")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    roi = parse_roi(args.roi)

    print(f"素材: {args.src}")
    if Path(args.src).suffix.lower() in IMG_EXT or "*" in args.src:
        items = image_frames(args.src, roi, args.rotate)
    else:
        items = video_frames(Path(args.src), args.fps, args.max_frames, roi, args.rotate)

    if not items:
        raise SystemExit("没取到任何帧")

    for n, (idx, t, frame) in enumerate(items):
        cv2.imwrite(str(out_dir / f"frame_{idx:05d}.png"), frame)

    if roi:
        x, y, w, h = roi
        prev = items[0][2].copy()
        cv2.rectangle(prev, (0, 0), (w - 1, h - 1), (0, 255, 255), 2)
        cv2.imwrite(str(out_dir / "roi_preview.png"), prev)

    print(f"\n采样 {len(items)} 帧 → {out_dir}/frame_*.png")
    palette_report([f for _, _, f in items])


if __name__ == "__main__":
    sys.exit(main())
