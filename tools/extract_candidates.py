#!/usr/bin/env python3
"""从战斗画面里抠出「数字精灵」候选 —— 后续训练数字分类器 + 调色域阈值的输入。

思路: 亮色掩码 → 连通域 → 过滤小碎块/大色块 → 同一行的字符合并成一个数字 token
      → 存小图 + CSV + 拼图（肉眼一看就知道靠不靠谱）

用法:
  python tools/extract_candidates.py inbox/battle.mp4 --fps 10 --roi 0,60,1280,540
  python tools/extract_candidates.py "inbox/*.png"

输出:
  out/cand/sprites/        每个候选的裁图  f000123_00.png
  out/cand/contact.png     候选拼图（重点看这个）
  out/cand/candidates.csv  frame,time,x,y,w,h,area,mean_bgr,mean_hsv
  out/cand/marked/         抽样帧 + 候选框（抽查漏检/误检）
"""
from __future__ import annotations

import argparse
import csv
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
    """deg = 顺时针旋转度数，只支持 0/90/180/270。"""
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def apply_roi(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def read_frames(src: str, fps: float, max_frames: int, roi, rotate: int = 0):
    """返回 [(frame_idx, timestamp, frame)]"""
    prep = lambda f: apply_roi(apply_rotate(f, rotate), roi)  # noqa: E731
    if "*" in src or Path(src).suffix.lower() in IMG_EXT:
        files = sorted(p for p in glob.glob(src) if Path(p).suffix.lower() in IMG_EXT)
        if not files:
            raise SystemExit(f"没找到图片: {src}")
        return [(i, float(i), prep(cv2.imread(f))) for i, f in enumerate(files[:max_frames])]

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"视频 {Path(src).name}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {src_fps:.1f}fps, "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / src_fps:.1f}s")
    step = max(1, int(round(src_fps / fps))) if fps > 0 else 1
    out, idx = [], 0
    while len(out) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out.append((idx, idx / src_fps, prep(frame)))
        idx += 1
    cap.release()
    return out


def bright_mask(frame_bgr: np.ndarray, v_min: int, s_min: int = 0) -> np.ndarray:
    """数字通常是「高亮度」的描边字：白字 (V高,S低) 或 橙/黄/绿字 (V高,S高)。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] >= v_min) & (hsv[:, :, 1] >= s_min)
    return (mask.astype(np.uint8)) * 255


def find_glyphs(mask: np.ndarray, min_h: int, max_h: int, min_area: int, max_area: int):
    """连通域 → 过滤成字符级方块。"""
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_h <= h <= max_h):
            continue
        if not (min_area <= area <= max_area):
            continue
        if w > 4 * h or w < 2:  # 太宽 → 大概率是血条/边框，不是数字
            continue
        fill = area / float(w * h)
        if fill < 0.15:  # 太稀疏 → 噪点/细线
            continue
        out.append((int(x), int(y), int(w), int(h), int(area), centroids[i]))
    return out


def merge_row(glyphs, gap_ratio: float = 1.2, ov_ratio: float = 0.5):
    """把同一行相邻字符合并成一个「数字 token」。"""
    boxes = [(x, y, w, h) for x, y, w, h, _, _ in glyphs]
    n = len(boxes)
    used = [False] * n
    merged = []
    for i in range(n):
        if used[i]:
            continue
        cur = list(boxes[i])
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if used[j]:
                    continue
                c = boxes[j]
                y1, y2 = max(cur[1], c[1]), min(cur[1] + cur[3], c[1] + c[3])
                ov = (y2 - y1) / max(1, min(cur[3], c[3]))
                gap = max(cur[0], c[0]) - min(cur[0] + cur[2], c[0] + c[2])
                if ov >= ov_ratio and gap <= gap_ratio * max(cur[3], c[3]):
                    x = min(cur[0], c[0])
                    y = min(cur[1], c[1])
                    x2 = max(cur[0] + cur[2], c[0] + c[2])
                    y2b = max(cur[1] + cur[3], c[1] + c[3])
                    cur = [x, y, x2 - x, y2b - y]
                    used[j] = True
                    changed = True
        merged.append(tuple(cur))
    return merged


def color_stats(frame_bgr: np.ndarray, box) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    x, y, w, h = box
    patch = frame_bgr[y : y + h, x : x + w]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 只统计"亮"的那部分像素，避免被背景拉偏
    sel = hsv[:, :, 2] >= int(np.percentile(hsv[:, :, 2], 80))
    if sel.sum() < 4:
        sel = np.ones(hsv.shape[:2], dtype=bool)
    mb = patch[sel].mean(axis=0)
    mh = hsv[sel].mean(axis=0)
    return tuple(int(v) for v in mb[::-1]), tuple(int(v) for v in mh)  # BGR->RGB


def contact_sheet(crops: list[np.ndarray], path: Path, cell_h: int = 44, cols: int = 16) -> None:
    if not crops:
        return
    cells = []
    for c in crops:
        h, w = c.shape[:2]
        scale = cell_h / h
        cells.append(cv2.resize(c, (max(1, int(w * scale)), cell_h), interpolation=cv2.INTER_NEAREST))
    rows = (len(cells) + cols - 1) // cols
    cell_w = max(c.shape[1] for c in cells) + 8
    sheet = np.full((rows * (cell_h + 8), cols * cell_w, 3), 40, np.uint8)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        y, x = r * (cell_h + 8) + 4, col * cell_w + 4
        sheet[y : y + cell_h, x : x + c.shape[1]] = c
    cv2.imwrite(str(path), sheet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--fps", type=float, default=10.0, help="采样帧率（回合制 10 够用）")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--roi", default=None, help="x,y,w,h")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="顺时针旋转度数（竖屏录制的素材需要 270）")
    ap.add_argument("--v-min", type=int, default=225, help="亮度阈值 0-255")
    ap.add_argument("--min-h", type=int, default=10, help="字符最小高度(px)")
    ap.add_argument("--max-h", type=int, default=70, help="字符最大高度(px)")
    ap.add_argument("--min-area", type=int, default=12)
    ap.add_argument("--max-area", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=40, help="每帧最多保留多少 token")
    ap.add_argument("--out", default="out/cand")
    ap.add_argument("--marked-every", type=int, default=20, help="每 N 帧存一张带框的抽查图")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "sprites").mkdir(parents=True, exist_ok=True)
    (out_dir / "marked").mkdir(parents=True, exist_ok=True)

    frames = read_frames(args.src, args.fps, args.max_frames, parse_roi(args.roi), args.rotate)
    if not frames:
        raise SystemExit("没取到帧")

    rows, crops = [], []
    for fi, (fidx, t, frame) in enumerate(frames):
        mask = bright_mask(frame, args.v_min)
        glyphs = find_glyphs(mask, args.min_h, args.max_h, args.min_area, args.max_area)
        tokens = merge_row(glyphs)
        # 越大的越可能是数字（过滤掉零碎高光）
        tokens = [b for b in tokens if b[2] >= 4 and b[3] >= args.min_h]
        tokens.sort(key=lambda b: -b[2] * b[3])
        tokens = tokens[: args.max_tokens]

        for k, (x, y, w, h) in enumerate(tokens):
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(frame.shape[1], x + w + 2), min(frame.shape[0], y + h + 2)
            crop = frame[y0:y1, x0:x1].copy()
            if crop.size == 0:
                continue
            rgb, hsv = color_stats(frame, (x, y, w, h))
            rows.append([fidx, f"{t:.2f}", x, y, w, h, w * h, rgb, hsv])
            crops.append(crop)
            cv2.imwrite(str(out_dir / "sprites" / f"f{fidx:06d}_{k:02d}.png"), crop)
            if len(rows) >= 20000:
                break

        if args.marked_every and fi % args.marked_every == 0:
            marked = frame.copy()
            for x, y, w, h in tokens:
                cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imwrite(str(out_dir / "marked" / f"f{fidx:06d}.png"), marked)
        if len(rows) >= 20000:
            break

    with open(out_dir / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "x", "y", "w", "h", "area", "mean_rgb", "mean_hsv"])
        w.writerows(rows)

    contact_sheet(crops[:800], out_dir / "contact.png")

    per_frame = len(rows) / max(1, len(frames))
    print(f"\n候选 {len(rows)} 个 / {len(frames)} 帧  ≈ {per_frame:.1f} 个/帧")
    print(f"  拼图   : {out_dir}/contact.png   ← 先看这个")
    print(f"  小图   : {out_dir}/sprites/")
    print(f"  抽查图 : {out_dir}/marked/")
    print(f"  CSV    : {out_dir}/candidates.csv")
    if per_frame > 30:
        print("  ⚠️  候选过多：UI 元素被当成数字了，提高 --v-min 或收紧 ROI")
    if per_frame < 0.2:
        print("  ⚠️  候选过少：降低 --v-min 或放宽 --min-h/--max-h")


if __name__ == "__main__":
    sys.exit(main())
