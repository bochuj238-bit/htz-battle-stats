#!/usr/bin/env python3
"""定向区域 OCR：先找「亮色+暖色活动区」，再对每个区域做放大 OCR。

为什么要这样：RapidOCR 的检测器在满帧花花绿绿画面上会漏掉伤害数字，
但在小裁图上（相当于放大）能 100% 读对。所以先粗定位，再局部识别。

用法:
  python tools/ocr_region.py /tmp/f44_native.png
  python tools/ocr_region.py inbox/battle.mp4 --fps 2 --rotate 270 --max-frames 60
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

ENGINE: RapidOCR | None = None


def get_engine() -> RapidOCR:
    global ENGINE
    if ENGINE is None:
        ENGINE = RapidOCR()
    return ENGINE


def activity_regions(frame: np.ndarray, min_area: int = 900) -> list[tuple[int, int, int, int]]:
    """亮色 + 暖色活动区（伤害/治疗数字都在这上面）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    warm = cv2.inRange(hsv, np.array([5, 90, 150]), np.array([35, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 232]), np.array([180, 80, 255]))
    m = cv2.bitwise_or(warm, cv2.dilate(white, np.ones((3, 3), np.uint8), iterations=2))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or w > 900 or h > 500:
            continue
        out.append((x, y, w, h))
    return out


def region_color(frame: np.ndarray, box) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    x0, y0, x1, y1 = box
    patch = frame[max(0, y0) : y1, max(0, x0) : x1]
    if patch.size == 0:
        return (0, 0, 0), (0, 0, 0)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sel = hsv[:, :, 2] >= int(np.percentile(hsv[:, :, 2], 80))
    if sel.sum() < 4:
        sel = np.ones(hsv.shape[:2], bool)
    b, g, r = patch[sel].mean(axis=0)
    h, s, v = hsv[sel].mean(axis=0)
    return (int(r), int(g), int(b)), (int(h), int(s), int(v))


def color_class(hsv) -> str:
    h, s, v = hsv
    if s < 60:
        return "white"
    if h <= 35:
        return "warm"      # 橘/黄 —— 伤害
    if 36 <= h <= 85:
        return "green"     # 绿 —— 治疗（待真实素材确认）
    if 86 <= h <= 140:
        return "blue"      # 蓝/青 —— 护盾/法力
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--roi", default=None)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--min-digit-h", type=int, default=45)
    ap.add_argument("--max-digit-h", type=int, default=140)
    ap.add_argument("--pad", type=int, default=30)
    ap.add_argument("--out", default="out/region")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    eng = get_engine()

    is_img = Path(args.src).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    frames = []
    if is_img:
        img = cv2.imread(args.src)
        frames = [(0, 0.0, img)]
    else:
        cap = cv2.VideoCapture(args.src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / args.fps)))
        idx = 0
        while len(frames) < args.max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append((idx, idx / fps, fr))
            idx += 1
        cap.release()

    rows = []
    for fidx, t, raw in frames:
        if args.rotate == 90:
            frame = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(raw, cv2.ROTATE_180)
        elif args.rotate == 270:
            frame = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            frame = raw
        if args.roi:
            x, y, w, h = [int(v) for v in args.roi.replace(" ", "").split(",")]
            frame = frame[y : y + h, x : x + w]

        regions = activity_regions(frame)
        seen: list[tuple[int, int, int, int]] = []
        for (rx, ry, rw, rh) in regions:
            x0 = max(0, rx - args.pad)
            y0 = max(0, ry - args.pad)
            x1 = min(frame.shape[1], rx + rw + args.pad)
            y1 = min(frame.shape[0], ry + rh + args.pad)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            res = eng(crop)
            if res is None or res.boxes is None:
                continue
            for box, text, score in zip(res.boxes, res.txts, res.scores):
                pts = np.asarray(box)
                bx0, by0 = int(pts[:, 0].min()) + x0, int(pts[:, 1].min()) + y0
                bx1, by1 = int(pts[:, 0].max()) + x0, int(pts[:, 1].max()) + y0
                bh = by1 - by0
                if not text.isdigit() or not (args.min_digit_h <= bh <= args.max_digit_h):
                    continue
                if any(abs(bx0 - s[0]) < 12 and abs(by0 - s[1]) < 12 for s in seen):
                    continue  # 相邻区域重复命中
                seen.append((bx0, by0, bx1 - bx0, bh))
                rgb, hsv = region_color(frame, (bx0, by0, bx1, by1))
                rows.append(
                    {
                        "frame": fidx,
                        "t": round(t, 2),
                        "value": int(text),
                        "conf": round(float(score), 3),
                        "box": [bx0, by0, bx1, by1],
                        "h": bh,
                        "rgb": rgb,
                        "hsv": hsv,
                        "color": color_class(hsv),
                    }
                )

    rows.sort(key=lambda r: (r["t"], r["box"][1], r["box"][0]))
    if rows:
        with open(out_dir / "numbers.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "value", "conf", "color", "h", "x", "y", "w", "h_box", "rgb", "hsv"])
            for r in rows:
                b = r["box"]
                w.writerow([r["t"], r["value"], r["conf"], r["color"], r["h"], b[0], b[1], b[2] - b[0], b[3] - b[1], r["rgb"], r["hsv"]])

    print(f"\n共 {len(rows)} 个大数字命中")
    bycolor: dict[str, int] = {}
    for r in rows:
        bycolor[r["color"]] = bycolor.get(r["color"], 0) + 1
    print(f"  按颜色: {bycolor}")
    print(f"  样例: {[(r['t'], r['value'], r['color'], r['conf']) for r in rows[:20]]}")
    print(f"\n  CSV → {out_dir}/numbers.csv")
    Path(out_dir / "numbers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
