#!/usr/bin/env python3
"""血条探测与测量。

血条的价值：它**天然绑定单位**——长在单位自己头顶，不需要「归属」这个难题。
浮动数字卡在归属上，血条不卡。所以用血条给「每单位每回合掉了多少血」，
再用浮动数字补明细。

本工具先用纯几何+色彩统计**自动找出血条**（不靠肉眼判读）：
  血条 = 宽扁的饱和色块（宽高比 >= 3、高度 3~20px）

  python tools/hp_bars.py "video.mp4" --rotate 270 --probe --at 54,68.7,126.8
  python tools/hp_bars.py "video.mp4" --rotate 270 --track --out out/bars_bd
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_candidates import apply_rotate  # noqa: E402


def bar_candidates(frame: np.ndarray, min_w: int = 30, max_h: int = 22, min_ratio: float = 3.0):
    """找出画面里所有「宽扁的饱和色块」= 血条/法力条候选。"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = ((hsv[:, :, 1] > 85) & (hsv[:, :, 2] > 80)).astype(np.uint8) * 255
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sat, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < min_w or not (3 <= bh <= max_h) or bw / max(bh, 1) < min_ratio:
            continue
        if area < 0.5 * bw * bh:  # 太稀疏，不是实心条
            continue
        m = lab == i
        hh = int(np.median(hsv[:, :, 0][m]))
        ss = int(np.median(hsv[:, :, 1][m]))
        vv = int(np.median(hsv[:, :, 2][m]))
        out.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "H": hh, "S": ss, "V": vv,
                    "x_pct": round(100 * x / w, 1), "y_pct": round(100 * y / h, 1)})
    return out


def region_of(y_pct: float, x_pct: float) -> str:
    if y_pct < 14:
        return "顶部UI"
    if x_pct < 16 and y_pct < 14:
        return "左上"
    return "战场"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--at", default="", help="逗号分隔时刻(秒)；--probe 用")
    ap.add_argument("--probe", action="store_true", help="打印血条候选，不写入")
    ap.add_argument("--track", action="store_true", help="逐帧测量并跟踪血条")
    ap.add_argument("--fps", type=float, default=2.0, help="--track 的采样帧率")
    ap.add_argument("--out", default="out/bars")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"打不开 {args.src}")

    if args.probe:
        for t in [float(x) for x in args.at.split(",") if x.strip()]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok:
                continue
            frame = apply_rotate(fr, args.rotate)
            bars = bar_candidates(frame)
            print(f"\n=== t={t}s  {frame.shape[1]}x{frame.shape[0]}  候选条 {len(bars)} ===")
            for b in sorted(bars, key=lambda z: (z["y"], z["x"]))[:30]:
                print(f"  y={b['y']:4d}({b['y_pct']:5.1f}%) x={b['x']:4d}({b['x_pct']:5.1f}%) "
                      f"w={b['w']:3d} h={b['h']:2d}  H={b['H']:3d} S={b['S']:3d} V={b['V']:3d}  [{region_of(b['y_pct'], b['x_pct'])}]")
        cap.release()
        return

    if not args.track:
        raise SystemExit("请指定 --probe 或 --track")

    # 逐帧测量：把各帧候选按位置聚类成"条槽位"，同一槽位长度随时间变化 = 血量变化
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps_src / args.fps)))
    fi = 0
    samples: list[tuple[float, list[dict]]] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fi % step == 0:
            frame = apply_rotate(fr, args.rotate)
            samples.append((fi / fps_src, bar_candidates(frame)))
        fi += 1
    cap.release()

    # 聚类成槽位
    slots: list[dict] = []
    for t, bars in samples:
        for b in bars:
            best, bd = None, 1e9
            for s in slots:
                d = abs(s["x"] - b["x"]) + abs(s["y"] - b["y"]) * 1.5
                if d < bd:
                    best, bd = s, d
            if best is not None and bd <= 25:
                best["x"] = 0.7 * best["x"] + 0.3 * b["x"]
                best["y"] = 0.7 * best["y"] + 0.3 * b["y"]
                best["series"].append((t, b["w"], b["H"], b["S"]))
            else:
                slots.append({"x": b["x"], "y": b["y"], "H": b["H"], "S": b["S"],
                              "series": [(t, b["w"], b["H"], b["S"])]})

    slots.sort(key=lambda s: (s["y"], s["x"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "slots.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "x", "y", "n_frames", "w_min", "w_max", "H", "S", "range_px"])
        for i, s in enumerate(slots):
            ws = [x[1] for x in s["series"]]
            w.writerow([i, round(s["x"]), round(s["y"]), len(s["series"]), min(ws), max(ws),
                        s["H"], s["S"], max(ws) - min(ws)])
            with open(out / f"slot_{i:02d}.csv", "w", newline="", encoding="utf-8") as f2:
                w2 = csv.writer(f2)
                w2.writerow(["t", "w", "H", "S"])
                for row in s["series"]:
                    w2.writerow([f"{row[0]:.2f}", row[1], row[2], row[3]])

    print(f"采样 {len(samples)} 帧，聚出 {len(slots)} 个条槽位 → {out}/slots.csv")
    print(f"\n{'#':>3} {'x':>5} {'y':>5} {'帧数':>5} {'宽度范围':>12} {'变化':>6}  {'H':>4} {'S':>4}")
    for i, s in enumerate(slots):
        if len(s["series"]) < 3:
            continue
        ws = [x[1] for x in s["series"]]
        print(f"{i:>3} {round(s['x']):>5} {round(s['y']):>5} {len(s['series']):>5} "
              f"{min(ws):>5}-{max(ws):<6} {max(ws)-min(ws):>6}  {s['H']:>4} {s['S']:>4}")


if __name__ == "__main__":
    main()
