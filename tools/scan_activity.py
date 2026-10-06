#!/usr/bin/env python3
"""活跃度扫描：找出全片「数字最密集」的时刻，用于标定和抽查。

用法:
  python tools/scan_activity.py inbox/battle.mp4 --rotate 270 --fps 2

输出:
  out/scan/activity.csv     t,n_big,n_all,bright_ratio
  out/scan/top/fXXXXX.png   活跃度最高的若干帧（全分辨率，用来肉眼抽查）
  终端打印: 活跃度最高的时刻表
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_candidates import apply_roi, apply_rotate, bright_mask, find_glyphs, merge_row  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--roi", default=None)
    ap.add_argument("--v-min", type=int, default=225)
    ap.add_argument("--big-h", type=int, default=22, help="『大数字』的最小高度(px)")
    ap.add_argument("--top", type=int, default=12, help="导出多少张最活跃帧")
    ap.add_argument("--out", default="out/scan")
    args = ap.parse_args()

    roi = None
    if args.roi:
        x, y, w, h = [int(v) for v in args.roi.replace(" ", "").split(",")]
        roi = (x, y, w, h)

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"打不开: {args.src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / args.fps)))

    out_dir = Path(args.out)
    (out_dir / "top").mkdir(parents=True, exist_ok=True)

    rows, idx, kept = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            f = apply_roi(apply_rotate(frame, args.rotate), roi)
            mask = bright_mask(f, args.v_min)
            glyphs = find_glyphs(mask, 8, 90, 12, 4000)
            tokens = merge_row(glyphs)
            big = [t for t in tokens if t[3] >= args.big_h]
            rows.append(
                {
                    "t": round(idx / src_fps, 2),
                    "n_big": len(big),
                    "n_all": len(tokens),
                    "bright": round(float((mask > 0).mean()), 5),
                    "_frame": f,
                }
            )
            kept += 1
        idx += 1
    cap.release()

    if not rows:
        raise SystemExit("没取到帧")

    rows.sort(key=lambda r: r["t"])
    with open(out_dir / "activity.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "n_big", "n_all", "bright_ratio"])
        for r in rows:
            w.writerow([f"{r['t']:.2f}", r["n_big"], r["n_all"], r["bright"]])

    top = sorted(rows, key=lambda r: (-r["n_big"], -r["n_all"]))[: args.top]
    top.sort(key=lambda r: r["t"])
    for r in top:
        cv2.imwrite(str(out_dir / "top" / f"f{int(r['t'] * 100):06d}_n{r['n_big']:02d}.png"), r["_frame"])

    n_big = np.array([r["n_big"] for r in rows])
    print(f"扫描 {kept} 帧（{src_fps:.0f}fps 源 / 采样 {args.fps}fps）")
    print(f"大数字个数: 平均 {n_big.mean():.1f}  中位 {np.median(n_big):.0f}  峰值 {n_big.max()}")
    print("\n最活跃的时刻（数字最多）:")
    for r in top:
        print(f"  t={r['t']:6.1f}s  大数字 {r['n_big']:3d}  全部候选 {r['n_all']:3d}  亮像素 {r['bright'] * 100:5.2f}%")
    print(f"\n  曲线: {out_dir}/activity.csv")
    print(f"  抽样帧: {out_dir}/top/  ← 挑这几张看数字长什么样")


if __name__ == "__main__":
    sys.exit(main())
