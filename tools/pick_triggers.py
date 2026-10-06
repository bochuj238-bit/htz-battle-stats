#!/usr/bin/env python3
"""触发帧采集器：从一场战斗里挑出「值得送给视觉模型读」的那几帧。

为什么需要它：浮动数字只存活约 0.7 秒，随机抓帧大概率抓空；而每帧都送云端
又太贵。所以用**本地检测器当触发器**（召回实测 ~100%，免费，87ms/帧），
只在画面里出现大数字时才抓帧，一分钟的战斗通常只需要送几帧。

输出:
  out/trig/<tag>/trig_000_t0016.20.png   挑出来的帧
  out/trig/<tag>/triggers.csv            每帧的时间/命中数/候选读数（喂给视觉模型前的线索）

用法:
  python tools/pick_triggers.py "video.mp4" --rotate 270 --numbers out/det2_bd/numbers.csv --out out/trig/bd
  # 没跑过检测也行，直接给视频（会现跑一遍）
  python tools/pick_triggers.py "video.mp4" --rotate 0 --out out/trig/ce --limit 8
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_candidates import apply_rotate  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent


def run_detect(video: str, rotate: int, cache: Path) -> Path:
    """没现成 numbers.csv 就现跑一遍检测（结果缓存下来）。"""
    if cache.exists():
        return cache
    tmp = cache.parent / "_trig_det"
    print("  没有现成检测结果，先跑一遍 detect_numbers …")
    subprocess.run(
        [str(PROJ / ".venv/bin/python"), str(HERE / "detect_numbers.py"), video,
         "--rotate", str(rotate), "--fps", "5", "--families", "warm,green", "--out", str(tmp)],
        check=True, cwd=PROJ,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes((tmp / "numbers.csv").read_bytes())
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--numbers", default="", help="已有的 numbers.csv；不填则现跑检测")
    ap.add_argument("--ttl", type=float, default=3.0,
                    help="同一个 (数值, 横向位置) 在这种秒数内只触发一次（压掉同一次飘字的连续多帧和长驻静态误检）")
    ap.add_argument("--drop-values", default="140,145",
                    help="这些数值视为固定 UI（等级前缀），不单独触发；传空串关闭")
    ap.add_argument("--per-trigger", type=int, default=1,
                    help="每个触发点连导几帧（数字只活 0.6~0.8s 且末帧会淡出，取 2 可给召回兑底）")
    ap.add_argument("--limit", type=int, default=60, help="最多导出多少帧")
    ap.add_argument("--min-hits", type=int, default=1, help="至少命中几个数字才算触发")
    ap.add_argument("--scale", type=int, default=0, help="导出宽度，0=原分辨率")
    ap.add_argument("--out", default="out/trig")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.numbers) if args.numbers else out / "numbers.csv"
    csv_path = run_detect(args.video, args.rotate, cache) if not args.numbers else Path(args.numbers)

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit("检测结果为空")
    for r in rows:
        r["t"] = float(r["time_s"])

    # 新颖性触发（不再用时间窗聚类）。
    # 时间窗的毛病：一个窗口横跨多帧时，代表帧可能根本不含关键数字
    # （实测 696 @t=68.71、1245 @t=126.78 就是这样被漏掉的）。
    # 换成：只有出现「之前没见过的 (数值, 横向位置)」才值得送进视觉模型。
    # 同一次飘字持续的那几帧、以及长驻的静态 UI 误检，都只会触发一次。
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_frame[int(r["frame"])].append(r)

    seen: dict[tuple, float] = {}
    drop_values = {v.strip() for v in args.drop_values.split(",") if v.strip()}
    picked = []
    for fi in sorted(by_frame, key=lambda f: float(by_frame[f][0]["t"])):
        group = by_frame[fi]
        t = float(group[0]["t"])
        novel = [c for c in group if c["value"] not in drop_values
                 and seen.get((c["value"], round(float(c["x"]) / 60)), -1e9) < t - args.ttl]
        if not novel:
            continue
        for c in group:
            seen[(c["value"], round(float(c["x"]) / 60))] = t
        rep = max(novel, key=lambda c: float(c["h"]))
        score = len(novel) + 0.02 * sum(float(c["h"]) for c in novel)
        picked.append((score, rep, group))

    picked.sort(key=lambda s: float(s[1]["t"]))  # 按时间顺序导出

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不开 {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    manifest = []
    for i, (score, rep, group) in enumerate(picked):
        fi = int(rep["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        frame = apply_rotate(fr, args.rotate)
        if args.scale:
            k = args.scale / frame.shape[1]
            frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        name = f"trig_{i:03d}_t{float(rep['t']):08.2f}.png"
        cv2.imwrite(str(out / name), frame)
        vals = sorted({x["value"] for x in group})
        fams = sorted({x["color_family"] for x in group})
        manifest.append({
            "file": name, "idx": i, "t": round(float(rep["t"]), 2), "frame": fi,
            "n_hits": len(group), "families": "/".join(fams),
            "candidate_values": " ".join(vals), "score": round(score, 1),
        })
    cap.release()

    with (out / "triggers.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    print(f"\n从 {len(by_frame)} 个有候选的帧里，新颖性触发出 {len(manifest)} 个待读帧 → {out}/")
    print(f"\n{'#':>4} {'时刻':>8} {'命中':>5} {'族':>12}  候选读数")
    for m in manifest[:25]:
        print(f"{m['idx']:>4} {m['t']:>8.2f} {m['n_hits']:>5} {m['families']:>12}  {m['candidate_values'][:52]}")
    if len(manifest) > 25:
        print(f"  … 还有 {len(manifest) - 25} 帧")
    print(f"\n  → {out}/triggers.csv")


if __name__ == "__main__":
    main()
