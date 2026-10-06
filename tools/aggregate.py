#!/usr/bin/env python3
"""把 detect_numbers.py 的逐帧命中，聚合成「真实的伤害/治疗事件」。

两道工序（这是全项目最容易翻车的地方）：
  1. 静态过滤：某个 (位置, 数值) 在大量帧里反复出现 → 是 UI 元素，不是伤害数字。
  2. 跨帧去重投票：同一位置 1.5s 内的多次命中 = 同一个数字的多帧残留
     或同一个数字的多段伤害，用「出现次数最多的读数」当最终值。

用法:
  python tools/aggregate.py out/det44/numbers.csv --out out/det44/events
  python tools/aggregate.py out/det44/numbers.csv --window 1.5 --radius 60 --static-pct 5
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(
                {
                    "t": float(r["time_s"]),
                    "value": int(r["value"]),
                    "conf": float(r["conf"]),
                    "family": r["color_family"],
                    "x": int(r["x"]),
                    "y": int(r["y"]),
                    "w": int(r["w"]),
                    "h": int(r["h"]),
                }
            )
    out.sort(key=lambda r: (r["t"], r["y"], r["x"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--window", type=float, default=1.5, help="同位置多少秒内视为同一个数字")
    ap.add_argument("--radius", type=float, default=60.0, help="位置聚类半径(px)")
    ap.add_argument("--static-pct", type=float, default=5.0,
                    help="某个(位置,数值)出现超过总帧数此百分比 → 判为静态UI并丢弃")
    ap.add_argument("--total-frames", type=int, default=0, help="总采样帧数（不填则用命中帧数估算）")
    ap.add_argument("--out", default="out/events")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        raise SystemExit("没有命中数据")

    total_frames = args.total_frames or (max(r["t"] for r in rows) * 10) or 1
    gx, gy = 60, 60
    pos_val = Counter((int(r["x"]) // gx, int(r["y"]) // gy, r["value"]) for r in rows)
    pos_only = Counter((int(r["x"]) // gx, int(r["y"]) // gy) for r in rows)
    thr = total_frames * args.static_pct / 100.0

    static_keys = {k for k, c in pos_val.items() if c > thr}
    static_spots = {k for k, c in pos_only.items() if c > thr * 2}

    kept, dropped = [], []
    for r in rows:
        key = (int(r["x"]) // gx, int(r["y"]) // gy, r["value"])
        spot = (int(r["x"]) // gx, int(r["y"]) // gy)
        if key in static_keys or spot in static_spots:
            dropped.append(r)
        else:
            kept.append(r)

    # 跨帧聚类
    clusters: list[dict] = []
    for r in kept:
        best = None
        for c in clusters:
            cx = sum(c["xs"]) / len(c["xs"])
            cy = sum(c["ys"]) / len(c["ys"])
            if abs(r["t"] - c["t_end"]) <= args.window and abs(r["x"] - cx) <= args.radius and abs(r["y"] - cy) <= args.radius:
                best = c
                break
        if best is None:
            clusters.append(
                {
                    "t_start": r["t"], "t_end": r["t"], "votes": Counter({r["value"]: 1}),
                    "xs": [r["x"]], "ys": [r["y"]], "confs": {r["value"]: [r["conf"]]},
                    "cs": Counter([r["family"]]),
                }
            )
        else:
            best["t_end"] = r["t"]
            best["votes"][r["value"]] += 1
            best["xs"].append(r["x"])
            best["ys"].append(r["y"])
            best["confs"].setdefault(r["value"], []).append(r["conf"])
            best["cs"][r["family"]] += 1

    events = []
    for c in clusters:
        value, n = c["votes"].most_common(1)[0]
        confs = c["confs"].get(value, [0.0])
        events.append(
            {
                "t_start": round(c["t_start"], 2),
                "t_end": round(c["t_end"], 2),
                "value": value,
                "votes": n,
                "n_candidates": sum(c["votes"].values()),
                "alt": {str(k): v for k, v in c["votes"].items() if k != value},
                "conf": round(max(confs), 3),
                "family": c["cs"].most_common(1)[0][0],
                "x": int(sum(c["xs"]) / len(c["xs"])),
                "y": int(sum(c["ys"]) / len(c["ys"])),
            }
        )
    events.sort(key=lambda e: (e["t_start"], e["y"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_start", "t_end", "value", "votes", "n_candidates", "alt_readings", "conf", "family", "x", "y"])
        for e in events:
            w.writerow([e["t_start"], e["t_end"], e["value"], e["votes"], e["n_candidates"],
                        json.dumps(e["alt"], ensure_ascii=False), e["conf"], e["family"], e["x"], e["y"]])
    Path(f"{out}.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    multi = [e for e in events if e["n_candidates"] > 1]
    unstable = [e for e in multi if e["alt"]]
    print(f"逐帧命中 {len(rows)} → 丢弃静态UI {len(dropped)} → 真实数字事件 {len(events)}")
    print(f"  其中多帧命中(需去重): {len(multi)}  读数不稳定(有异读): {len(unstable)}")
    print("\n  事件列表:")
    for e in events:
        alt = f"  (异读 {e['alt']})" if e["alt"] else ""
        print(f"    t={e['t_start']:6.2f}~{e['t_end']:5.2f}s  {e['value']:>6d}  votes={e['votes']}  conf={e['conf']}  ({e['x']},{e['y']}){alt}")
    print(f"\n  CSV  → {out}.csv")
    print(f"  JSON → {out}.json")


if __name__ == "__main__":
    main()
