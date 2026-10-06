#!/usr/bin/env python3
"""单位归属：把每个伤害/治疗数字，挂到具体单位头上。

做法：
  1. 每帧用 RapidOCR 认出画面里的中文单位名（本游戏实测置信度 0.95~1.0）→ 得到"锚点"
  2. 数字事件按「到锚点的距离」归属，超过 --max-dist 的丢弃
  3. 输出「每个单位 × 每个数字」的明细 + 汇总

用法:
  # 先跑 detect + aggregate 得到事件表，再跑本工具
  python tools/assign_units.py "<视频>" --rotate 270 --events out/det/events.csv --at 34.8
  python tools/assign_units.py "<视频>" --rotate 270 --events out/det/events.csv --fps 1
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_candidates import apply_rotate  # noqa: E402

ENGINE: RapidOCR | None = None

# 画面上的固定 UI 文案，不是单位名
UI_BLACKLIST = {
    "第1回合", "第2回合", "第3回合", "回合", "对手", "战斗", "自动", "视角",
    "集火", "防御", "攻击", "逃跑", "物品", "技能", "阵法", "切磋", "取消",
}
# 排除区域（比例坐标 x0,y0,x1,y1）—— 回合号 / 按钮区
UI_ZONES = [(0.0, 0.0, 0.14, 0.16), (0.86, 0.0, 1.0, 0.20), (0.0, 0.86, 1.0, 1.0)]


def get_engine() -> RapidOCR:
    global ENGINE
    if ENGINE is None:
        ENGINE = RapidOCR()
    return ENGINE


def in_ui_zone(x: float, y: float, w: int, h: int) -> bool:
    for x0, y0, x1, y1 in UI_ZONES:
        if x0 * w <= x <= x1 * w and y0 * h <= y <= y1 * h:
            return True
    return False


def find_anchors(frame: np.ndarray, min_h: int = 34, max_h: int = 80) -> list[dict]:
    """OCR 出单位名字 → 锚点。"""
    h, w = frame.shape[:2]
    res = get_engine()(frame)
    anchors = []
    if res is None or res.boxes is None:
        return anchors
    for box, text, score in zip(res.boxes, res.txts, res.scores):
        txt = text.strip().strip("·.,。°v")
        if not txt or txt in UI_BLACKLIST:
            continue
        if not any("\u4e00" <= ch <= "\u9fff" for ch in txt):
            continue  # 只要中文名字
        if len(txt) < 2:
            continue
        pts = np.asarray(box)
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        bh = y1 - y0
        if not (min_h <= bh <= max_h):
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if in_ui_zone(cx, cy, w, h):
            continue
        anchors.append(
            {"name": txt, "conf": round(float(score), 3), "cx": cx, "cy": cy,
             "w": x1 - x0, "h": bh}
        )
    return anchors


def dedup_anchors(anchors: list[dict], radius: float = 80.0) -> list[dict]:
    out: list[dict] = []
    for a in sorted(anchors, key=lambda a: -a["conf"]):
        if all((abs(a["cx"] - b["cx"]) > radius or abs(a["cy"] - b["cy"]) > radius) for b in out):
            out.append(a)
    return out


def load_events(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        # 兼容 aggregate.py 的 events.csv（family / t_start）和 detect_numbers.py 的 numbers.csv（color_family / time_s）
        out.append(
            {
                "t": float(r.get("t_start") or r.get("time_s") or 0),
                "value": int(r["value"]),
                "conf": float(r["conf"]),
                "family": r.get("family") or r.get("color_family") or "?",
                "x": float(r["x"]),
                "y": float(r["y"]),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="视频或图片（用来找锚点）")
    ap.add_argument("--events", required=True, help="aggregate.py 产出的 events.csv")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--roi", default=None)
    ap.add_argument("--at", type=float, default=None, help="只用这一时刻的帧找锚点")
    ap.add_argument("--fps", type=float, default=0.5, help="多点找锚点时的采样率")
    ap.add_argument("--max-dist", type=float, default=260.0, help="数字到锚点的最大归属距离(px)")
    ap.add_argument("--radius", type=float, default=80.0, help="锚点去重半径")
    ap.add_argument("--out", default="out/assign")
    args = ap.parse_args()

    # ---- 取锚点（可以在多个时刻取，合并） ----
    anchors: list[dict] = []
    if Path(args.src).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        frame = cv2.imread(args.src)
        if args.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        anchors = find_anchors(frame)
    else:
        cap = cv2.VideoCapture(args.src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        times = [args.at] if args.at is not None else list(np.arange(0, 200, 1 / max(args.fps, 0.01)))
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                break
            fr = apply_rotate(fr, args.rotate)
            anchors.extend(find_anchors(fr))
        cap.release()
    anchors = dedup_anchors(anchors, args.radius)

    print(f"找到单位锚点 {len(anchors)} 个:")
    for a in sorted(anchors, key=lambda a: (a["cy"], a["cx"])):
        print(f"   {a['name']:>10s}  conf={a['conf']:.2f}  位置=({a['cx']:.0f},{a['cy']:.0f})")

    events = load_events(args.events)
    per_unit: dict[str, list[dict]] = defaultdict(list)
    orphan = []
    for e in events:
        if not anchors:
            break
        a = min(anchors, key=lambda a: (a["cx"] - e["x"]) ** 2 + (a["cy"] - e["y"]) ** 2)
        d = ((a["cx"] - e["x"]) ** 2 + (a["cy"] - e["y"]) ** 2) ** 0.5
        if d > args.max_dist:
            orphan.append((e, d))
            continue
        per_unit[a["name"]].append({**e, "dist": round(d, 1)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_unit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "t", "value", "family", "conf", "dist", "x", "y"])
        for name, items in per_unit.items():
            for it in sorted(items, key=lambda i: i["t"]):
                w.writerow([name, it["t"], it["value"], it["family"], it["conf"], it["dist"], int(it["x"]), int(it["y"])])

    print(f"\n归属结果（共 {len(events)} 个数字事件，{len(orphan)} 个找不到单位）:")
    for name, items in sorted(per_unit.items(), key=lambda kv: -sum(i["value"] for i in kv[1])):
        total = sum(i["value"] for i in items)
        print(f"   {name:>10s}  数字 {len(items):3d} 个   合计 {total:>8d}   明细 {[(i['t'], i['value']) for i in items[:6]]}")
    print(f"\n   CSV → {out}/per_unit.csv")
    if orphan:
        print(f"   未归属样例: {[(round(e['t'],1), e['value']) for e, _ in orphan[:8]]}")


if __name__ == "__main__":
    main()
