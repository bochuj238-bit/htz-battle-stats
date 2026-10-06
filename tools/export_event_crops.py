#!/usr/bin/env python3
"""把事件列表导出成「带编号的裁图证据表」，交给视觉模型逐格核对。

用法:
  python tools/export_event_crops.py "<视频>" --rotate 270 --events out/det_bd/events.csv --out out/evidence_bd
  # 只导绿色（治疗）事件：
  python tools/export_event_crops.py "<视频>" --rotate 270 --events out/det_bd/events.csv --family green

输出:
  out/evidence_*/sheet_00.png ...   每张 100 格（10x10），左上角是格子编号
  out/evidence_*/index.csv          编号 → 事件明细，方便和视觉模型的结论对齐
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_candidates import apply_rotate  # noqa: E402

CELL = 96
COLS = 10
PER_SHEET = 100


def load_events(path: str, family: str | None) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        fam = r.get("family") or r.get("color_family") or "?"
        if family and fam != family:
            continue
        out.append(
            {
                "t": float(r.get("t_start") or r.get("time_s") or 0),
                "value": int(r["value"]),
                "conf": float(r["conf"]),
                "family": fam,
                "x": int(float(r["x"])),
                "y": int(float(r["y"])),
                "alt": r.get("alt_readings", ""),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--events", required=True)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--roi", default=None)
    ap.add_argument("--family", default=None, help="只要某个颜色族（warm/green/...）")
    ap.add_argument("--cell", type=int, default=CELL)
    ap.add_argument("--per-sheet", type=int, default=PER_SHEET)
    ap.add_argument("--out", default="out/evidence")
    args = ap.parse_args()

    events = load_events(args.events, args.family)
    if not events:
        raise SystemExit("没有事件")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"打不开: {args.src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    cache: dict[int, np.ndarray] = {}
    cells: list[tuple[int, np.ndarray, dict]] = []
    for i, e in enumerate(events):
        fi = int(e["t"] * fps)
        if fi not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            fr = apply_rotate(fr, args.rotate)
            if args.roi:
                x, y, w, h = [int(v) for v in args.roi.replace(" ", "").split(",")]
                fr = fr[y : y + h, x : x + w]
            cache = {fi: fr}
        fr = cache[fi]
        c = args.cell // 2
        x0, y0 = max(0, e["x"] - c), max(0, e["y"] - c)
        crop = fr[y0 : y0 + args.cell, x0 : x0 + args.cell]
        if crop.size == 0:
            continue
        canvas = np.full((args.cell, args.cell, 3), 30, np.uint8)
        ch, cw = crop.shape[:2]
        canvas[:ch, :cw] = crop
        cells.append((i, canvas, e))
    cap.release()

    idx_rows = []
    for s in range(0, len(cells), args.per_sheet):
        chunk = cells[s : s + args.per_sheet]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = np.full((rows * (args.cell + 22), COLS * (args.cell + 8), 3), 20, np.uint8)
        for k, (i, canvas, e) in enumerate(chunk):
            r, col = divmod(k, COLS)
            y = r * (args.cell + 22) + 18
            x = col * (args.cell + 8) + 4
            sheet[y : y + args.cell, x : x + args.cell] = canvas
            cv2.putText(sheet, f"#{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        p = out / f"sheet_{s // args.per_sheet:02d}.png"
        cv2.imwrite(str(p), sheet)
        print(f"  {p}  格子 #{s} ~ #{min(s + args.per_sheet, len(cells)) - 1}")
        for k, (i, _, e) in enumerate(chunk):
            idx_rows.append([i, e["t"], e["value"], e["family"], e["conf"], e["x"], e["y"], e["alt"]])

    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "t", "ocr_value", "family", "conf", "x", "y", "alt_readings"])
        w.writerows(idx_rows)
    print(f"\n共 {len(cells)} 格 → {out}/index.csv")


if __name__ == "__main__":
    main()
