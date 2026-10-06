#!/usr/bin/env python3
"""构建「数字/非数字」分类器训练集：批量导出候选裁图 + 生成标注拼图。

为什么要做：实测显示尺寸/时长/置信度都无法区分真伤害数字与 UI 数字
（真数字中位高 58px，误检中位 70px），只能靠外观形态（渐变描边字 vs 白字菱形底框）。
所以用视觉模型标注一批候选，训一个小分类器。

用法:
  python tools/build_dataset.py "video.mp4" --rotate 270 --numbers out/det_bd/numbers.csv \
      --sample 750 --tag bd --out out/ds

输出:
  out/ds/crops_<tag>.npz     所有裁图（64x64 RGB）+ id
  out/ds/sheet_<tag>_NN.png  标注用拼图（10x10，格子左上角 #id）
  out/ds/manifest_<tag>.csv  id → 来源明细
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_candidates import apply_rotate  # noqa: E402

SIZE = 64
PAD_RATIO = 0.35   # 裁图在 token 框基础上外扩的比例（保留底框等上下文）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--roi", default=None)
    ap.add_argument("--sample", type=int, default=750, help="随机抽样多少条（0=全部）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="out/ds")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.numbers, encoding="utf-8")))
    if args.sample and len(rows) > args.sample:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]
    rows.sort(key=lambda r: float(r["time_s"]))
    print(f"{args.tag}: 取 {len(rows)} 个候选")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    imgs, ids, manifest = [], [], []
    frame_cache: dict[int, np.ndarray] = {}
    for i, r in enumerate(rows):
        fi = int(float(r["time_s"]) * fps)
        if fi not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            fr = apply_rotate(fr, args.rotate)
            if args.roi:
                x, y, w, h = [int(v) for v in args.roi.replace(" ", "").split(",")]
                fr = fr[y : y + h, x : x + w]
            frame_cache = {fi: fr}
        fr = frame_cache[fi]

        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        px, py = int(w * PAD_RATIO) + 6, int(h * PAD_RATIO) + 6
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(fr.shape[1], x + w + px), min(fr.shape[0], y + h + py)
        crop = fr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        scale = SIZE / max(crop.shape[0], crop.shape[1])
        resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
        canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
        oy, ox = (SIZE - resized.shape[0]) // 2, (SIZE - resized.shape[1]) // 2
        canvas[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized

        gid = f"{args.tag}:{i}"
        imgs.append(canvas)
        ids.append(gid)
        manifest.append([gid, r["time_s"], r["value"], r["color_family"], r["conf"], x, y, w, h])
    cap.release()

    np.savez_compressed(out / f"crops_{args.tag}.npz", imgs=np.stack(imgs), ids=np.array(ids))
    with open(out / f"manifest_{args.tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "t", "value", "family", "conf", "x", "y", "w", "h"])
        w.writerows(manifest)

    # 标注拼图
    per_sheet = 100
    cols = 10
    for s in range(0, len(imgs), per_sheet):
        chunk = imgs[s : s + per_sheet]
        chunk_ids = ids[s : s + per_sheet]
        rows_n = (len(chunk) + cols - 1) // cols
        cell = SIZE + 20
        sheet = np.full((rows_n * cell, cols * (SIZE + 6), 3), 25, np.uint8)
        for k, (im, gid) in enumerate(zip(chunk, chunk_ids)):
            rr, cc = divmod(k, cols)
            yy, xx = rr * cell + 18, cc * (SIZE + 6) + 3
            sheet[yy : yy + SIZE, xx : xx + SIZE] = im
            cv2.putText(sheet, f"#{s + k}", (xx, yy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        p = out / f"sheet_{args.tag}_{s // per_sheet:02d}.png"
        cv2.imwrite(str(p), sheet)
        print(f"  {p}  #{s}~#{min(s + per_sheet, len(imgs)) - 1}")
    print(f"\n  {out}/crops_{args.tag}.npz   {len(imgs)} 张 {SIZE}x{SIZE}")


if __name__ == "__main__":
    main()
