#!/usr/bin/env python3
"""对单帧/图片跑 OCR，把检测结果按「伤害数字 / 治疗数字 / 中文标签 / 其他」分类打印出来。

这是标定用的主力工具：先用它确认「纯数字+尺寸」过滤规则是否够用。

用法:
  python tools/ocr_frame.py /tmp/f44_native.png
  python tools/ocr_frame.py /tmp/f44_native.png --min-digit-h 40 --dump out/ocr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

ENGINE = None


def get_engine() -> RapidOCR:
    global ENGINE
    if ENGINE is None:
        ENGINE = RapidOCR()
    return ENGINE


def classify(text: str, box_h: float, min_digit_h: float) -> str:
    digits_only = text.isdigit()
    if digits_only and box_h >= min_digit_h:
        return "digit_big"
    if digits_only:
        return "digit_small"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "cjk"
    return "other"


def mean_color(frame_bgr: np.ndarray, pts) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    x0, y0, x1, y1 = pts
    patch = frame_bgr[int(y0) : int(y1), int(x0) : int(x1)]
    if patch.size == 0:
        return (0, 0, 0), (0, 0, 0)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sel = hsv[:, :, 2] >= int(np.percentile(hsv[:, :, 2], 85))  # 只取字，不取底
    if sel.sum() < 4:
        sel = np.ones(hsv.shape[:2], bool)
    b, g, r = patch[sel].mean(axis=0)
    h, s, v = hsv[sel].mean(axis=0)
    return (int(r), int(g), int(b)), (int(h), int(s), int(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--min-digit-h", type=float, default=40.0, help="算『大数字』的最小框高")
    ap.add_argument("--dump", default=None, help="把大数字裁图存到这里")
    ap.add_argument("--json", default=None, help="结果写 JSON")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"读不到图: {args.image}")
    if args.rotate == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif args.rotate == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif args.rotate == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    res = get_engine()(img)
    if res is None or res.boxes is None:
        raise SystemExit("没检测到任何文字")

    rows = []
    for box, text, score in zip(res.boxes, res.txts, res.scores):
        pts = np.asarray(box)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        h = y1 - y0
        rgb, hsv = mean_color(img, (x0, y0, x1, y1))
        rows.append(
            {
                "text": text,
                "score": round(float(score), 3),
                "kind": classify(text, h, args.min_digit_h),
                "box": [int(x0), int(y0), int(x1), int(y1)],
                "h": int(h),
                "rgb": rgb,
                "hsv": hsv,
            }
        )

    order = {"digit_big": 0, "digit_small": 1, "cjk": 2, "other": 3}
    rows.sort(key=lambda r: (order[r["kind"]], -r["h"]))

    cur = None
    for r in rows:
        if r["kind"] != cur:
            cur = r["kind"]
            label = {
                "digit_big": "★ 大数字（伤害/治疗候选）",
                "digit_small": "小数字（出手顺序/徽章，要丢弃）",
                "cjk": "中文标签（单位名字 = 锚点来源）",
                "other": "其他",
            }[cur]
            print(f"\n--- {label} ---")
        print(f"  {r['text']:>8s}  conf={r['score']:.2f}  h={r['h']:3d}  rgb={r['rgb']}  hsv={r['hsv']}  box={r['box']}")

    big = [r for r in rows if r["kind"] == "digit_big"]
    print(f"\n合计 {len(rows)} 个文本框；其中大数字 {len(big)} 个")

    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(big):
            x0, y0, x1, y1 = r["box"]
            pad = 8
            crop = img[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
            cv2.imwrite(str(out / f"big_{i:02d}_{r['text']}.png"), crop)
        print(f"  大数字裁图 → {out}/")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON → {args.json}")


if __name__ == "__main__":
    main()
