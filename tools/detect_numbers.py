#!/usr/bin/env python3
"""伤害/治疗数字检测器（核心工具）。

原理（本游戏实测标定）：
  伤害数字 = 橘色描边 + 白色内芯，尺寸约 45~120px（原生 2868 宽的素材）。
  关键发现：按「颜色族」做掩码能干净地分离数字与光效/闪电（后者是粉紫/低饱和），
  再用连通域 + 同行合并得到候选框，最后用 RapidOCR「纯识别」模式读数字（约 7ms/框）。

用法:
  python tools/detect_numbers.py /tmp/f44_native.png
  python tools/detect_numbers.py inbox/battle.mp4 --fps 5 --rotate 270 --max-frames 200
  python tools/detect_numbers.py inbox/battle.mp4 --fps 5 --rotate 270 --out out/det

输出:
  out/det/numbers.csv     每个数字一行：时间/数值/颜色族/置信度/坐标
  out/det/annotated/      带框的抽查图
  out/det/contact.png     命中的数字裁图拼图（重点看这个判断准不准）
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_candidates import apply_rotate  # noqa: E402

ENGINE: RapidOCR | None = None

# 颜色族：阈值来自视觉审核的实测色值（幻唐志战斗画面）
#   伤害=橘黄(与怪名金黄同色，靠尺寸区分)  治疗=荧光绿(90,255,110)  等级前缀=白色(245,250,255)
COLOR_RANGES = {
    "warm":  ((0, 120, 180), (35, 255, 255)),    # 橘黄 —— 伤害数字
    "green": ((35, 110, 200), (80, 255, 255)),   # 荧光绿 —— 治疗数字
    "blue":  ((86, 100, 170), (140, 255, 255)),  # 蓝青 —— 护盾/法力（待确认）
    "red":   ((141, 120, 150), (179, 255, 255)), # 品红/红（待确认）
}

# 固定 UI 区域（比例坐标 x0,y0,x1,y1）：直接排除，免疫聊天框/面板干扰
UI_ZONES = [
    (0.00, 0.00, 0.16, 0.13),   # 左上：时间/回合/对手
    (0.60, 0.00, 1.00, 0.14),   # 右上：头像+气血/魔法数值
    (0.00, 0.86, 1.00, 1.00),   # 底部：按钮栏
    (0.00, 0.72, 0.36, 1.00),   # 左下：聊天框（平涂绿字，与单位名同色但是无描边）
    (0.64, 0.79, 0.74, 0.88),   # 实测：某固定 UI 元素（本地 OCR 反复读成 101），x 永远钉在 69.9%
]


def get_engine() -> RapidOCR:
    global ENGINE
    if ENGINE is None:
        ENGINE = RapidOCR()
    return ENGINE


def color_masks(frame: np.ndarray, families: list[str]) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = {}
    for fam in families:
        lo, hi = COLOR_RANGES[fam]
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        # 断开与背景光效的细连接
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out[fam] = m
    return out


def merge_row(boxes: list[tuple[int, int, int, int]], gap_ratio=1.1, ov_ratio=0.55):
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
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


def find_tokens(mask: np.ndarray, min_h: int, max_h: int, min_fill: float = 0.16):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    glyphs = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (min_h <= h <= max_h):
            continue
        if w > 6 * h or w < 3:
            continue
        if area / float(w * h) < min_fill:
            continue
        glyphs.append((x, y, w, h))
    return glyphs


def core_ratio(frame: np.ndarray, box, pad: int = 0) -> float:
    """数字框内「白芯」像素占比。伤害数字=橘描边+白芯；金币框/纯色 UI 几乎没有白芯。"""
    x, y, w, h = box
    patch = frame[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    core = ((hsv[:, :, 2] >= 240) & (hsv[:, :, 1] <= 60))
    return float(core.mean())


def white_frac(frame: np.ndarray, box) -> float:
    """框内「白色文字」占比。等级前缀/出手顺序编号都是白字，用来排除。"""
    x, y, w, h = box
    patch = frame[max(0, y) : y + h, max(0, x) : x + w]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    bright = hsv[:, :, 2] >= 200
    if bright.sum() < 6:
        return 0.0
    return float(((hsv[:, :, 1] <= 35) & bright).sum() / bright.sum())


def in_ui_zone(cx: float, cy: float, w: int, h: int) -> bool:
    return any(x0 * w <= cx <= x1 * w and y0 * h <= cy <= y1 * h for x0, y0, x1, y1 in UI_ZONES)


def read_crop(frame: np.ndarray, box, pad: int) -> tuple[str, float]:
    x, y, w, h = box
    crop = frame[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
    if crop.size == 0:
        return "", 0.0
    res = get_engine()(crop, use_det=False, use_cls=False, use_rec=True)
    if res is None or not res.txts:
        return "", 0.0
    return res.txts[0], float(res.scores[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--roi", default=None, help="x,y,w,h")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=100000)
    ap.add_argument("--families", default="warm,green,blue,red", help="要检测的颜色族")
    ap.add_argument("--min-h", type=int, default=0)
    ap.add_argument("--max-h", type=int, default=0)
    ap.add_argument("--h-min-pct", type=float, default=0.040,
                    help="数字高度下限，占画面高度比例（实测真浮动数字 4.5~6.5%%）")
    ap.add_argument("--h-max-pct", type=float, default=0.100, help="数字高度上限（占画面高度比例）")
    ap.add_argument("--max-white-frac", type=float, default=0.55,
                    help="框内白色字占比超过此值则拒（等级前缀/出手顺序都是白字）；设 1 关闭")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--core-ratio", type=float, default=0.0,
                    help="要求框内白芯占比≥此值（默认关闭：实测真数字白芯比也是0，此判据不可靠）")
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=25, help="每 N 帧存一张带框抽查图")
    ap.add_argument("--out", default="out/det")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    fams = [f.strip() for f in args.families.split(",") if f.strip()]

    # --- 取帧（图片或视频） ---
    is_img = Path(args.src).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    frames = []
    if is_img:
        frames = [(0, 0.0, cv2.imread(args.src))]
    else:
        cap = cv2.VideoCapture(args.src)
        if not cap.isOpened():
            raise SystemExit(f"打不开: {args.src}")
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
        print(f"取帧 {len(frames)} 张（源 {fps:.0f}fps → 采样 {args.fps}fps）")

    rows, crops = [], []
    for fi, (fidx, t, raw) in enumerate(frames):
        frame = apply_rotate(raw, args.rotate)
        if args.roi:
            x, y, w, h = [int(v) for v in args.roi.replace(" ", "").split(",")]
            frame = frame[y : y + h, x : x + w]

        annotated = frame.copy()
        fh = frame.shape[0]
        min_h = args.min_h or int(args.h_min_pct * fh)
        max_h = args.max_h or int(args.h_max_pct * fh)
        for fam, mask in color_masks(frame, fams).items():
            glyphs = find_tokens(mask, min_h, max_h)
            for box in merge_row(glyphs):
                bx, by, bw, bh = box
                if bw < 12 or bh < min_h:
                    continue
                if in_ui_zone(bx + bw / 2, by + bh / 2, frame.shape[1], fh):
                    continue
                if args.max_white_frac < 1 and white_frac(frame, box) > args.max_white_frac:
                    continue
                if args.core_ratio > 0 and core_ratio(frame, box) < args.core_ratio:
                    continue
                text, conf = read_crop(frame, box, args.pad)
                if not text.isdigit() or conf < args.min_conf:
                    continue
                rows.append(
                    {
                        "frame": fidx, "t": round(t, 2), "value": int(text), "conf": round(conf, 3),
                        "family": fam, "x": bx, "y": by, "w": bw, "h": bh,
                    }
                )
                crops.append(frame[max(0, by - 8) : by + bh + 8, max(0, bx - 8) : bx + bw + 8])
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
                cv2.putText(annotated, f"{text}", (bx, max(24, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

        if args.save_every and fi % args.save_every == 0:
            cv2.imwrite(str(out_dir / "annotated" / f"f{fidx:06d}.png"), annotated)

    with open(out_dir / "numbers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "value", "conf", "color_family", "x", "y", "w", "h"])
        for r in rows:
            w.writerow([r["frame"], r["t"], r["value"], r["conf"], r["family"], r["x"], r["y"], r["w"], r["h"]])

    if crops:
        cells = []
        for c in crops[:400]:
            h, w = c.shape[:2]
            s = 56 / max(1, h)
            cells.append(cv2.resize(c, (max(1, int(w * s)), 56), interpolation=cv2.INTER_NEAREST))
        cols = 12
        cw = max(c.shape[1] for c in cells) + 8
        rn = (len(cells) + cols - 1) // cols
        sheet = np.full((rn * 64, cols * cw, 3), 40, np.uint8)
        for i, c in enumerate(cells):
            r, col = divmod(i, cols)
            sheet[r * 64 + 4 : r * 64 + 60, col * cw + 4 : col * cw + 4 + c.shape[1]] = c
        cv2.imwrite(str(out_dir / "contact.png"), sheet)

    per_frame = len(rows) / max(1, len(frames))
    byfam: dict[str, int] = {}
    for r in rows:
        byfam[r["family"]] = byfam.get(r["family"], 0) + 1
    print(f"\n命中大数字 {len(rows)} 个 / {len(frames)} 帧 ≈ {per_frame:.2f} 个/帧")
    print(f"  按颜色族: {byfam}")
    vals = [(r["t"], r["value"], r["conf"], r["family"]) for r in rows[:15]]
    print(f"  前 15 个: {vals}")
    print(f"\n  CSV     → {out_dir}/numbers.csv")
    print(f"  拼图    → {out_dir}/contact.png")
    print(f"  抽查图  → {out_dir}/annotated/")


if __name__ == "__main__":
    sys.exit(main())
