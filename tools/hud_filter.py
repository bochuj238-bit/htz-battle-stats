#!/usr/bin/env python3
"""单位 HUD 过滤 + 归属（修正版）。

依据视觉审核的实测结论：
  1. 每个单位名字左边都有等级前缀（140/145），是最大误检源 → 用「名字行排除区」干掉。
  2. 归属不能取「最近的单位名字」：名字在**脚底**、伤害数字从**胸口飘到头顶**，
     2.5D 斜视角下前排队友的数字在屏幕上会压在后排名字上方 → 改用 **x 轴对齐** 归属。
  3. 单位名字 OCR 实测置信度 0.85~1.00，可靠。

流程:
  detect_numbers.py 的 numbers.csv
    → 本工具（干掉落在名字行排除区的候选 + 按 x 对齐归属单位）
    → aggregate.py（静态过滤 + 跨帧去重）
    → 按单位×回合汇总

用法:
  python tools/hud_filter.py "video.mp4" --rotate 270 --numbers out/det_bd/numbers.csv --out out/hud_bd
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_candidates import apply_rotate  # noqa: E402

ENGINE: RapidOCR | None = None

# 画面固定 UI 文案（不是单位名）
UI_BLACKLIST = {
    "回合", "对手", "战斗", "自动", "视角", "集火", "防御", "攻击", "逃跑",
    "物品", "技能", "阵法", "切磋", "取消", "好友", "帮", "设置", "排位", "战况",
    "奖励", "无克制", "克制", "帮派", "队伍", "等级", "经验", "气血", "魔法",
}
# 名字里出现的噪声关键词（聊天/系统文案）
BAD_KEYWORDS = ("队伍", "世界", "加入", "来人", "召唤", "出战", "等级", "请", "赛", "榜",
                "法宝", "商店", "商店", "任务", "报", "界面", "切换", "排序", "隐藏")
LEVEL_PREFIX = re.compile(r"^(飞升|等级[:：]?)?\d{1,3}")
# 固定 UI 区域（比例坐标 x0,y0,x1,y1）：回合号 / 头像面板 / 底部栏 / 聊天框
UI_ZONES = [
    (0.00, 0.00, 0.16, 0.13),   # 左上回合/对手
    (0.60, 0.00, 1.00, 0.14),   # 右上头像+气血面板
    (0.00, 0.86, 1.00, 1.00),   # 底部按钮栏
    (0.00, 0.60, 0.42, 0.90),   # 左下聊天框
]


def get_engine() -> RapidOCR:
    if ENGINE is None:
        globals()["ENGINE"] = RapidOCR()
    return ENGINE  # type: ignore[return-value]


def in_ui_zone(cx: float, cy: float, w: int, h: int) -> bool:
    return any(x0 * w <= cx <= x1 * w and y0 * h <= cy <= y1 * h for x0, y0, x1, y1 in UI_ZONES)


def is_green_text(frame: np.ndarray, box: tuple[float, float, float, float], min_frac: float = 0.30) -> bool:
    """单位名字是绿色字（实测）；聊天/系统文案不是。用笔画像素里绿色占比判定。"""
    x0, y0, x1, y1 = (int(v) for v in box)
    patch = frame[max(0, y0) : y1, max(0, x0) : x1]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    bright = hsv[:, :, 2] >= 130
    if bright.sum() < 8:
        return False
    h, s = hsv[:, :, 0][bright], hsv[:, :, 1][bright]
    green = (h >= 40) & (h <= 100) & (s >= 70)
    return float(green.mean()) >= min_frac


def clean_name(txt: str) -> str:
    t = txt.strip().strip("·.,。°v|^へ")
    t = LEVEL_PREFIX.sub("", t)          # 去掉等级前缀（飞升140 / 等级145 / 140）
    t = t.lstrip("·.。").strip()
    return t


def find_hud(frame: np.ndarray, min_h: int = 28, max_h: int = 90) -> list[dict]:
    """OCR 出所有单位名字 → 锚点 + 名字行排除区。"""
    h, w = frame.shape[:2]
    res = get_engine()(frame)
    out: list[dict] = []
    if res is None or res.boxes is None:
        return out
    for box, text, score in zip(res.boxes, res.txts, res.scores):
        raw = text.strip()
        if not raw or raw in UI_BLACKLIST or any(k in raw for k in BAD_KEYWORDS):
            continue
        if "[" in raw or "]" in raw or "（" in raw or "(" in raw:
            continue
        txt = clean_name(raw)
        if not txt or txt in UI_BLACKLIST or len(txt) < 2 or len(txt) > 8:
            continue
        cjk = sum(1 for ch in txt if "\u4e00" <= ch <= "\u9fff")
        if cjk < 2:
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
        if not is_green_text(frame, (x0, y0, x1, y1)):
            continue
        # 名字行排除区：向左扩 2.2 倍字高（覆盖等级前缀），上下各扩 0.5 倍字高
        out.append(
            {
                "name": txt,
                "conf": round(float(score), 3),
                "cx": cx,
                "cy": cy,
                "box": (x0, y0, x1, y1),
                "excl": (x0 - 2.2 * bh, y0 - 0.5 * bh, x1 + 0.3 * bh, y1 + 0.5 * bh),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--every", type=int, default=10, help="每 N 帧重算一次 HUD（单位不太动）")
    ap.add_argument("--x-tol", type=float, default=110.0, help="x 轴对齐容差(px)")
    ap.add_argument("--y-up", type=float, default=420.0, help="数字相对名字最多高多少(px)")
    ap.add_argument("--y-down", type=float, default=60.0, help="数字相对名字最多低多少(px)")
    ap.add_argument("--out", default="out/hud")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.numbers, encoding="utf-8")))
    rows.sort(key=lambda r: float(r["time_s"]))
    if not rows:
        raise SystemExit("没有候选")

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"打不开: {args.src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 按帧分组，逐帧（每 every 帧）算一次 HUD
    by_frame: dict[int, list[dict]] = {}
    for r in rows:
        by_frame.setdefault(int(r["frame"]), []).append(r)

    hud_of: dict[int, list[dict]] = {}
    frames_sorted = sorted(by_frame)
    for i, fi in enumerate(frames_sorted):
        if i % args.every:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        hud_of[fi] = find_hud(apply_rotate(fr, args.rotate))
    cap.release()

    hud_frames = sorted(hud_of)
    if not hud_frames:
        raise SystemExit("没算到 HUD")
    all_names = sorted({a["name"] for h in hud_of.values() for a in h})
    empty = sum(1 for h in hud_of.values() if not h)
    print(f"HUD 采样 {len(hud_frames)} 帧（空帧 {empty}），识别到单位名 {len(all_names)} 个: {all_names}")

    def nearest_hud(fi: int) -> list[dict]:
        f = min(hud_frames, key=lambda x: abs(x - fi))
        return hud_of[f]

    out_rows, dropped_name, dropped_attr = [], 0, 0
    for fi in frames_sorted:
        hud = nearest_hud(fi)
        for r in by_frame[fi]:
            x, y = float(r["x"]), float(r["y"])
            # 1) 落在名字行排除区 → 丢弃（等级前缀/名字笔画）
            if any(ex0 <= x <= ex1 and ey0 <= y <= ey1 for a in hud for ex0, ey0, ex1, ey1 in [a["excl"]]):
                dropped_name += 1
                continue
            # 2) 按 x 轴对齐归属（数字在单位胸口→头顶，位于名字上方）
            cands = [a for a in hud if abs(a["cx"] - x) <= args.x_tol and -args.y_down <= (a["cy"] - y) <= args.y_up]
            if not cands:
                dropped_attr += 1
                continue
            best = min(cands, key=lambda a: abs(a["cx"] - x))
            out_rows.append({**r, "unit": best["name"], "dx": round(best["cx"] - x, 1),
                             "dy": round(best["cy"] - y, 1)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "numbers_unit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "value", "conf", "color_family", "unit", "x", "y", "w", "h", "dx", "dy"])
        for r in out_rows:
            w.writerow([r["frame"], r["time_s"], r["value"], r["conf"], r["color_family"],
                        r["unit"], r["x"], r["y"], r["w"], r["h"], r["dx"], r["dy"]])

    print(f"\n候选 {len(rows)} → 名字行排除区丢弃 {dropped_name} → 无法归属丢弃 {dropped_attr} → 保留 {len(out_rows)}")
    import collections
    c = collections.Counter(r["unit"] for r in out_rows)
    print("  按单位分布（Top12）:", c.most_common(12))
    print(f"  CSV → {out}/numbers_unit.csv")


if __name__ == "__main__":
    main()
