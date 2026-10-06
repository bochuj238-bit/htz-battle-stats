#!/usr/bin/env python3
"""抓一帧屏幕存成图片 —— 设置抓取区域时用来自查。

为什么需要它：游戏窗口不是全屏时，工具只能抓屏幕的某一块（--region x,y,w,h）。
坐标填错了不会报错，只会「什么都识别不到」，白打一场。
所以先用这个抓一帧看看框对没框对。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_region(s: str):
    parts = [int(x) for x in s.replace("，", ",").replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("区域要 4 个数：x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError("宽和高必须是正数")
    return {"left": x, "top": y, "width": w, "height": h}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--region", default="", help="x,y,w,h；留空则抓整个屏幕")
    ap.add_argument("--out", default="out/preview.png")
    args = ap.parse_args()

    import mss  # type: ignore

    factory = getattr(mss, "MSS", None) or mss.mss
    try:
        with factory() as sct:
            if args.region:
                mon = parse_region(args.region)
            else:
                mons = sct.monitors
                if args.monitor >= len(mons):
                    print(f"  没有 {args.monitor} 号屏幕，本机只有 0~{len(mons) - 1}")
                    return 1
                mon = mons[args.monitor]
            img = np.array(sct.grab(mon))[:, :, :3]
    except Exception as e:  # noqa: BLE001
        print(f"  抓屏失败：{e}")
        return 1

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)

    h, w = img.shape[:2]
    mean = float(img.mean())
    std = float(img.std())
    print(f"  抓到 {w}x{h} 的画面（平均亮度 {mean:.0f}/255，变化度 {std:.0f}）")
    print(f"  已存到 {p.resolve()}")
    if mean < 3:
        print("  ⚠ 画面几乎全黑 —— 游戏可能是「独占全屏」，抓不到。")
        print("     请到游戏设置里改成「窗口化」或「无边框窗口」再试。")
    elif std < 2:
        print("  ⚠ 整个画面几乎是纯色 —— 同上，多半是独占全屏没抓到内容。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
