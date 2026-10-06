#!/usr/bin/env python3
"""针对性排查某个误检：把画面裁出来放大，直接问视觉模型「这是什么」。

用法:
    VLM_API_KEY=... ./.venv/bin/python tools/probe_fp.py --video "x.mp4" --rotate 270 \\
        --at 42.2,82.2,101.9 --x-pct 35 --question "这个位置上的东西到底是什么？"
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_candidates import apply_rotate  # noqa: E402


def ask(base_url: str, model: str, key: str, b64: str, question: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "temperature": 0, "max_tokens": 700,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--at", required=True, help="逗号分隔时刻")
    ap.add_argument("--x-pct", type=float, default=-1, help="要放大的横坐标百分比；-1 = 整帧")
    ap.add_argument("--y-pct", type=float, default=-1)
    ap.add_argument("--zoom", type=float, default=2.5)
    ap.add_argument("--crop-w", type=int, default=700, help="裁图宽度(原始像素)")
    ap.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "qwen-vl-max"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    ap.add_argument("--question", default=(
        "这是回合制网游《幻唐志》的战斗画面。请**只看红框标出的区域**，简短回答：\n"
        "1) 那里的东西到底是什么？（是飘起来的伤害数字？还是 UI 里的东西？）\n"
        "2) 如果是 UI，它是哪一种：等级前缀 / 出手顺序编号 / 状态层数 / 头顶徽章 / 气血数值 / 其它？\n"
        "3) 如果是数字，值是多少？\n"
        "回答控制在 100 字以内。"))
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("缺 VLM_API_KEY")

    cap = cv2.VideoCapture(args.video)
    for t in [float(x) for x in args.at.split(",") if x.strip()]:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        frame = apply_rotate(fr, args.rotate)
        h, w = frame.shape[:2]
        if 0 < args.x_pct < 100:
            cx = int(args.x_pct / 100 * w)
            cy = int((args.y_pct / 100 * h) if 0 < args.y_pct < 100 else 0.45 * h)
            x0, x1 = max(0, cx - args.crop_w // 2), min(w, cx + args.crop_w // 2)
            y0, y1 = max(0, cy - 260), min(h, cy + 260)
            frame = frame[y0:y1, x0:x1]
            # 画个红框标出目标中心
            cv2.rectangle(frame, (args.crop_w // 2 - 90, (cy - y0) - 55), (args.crop_w // 2 + 90, (cy - y0) + 55), (0, 0, 255), 3)
        if args.zoom != 1:
            frame = cv2.resize(frame, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_CUBIC)
        out = PROJ_OUT / f"probe_t{t:07.2f}.png"
        PROJ_OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame)
        b64 = base64.b64encode(out.read_bytes()).decode()
        print(f"\n===== t={t}s  ({out.name}, {frame.shape[1]}x{frame.shape[0]}) =====")
        try:
            print(ask(args.base_url, args.model, args.api_key, b64, args.question))
        except Exception as e:  # noqa: BLE001
            print(f"  调用失败: {type(e).__name__}: {e}")
    cap.release()


PROJ_OUT = Path("out/probe_fp")

if __name__ == "__main__":
    main()
