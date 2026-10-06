#!/usr/bin/env python3
"""端到端管线：战斗视频 → 每回合 × 每单位 的伤害/治疗统计表 + 手机面板。

架构（本地触发 + 云端读数）:
    ┌─ 本地 CV 检测（高召回，免费，87ms/帧）
    │     ↓ 只在画面出现大数字时触发
    ├─ 抓触发帧（每回合通常 3~8 帧）
    │     ↓ 打包
    ├─ 云端视觉模型读整帧 → 严格 JSON（值 / 类型 / 位置 / 归属）
    │     ↓
    ├─ 跨帧去重聚合 → 每回合 × 每单位统计
    └─ 手机面板 panel.html（扫码看）

用法:
    # 只用本地检测结果出个草稿表（不联网，精度低，用来看链路）
    python tools/run_pipeline.py "video.mp4" --rotate 270 --reader local --out out/run_bd

    # 正式跑（要联网 + API key）
    export VLM_API_KEY=sk-xxx
    python tools/run_pipeline.py "video.mp4" --rotate 270 --reader vlm \\
        --base-url https://api.siliconflow.cn/v1 --model Qwen/Qwen2.5-VL-72B-Instruct \\
        --out out/run_bd

    # 已经手工拿到 vlm_raw.json，跳过读图直接出表
    python tools/run_pipeline.py "video.mp4" --rotate 270 --reader file \\
        --vlm-json out/vlm/vlm_raw.json --out out/run_bd
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
PY = str(PROJ / ".venv/bin/python")


def sh(cmd: list[str], quiet: bool = False) -> None:
    print(f"\n$ {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, cwd=PROJ, capture_output=quiet, text=True)
    if r.returncode != 0:
        print(r.stdout or "", r.stderr or "")
        raise SystemExit(f"命令失败: {cmd[1]}")
    if quiet and r.stdout:
        for line in r.stdout.strip().splitlines()[-6:]:
            print("   " + line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--reader", default="local", choices=["local", "vlm", "file"])
    ap.add_argument("--vlm-json", default="", help="--reader file 时用")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--limit", type=int, default=40, help="最多送几帧给视觉模型（控成本）")
    ap.add_argument("--out", default="out/run")
    ap.add_argument("--skip-detect", action="store_true", help="复用已有的 numbers.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. 本地检测 ────────────────────────────────────────────────
    det = out / "det"
    numbers = det / "numbers.csv"
    if not args.skip_detect or not numbers.exists():
        sh([PY, str(HERE / "detect_numbers.py"), args.video, "--rotate", str(args.rotate),
            "--fps", "5", "--families", "warm,green", "--out", str(det)], quiet=True)

    # ── 2. 挑触发帧 ───────────────────────────────────────────────
    trig = out / "trig"
    sh([PY, str(HERE / "pick_triggers.py"), args.video, "--rotate", str(args.rotate),
        "--numbers", str(numbers), "--out", str(trig), "--limit", str(args.limit)], quiet=True)

    # ── 3. 读数 ───────────────────────────────────────────────────
    vlm_json = Path(args.vlm_json) if args.vlm_json else out / "vlm/vlm_raw.json"
    if args.reader == "vlm":
        cmd = [PY, str(HERE / "vlm_reader.py"), str(trig), "--glob", "--out", str(out / "vlm")]
        if args.base_url:
            cmd += ["--base-url", args.base_url]
        if args.model:
            cmd += ["--model", args.model]
        if args.api_key:
            cmd += ["--api-key", args.api_key]
        sh(cmd, quiet=False)
    elif args.reader == "local":
        # 用本地检测的读数当"模型输出"，只为把链路跑通看效果（精度低）
        conv_dir = out / "vlm"
        conv_dir.mkdir(parents=True, exist_ok=True)
        by_frame: dict[int, list[dict]] = {}
        idx2t = {}
        with (trig / "triggers.csv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                idx2t[int(r["frame"])] = float(r["t"])
        for r in csv.DictReader(numbers.open(encoding="utf-8")):
            by_frame.setdefault(int(r["frame"]), []).append(r)
        frames_json = []
        for fi, t in sorted(idx2t.items(), key=lambda kv: kv[1]):
            nums = [{
                "value": int(x["value"]) if x["value"].isdigit() else 0,
                "type": "heal" if x["color_family"] == "green" else "damage",
                "x_pct": round(100 * float(x["x"]) / 2868, 1),
                "y_pct": round(100 * float(x["y"]) / 1320, 1),
                "unit": None, "confidence": "low",
            } for x in by_frame.get(fi, [])]
            frames_json.append({"frame": f"t{t:08.2f}", "round": None, "numbers": nums, "notes": "local"})
        vlm_json.write_text(json.dumps(frames_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[local 模式] 用本地检测结果冒充读数 → {vlm_json}")
    else:
        if not vlm_json.exists():
            raise SystemExit(f"--reader file 但找不到 {vlm_json}")

    # ── 4. 聚合出表 ───────────────────────────────────────────────
    sh([PY, str(HERE / "round_stats.py"), str(vlm_json), "--out", str(out),
        "--low-conf-counts"], quiet=False)

    print(f"\n完成。面板: {(out / 'panel.html').resolve()}")


if __name__ == "__main__":
    main()
