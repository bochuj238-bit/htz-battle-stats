#!/usr/bin/env python3
"""验证云端视觉 API 是否可用：一次最小调用，确认 key / base-url / model 三样都对。

不会打印 key 本身（只显示掩码）。用真实的一帧触发图做测试，顺便看读数质量。

用法:
    ./.venv/bin/python tools/check_api.py                       # 读 .env
    ./.venv/bin/python tools/check_api.py --model qwen-vl-plus  # 换模型
    ./.venv/bin/python tools/check_api.py --list                # 试一批常见模型名
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent

PROMPT = """这是回合制网游《幻唐志》的战斗画面截图。请找出画面里正在漂浮的**伤害数字**（橘黄大字）和**治疗数字**（荧光绿大字）。
忽略：等级前缀(140/145)、出手顺序编号、状态层数、徽章、气血数值、金币、聊天文字、金色技能名汉字。
只输出 JSON，不要解释：
{"numbers":[{"value":整数,"type":"damage"或"heal","x_pct":0~100,"y_pct":0~100,"unit":"名字或null","confidence":"high/medium/low"}], "notes":"简短说明"}"""

CANDIDATES = [
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", ["qwen-vl-max", "qwen-vl-plus", "qwen3-vl-plus", "qwen-vl-max-latest"]),
    ("https://open.bigmodel.cn/api/paas/v4", ["glm-4v-plus"]),
    ("https://api.siliconflow.cn/v1", ["Qwen/Qwen2.5-VL-72B-Instruct"]),
]


def load_env() -> None:
    env = PROJ / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def mask(k: str) -> str:
    return f"{k[:6]}…{k[-4:]}（长度 {len(k)}）" if len(k) > 12 else "（太短，可疑）"


def call(base_url: str, model: str, key: str, img_b64: str, timeout: int = 90) -> tuple[bool, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]}],
        "temperature": 0, "max_tokens": 1200,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return True, data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        return False, f"HTTP {e.code}: {body}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--list", action="store_true", help="依次试一批常见模型名")
    args = ap.parse_args()

    load_env()
    key = os.environ.get("VLM_API_KEY", "")
    if not key:
        raise SystemExit("没找到 VLM_API_KEY（放 .env 或 export）")
    print(f"key = {mask(key)}")

    img = Path(args.image) if args.image else next(iter(sorted((PROJ / "out/trig4/bd").glob("*.png"))), None)
    if img is None or not img.exists():
        raise SystemExit("找不到测试图片")
    b64 = base64.b64encode(img.read_bytes()).decode()
    print(f"测试图 = {img.name}  ({img.stat().st_size // 1024} KB)\n")

    tries: list[tuple[str, str]] = []
    if args.list:
        for base, models in CANDIDATES:
            tries += [(base, m) for m in models]
    else:
        tries = [(args.base_url or os.environ.get("VLM_BASE_URL", CANDIDATES[0][0]),
                  args.model or os.environ.get("VLM_MODEL", CANDIDATES[0][1][0]))]

    for base, model in tries:
        print(f"→ {base}  model={model}")
        ok, out = call(base, model, key, b64)
        if ok:
            print("  ✓ 可用\n")
            print(out[:900])
            print("\n【这一组就是可用配置】")
            print(f"  VLM_BASE_URL={base}")
            print(f"  VLM_MODEL={model}")
            return
        print(f"  ✗ {out}\n")

    print("全部失败。请把上面的报错发给助手确认服务商。")


if __name__ == "__main__":
    sys.exit(main())
