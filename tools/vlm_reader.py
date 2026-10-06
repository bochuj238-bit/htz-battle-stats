#!/usr/bin/env python3
"""视觉模型读数器（架构主线）。

为什么走这条路：实测本地 OCR 管线在《幻唐志》这种特效爆炸的画面上精确率只有 15~32%，
而视觉模型**读整帧**又准又省事（实测准确读出「3 个绿 602 群体治疗」「橘 1720 伤害」「绿 696」，
并给出位置与归属单位）。回合制节奏慢，一回合只需要抓几帧，调用成本极低。

架构:
  每回合抓 N 帧（默认 3 帧：回合中段 / 末段 / 结束后）
    → 视觉模型读整帧 → 严格 JSON
    → round_stats.py 去重聚合 → 每单位×每回合统计表 → 网页面板

支持的模型后端（都是 OpenAI 兼容的 chat/completions）:
  --base-url https://api.siliconflow.cn/v1  --model Qwen/Qwen2.5-VL-72B-Instruct   # 硅基流动
  --base-url https://open.bigmodel.cn/api/paas/v4 --model glm-4v-plus              # 智谱
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen-vl-max # 通义
  --base-url https://openrouter.ai/api/v1 --model google/gemini-2.0-flash         # OpenRouter
  API key 从 --api-key 或环境变量 VLM_API_KEY 读。

用法:
  export VLM_API_KEY=sk-xxx
  python tools/vlm_reader.py "frame.png" --out out/vlm/frame.json
  python tools/vlm_reader.py "out/frames_bd" --glob --out out/vlm
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

PROMPT = """你在读一张回合制网络游戏《幻唐志：逍遥外传》的战斗画面截图。任务是：找出画面里所有**正在漂浮的伤害数字**和**回血数字**，并判断它属于哪个单位。

判定要点：
- **浮动伤害数字**：橘黄色，字体很大（约占画面高度 4%~6%），从单位身上飘起后消失。
- **浮动治疗数字**：荧光绿色，同样很大。
- 以下都**不算**浮动数字，必须忽略：单位名字左边的等级前缀（如 140、145、飞升140）、出手顺序编号（戴在菱形/圆形底框里的小数字）、状态层数、头顶徽章（🐾5 💧11 之类）、气血/魔法数值、金币奖励、聊天框文字。
- 单位名字显示在单位**脚下**，颜色为绿色（我方）或金黄色（敌/怪）；伤害数字从单位胸口上浮到头顶上方。
  所以**判断归属时以「数字与哪个单位在水平方向上对齐」为准**，不要简单取屏幕上离得最近的名字。

只输出一个 JSON 对象，不要任何解释、不要 markdown 代码块：

{
  "round": 整数，画面左上角「第 N 回合」的 N，看不清就填 null,
  "numbers": [
    {
      "value": 整数,
      "type": "damage" 或 "heal",
      "x_pct": 数字中心横坐标占画面宽度的百分比(0~100),
      "y_pct": 数字中心纵坐标占画面高度的百分比(0~100),
      "unit": "该数字所属单位的名字（按水平对齐判断）；实在看不出来填 null",
      "confidence": "high" 或 "medium" 或 "low"
    }
  ],
  "notes": "有什么看不清或被特效遮挡的，简短说明；没有就填空字符串"
}
"""


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call_vlm(base_url: str, model: str, api_key: str, image_b64: str, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 2000,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_json_block(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"模型没返回 JSON: {text[:200]}")
    return json.loads(t[start : end + 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="图片路径，或图片目录（配 --glob）")
    ap.add_argument("--glob", action="store_true", help="src 是目录，处理里面所有 png")
    ap.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", "https://api.siliconflow.cn/v1"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    ap.add_argument("--out", default="out/vlm")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要发送的内容，不真调用")
    args = ap.parse_args()

    files = sorted(Path(args.src).glob("*.png")) if args.glob else [Path(args.src)]
    if not files:
        raise SystemExit("没有图片")
    if not args.api_key and not args.dry_run:
        raise SystemExit("缺少 API key：设环境变量 VLM_API_KEY，或用 --api-key")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for p in files:
        print(f"读 {p.name} …", flush=True)
        b64 = encode_image(p)
        if args.dry_run:
            print(f"  [dry-run] 会发送 {len(b64) // 1024} KB 图片给 {args.model}")
            continue
        try:
            raw = call_vlm(args.base_url, args.model, args.api_key, b64)
        except urllib.error.HTTPError as e:
            print(f"  ✗ HTTP {e.code}: {e.read()[:300]!r}")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {type(e).__name__}: {e}")
            continue
        try:
            obj = parse_json_block(raw)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ JSON 解析失败: {e}\n原文: {raw[:300]}")
            continue
        obj["frame"] = p.name
        results.append(obj)
        n_dmg = sum(1 for x in obj.get("numbers", []) if x.get("type") == "damage")
        n_heal = sum(1 for x in obj.get("numbers", []) if x.get("type") == "heal")
        print(f"  ✓ 回合={obj.get('round')}  伤害{n_dmg} 治疗{n_heal}  {obj.get('notes', '')[:60]}")

    if results:
        (out / "vlm_raw.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  → {out}/vlm_raw.json   共 {len(results)} 帧")


if __name__ == "__main__":
    sys.exit(main())
