#!/usr/bin/env python3
"""打一个「发给测试者」的包：预填 API key + 附上给对方的说明。

用法：
    # 预填 key（对方拿到就能直接用，不用自己申请）
    ./.venv/bin/python tools/make_test_zip.py --key sk-xxxxxxxx

    # 不带 key（对方得自己填，或者只能跑离线测试）
    ./.venv/bin/python tools/make_test_zip.py

⚠️ 带 key 发出去 = 对方能花你的钱。建议：
    1. 去控制台**单独建一个 key** 专用于测试（别用你日常那个）
    2. 测试完就去控制台把它删掉/重置

产物：发布包/幻唐志战斗统计工具-测试版.zip
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # 项目根
PKG = ROOT / "发布包" / "幻唐志战斗统计工具-Windows"       # 干净的原始包
INTRO = ROOT / "发给测试者的话.txt"                       # 给对方的说明
OUT = ROOT / "发布包" / "幻唐志战斗统计工具-测试版.zip"

CONFIG_TPL = """# 幻唐志战斗统计工具 · 配置（测试版）
# 这个 key 是发你这台测试用的，测完会被作废，随便用。

VLM_API_KEY={key}
VLM_BASE_URL={base}
VLM_MODEL={model}
REGION=
MONITOR=1
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    # 也可以不入参，用环境变量 VLM_API_KEY（避免 key 出现在命令行/ ps 里）
    ap.add_argument("--key", default=os.environ.get("VLM_API_KEY", ""),
                    help="预填进去的 API key；留空则读环境变量 VLM_API_KEY；再没有就让对方自己填")
    ap.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--model", default="qwen-vl-max")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if not PKG.is_dir():
        print(f"找不到原始包：{PKG}")
        return 1

    out = Path(args.out)
    tmp = out.parent / "_tmp_测试版"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # 1) 拷包（跳过 .venv / out / pycache，那些是本地状态的垃圾）
    n = 0
    for p in sorted(PKG.rglob("*")):
        rel = p.relative_to(PKG)
        if any(part in (".venv", "out", "__pycache__") for part in rel.parts):
            continue
        if p.name == ".DS_Store":
            continue
        dst = tmp / rel
        if p.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            n += 1

    # 2) 覆写 config.txt（预填 key）
    (tmp / "config.txt").write_text(
        CONFIG_TPL.format(key=args.key, base=args.base_url, model=args.model),
        encoding="utf-8")

    # 3) 附上给对方的说明（放最外层，一眼能看到）
    if INTRO.is_file():
        shutil.copy2(INTRO, tmp / "★先看这个★-测试说明.txt")
        n += 1

    # 4) 打包（必须用 zipfile：会正确写 UTF-8 标记位，Windows 解压中文名不乱码）
    #    外面套一层目录 —— 否则对方解压会撒出一堆散文件，很容易搞丢文件。
    if out.exists():
        out.unlink()
    top = "幻唐志战斗统计工具"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                z.write(p, Path(top) / p.relative_to(tmp))

    shutil.rmtree(tmp)

    size = out.stat().st_size / 1048576
    print(f"\n✓ 打好了：{out}")
    print(f"  大小 {size:.1f} MB，{n} 个文件")
    print(f"  API key: {'已预填 → 对方拿来就能用' if args.key else '未填 → 对方只能跑离线测试'}")
    if args.key:
        masked = args.key[:6] + "…" + args.key[-4:] if len(args.key) > 12 else "***"
        print(f"  填的是: {masked}")
        print("\n  ⚠️ 记得测试完去控制台把这个 key 删掉或重置。")
    print(f"\n  发给对方：这一个文件就够了。")
    print(f"  同时把「发给测试者的话.txt」的内容贴到聊天里（或让对方看包里的 ★先看这个★）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
