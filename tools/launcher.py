#!/usr/bin/env python3
"""幻唐志战斗统计工具 · 启动器（Windows 双击即用）。

这个文件负责所有「环境相关」的脏活，让 .bat 保持极简：
  1. 检查 Python
  2. 首次运行时建虚拟环境、装依赖（带国内镜像加速）
  3. 把随包的本地 OCR 模型放进虚拟环境
  4. 读取 / 引导填写 API key（存在 config.txt）
  5. 给出菜单，拼好命令行去调 tools/runner.py

用 Python 而不是 .bat 来做交互，是因为中文、路径、引号在 .bat 里全是坑。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # 工具包根目录
VENV = ROOT / ".venv"
CONFIG = ROOT / "config.txt"
IS_WIN = os.name == "nt"
PY_IN_VENV = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def pause(msg: str = "\n按回车键继续…") -> None:
    try:
        input(msg)
    except (EOFError, KeyboardInterrupt):
        pass


def say(s: str = "") -> None:
    print(s, flush=True)


def hr(title: str = "") -> None:
    say("\n" + "=" * 62)
    if title:
        say(f"  {title}")
        say("=" * 62)


# ─────────────────────────── 环境准备 ───────────────────────────

def venv_ok() -> bool:
    """虚拟环境存在、python 能跑、核心依赖已装好。"""
    if not PY_IN_VENV.exists():
        return False
    try:
        r = subprocess.run([str(PY_IN_VENV), "-c", "import cv2, numpy, onnxruntime, rapidocr, mss"],
                           capture_output=True, timeout=180)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _venv_site_packages() -> Path | None:
    """找到虚拟环境的 site-packages（Windows 和 mac/linux 布局不一样）。"""
    if IS_WIN:
        p = VENV / "Lib" / "site-packages"
        return p if p.is_dir() else None
    hits = sorted((VENV / "lib").glob("python*/site-packages"))
    return hits[0] if hits else None


def install_models() -> None:
    """把随包带的本地 OCR 模型塞进虚拟环境，省掉首次运行时去网上下载。

    模型只有 30MB，从 modelscope 下也没问题，但带上就不用等、也不怕网络抽风。
    拷不进去也无所谓 —— rapidocr 会自己去下，不会卡死。
    """
    src = ROOT / "models"
    if not src.is_dir():
        return
    sp = _venv_site_packages()
    if sp is None or not (sp / "rapidocr").is_dir():
        return
    dst = sp / "rapidocr" / "models"
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sorted(src.glob("*.onnx")):
        tgt = dst / f.name
        if tgt.is_file() and tgt.stat().st_size == f.stat().st_size:
            continue  # 已经在位
        try:
            shutil.copy2(f, tgt)
            copied += 1
        except Exception as e:  # noqa: BLE001
            say(f"  ! 模型 {f.name} 放不进去（{e}），首次运行时会自动下载")
    if copied:
        say(f"  ✓ 已放入 {copied} 个本地识别模型（离线可用）")


def setup_env() -> bool:
    hr("第一步：准备运行环境（只需做一次，约 2~5 分钟）")
    say(f"安装到: {VENV}")

    if not VENV.exists():
        say("\n[1/2] 创建虚拟环境…")
        try:
            venv.create(VENV, with_pip=True)
        except Exception as e:  # noqa: BLE001
            say(f"  创建失败：{e}")
            say("  如果提示缺少 venv 模块，请重装 Python 并勾选「Add Python to PATH」。")
            pause()
            return False
    else:
        say("\n[1/2] 虚拟环境已存在，跳过。")

    say("\n[2/2] 安装依赖（从清华镜像下载，国内快很多）…")
    req = ROOT / "requirements.txt"
    base = [str(PY_IN_VENV), "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    for extra in ([], ["-i", MIRROR]):
        cmd = base + ["-r", str(req)] + extra
        say(f"  $ {' '.join(cmd[2:])}")
        try:
            r = subprocess.run(cmd, timeout=1800)
        except subprocess.TimeoutExpired:
            say("  超时了，检查一下网络。")
            continue
        if r.returncode == 0:
            say("\n  ✓ 依赖装完了")
            install_models()
            return True
        say("  这次没成功，换下一个源重试…")

    say("\n✗ 依赖安装失败。可以手动执行：")
    say(f'  "{PY_IN_VENV}" -m pip install -r "{req}" -i {MIRROR}')
    pause()
    return False


# ─────────────────────────── 配置 ───────────────────────────

def read_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if CONFIG.exists():
        for line in CONFIG.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip().lower()] = v.strip()
    return cfg


def write_config(cfg: dict[str, str]) -> None:
    lines = [
        "# 幻唐志战斗统计工具 · 配置",
        "#",
        "# ======================= 必填 =======================",
        "# VLM_API_KEY  云端视觉模型的 API key。没填的话启动器会问你。",
        "#",
        "#   阿里云百炼（通义千问）控制台： https://bailian.console.aliyun.com/",
        "#   路径：登录 → 模型服务 → API-KEY 管理 → 创建我的 API-KEY",
        "#   新用户一般有免费额度；正常用的话一场战斗约 ¥0.3~0.5",
        "#",
        "# ==================== 选填 / 换服务商 ====================",
        "# 下面几家都是 OpenAI 兼容接口，改地址 + 模型名就能换，不用改代码。",
        "#",
        "#   阿里云百炼（已实测）  https://dashscope.aliyuncs.com/compatible-mode/v1",
        "#                         qwen-vl-max",
        "#   硅基流动              https://api.siliconflow.cn/v1",
        "#                         Qwen/Qwen2.5-VL-72B-Instruct",
        "#   智谱                  https://open.bigmodel.cn/api/paas/v4",
        "#                         glm-4v-plus",
        "#",
        "# REGION  游戏窗口在屏幕上的位置 x,y,w,h。留空 = 抓整个屏幕。",
        "#         启动器里选「2) 抓屏预览」可以边看边调。",
        "#",
        "# 注意：VLM_API_KEY 是你自己的密钥，别把这个文件发给别人。",
        "",
        f"VLM_API_KEY={cfg.get('vlm_api_key', '')}",
        f"VLM_BASE_URL={cfg.get('vlm_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}",
        f"VLM_MODEL={cfg.get('vlm_model', 'qwen-vl-max')}",
        f"REGION={cfg.get('region', '')}",
        f"MONITOR={cfg.get('monitor', '1')}",
        "",
    ]
    CONFIG.write_text("\n".join(lines), encoding="utf-8")


def ensure_config() -> dict[str, str]:
    cfg = read_config()
    if cfg.get("vlm_api_key"):
        return cfg
    hr("填 API key")
    say("这个工具靠云端视觉模型识别伤害数字，需要一个 API key（一场战斗约 ¥0.3~0.5）。")
    say("阿里云百炼控制台可以创建： https://bailian.console.aliyun.com/")
    say("（选「通义千问」→ API-KEY 管理 → 创建）\n")
    say("也可以直接编辑 config.txt，把 VLM_API_KEY= 后面填上。\n")
    try:
        key = input("把 key 粘进来后回车（直接回车则跳过，稍后自己编辑 config.txt）：").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""
    cfg["vlm_api_key"] = key
    cfg.setdefault("vlm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    cfg.setdefault("vlm_model", "qwen-vl-max")
    write_config(cfg)
    say(f"\n已写入 {CONFIG}")
    return cfg


# ─────────────────────────── 抓取区域 ───────────────────────────

def ask_monitor(cfg: dict[str, str]) -> str:
    cur = cfg.get("monitor", "1") or "1"
    try:
        v = input(f"屏幕编号（回车 = {cur}；不确定选菜单 3 看）：").strip()
    except (EOFError, KeyboardInterrupt):
        return cur
    if not v:
        return cur
    cfg["monitor"] = v
    write_config(cfg)
    return v


def ask_region(cfg: dict[str, str]) -> str:
    """问抓取区域。填过一次就记住，下次直接回车沿用。

    区域填错不会报错，只会「什么都识别不到」——所以这里引导用户先去预览。
    """
    cur = cfg.get("region", "")
    say("\n抓取区域：游戏不是全屏时只抓那一块，又快又准。")
    say("不确定怎么填 → 先选菜单「2) 抓屏预览」，看图量好再回来。")
    if cur:
        say(f"上次用的是 {cur}")
        prompt = "区域 x,y,w,h（回车沿用上次 / 填 - 改成抓整个屏幕）："
    else:
        prompt = "区域 x,y,w,h（回车 = 抓整个屏幕）："
    try:
        v = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return cur
    if v == "-":
        cfg["region"] = ""
        write_config(cfg)
        say("  已改成抓整个屏幕。")
        return ""
    if not v:
        return cur
    cfg["region"] = v
    write_config(cfg)
    say(f"  已记住：{v}")
    return v


def open_file(path: Path) -> None:
    """用系统默认看图工具打开（让用户不用自己找路径）。"""
    try:
        if IS_WIN:
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:  # noqa: BLE001
        say(f"  （自动打开失败，手动看这个文件：{path}）")


# ─────────────────────────── 跑 ───────────────────────────

def py_run(tool: str, extra: list[str], env_extra: dict[str, str] | None = None) -> int:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    cmd = [str(PY_IN_VENV), "-u", str(ROOT / "tools" / tool)] + extra
    try:
        return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    except KeyboardInterrupt:
        return 130


def run_runner(extra: list[str], with_overlay: bool = False) -> None:
    cfg = read_config()
    if not cfg.get("vlm_api_key") and "--mock" not in extra:
        say("\n⚠ config.txt 里还没有 VLM_API_KEY，无法调用云端模型。")
        say("  想先不联网看看效果，可以选菜单「5) 离线测试」。")
        pause()
        return
    env_extra = {
        "VLM_API_KEY": cfg.get("vlm_api_key", ""),
        "VLM_BASE_URL": cfg.get("vlm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "VLM_MODEL": cfg.get("vlm_model", "qwen-vl-max"),
    }
    out_dir = ROOT / "out" / "live"
    extra = extra + ["--out", str(out_dir)]

    hr("开始运行")
    say(f"模型: {env_extra['VLM_MODEL']}")
    if with_overlay:
        say("统计窗口会压在游戏上角（拖动移动 / 右键调透明度 / × 关闭）")
    say("（想停下来按 Ctrl+C）\n")

    ov = None
    if with_overlay:
        env = dict(os.environ, **env_extra)
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            ov = subprocess.Popen(
                [str(PY_IN_VENV), "-u", str(ROOT / "tools" / "overlay.py"),
                 "--stats", str(out_dir / "stats.json")],
                cwd=str(ROOT), env=env)
        except Exception as e:  # noqa: BLE001
            say(f"  ! 悬浮窗起不来（{e}）—— 用手机扫码看也行，地址下面会打。")
            ov = None

    try:
        py_run("runner.py", extra, env_extra)
    finally:
        if ov is not None and ov.poll() is None:
            ov.terminate()
    pause("\n回车返回菜单…")


def do_preview(cfg: dict[str, str]) -> None:
    hr("抓屏预览（先确认抓的是不是游戏画面）")
    say("这一步会抓一帧屏幕存成图片，让你亲眼看看框对没框对。")
    say("⚠ 先把游戏打开、进到战斗画面（或者主界面也行），再往下走。\n")
    mon = ask_monitor(cfg)
    cfg = read_config()
    region = ask_region(cfg)
    extra = ["--monitor", mon, "--out", "out/preview.png"]
    if region:
        extra += ["--region", region]
    say()
    rc = py_run("snapshot.py", extra)
    p = ROOT / "out" / "preview.png"
    if rc == 0 and p.exists():
        say(f"\n  马上打开图片给你看：{p}")
        open_file(p)
        say("\n  看到画面里【正好是游戏内容】就说明区域对了。")
        say("  如果框歪了 / 框太大太小，重新选菜单 2 再调一次。")
    pause()


def menu() -> None:
    while True:
        cfg = read_config()
        hr("幻唐志 · 回合战斗统计工具")
        key_state = "已配置" if cfg.get("vlm_api_key") else "**还没填**"
        region_state = cfg.get("region") or "整个屏幕"
        say(f"  API key: {key_state}    模型: {cfg.get('vlm_model', '-')}")
        say(f"  抓取区域: {region_state}")
        say()
        say("  1) 【正式用】实时统计 —— 抓游戏画面，悬浮窗 或 手机扫码看")
        say("  2) 抓屏预览 —— 第一次用必做：先看抓的是不是游戏画面")
        say("  3) 看有哪些屏幕（填上面坐标要参考它）")
        say("  4) 用录像文件跑一遍（复盘 / 验证效果）")
        say("  5) 离线测试（不联网、不花钱，只看能抓到哪些数字）")
        say("  6) 验证 API key 能不能用")
        say("  7) 重新填写 API key")
        say("  8) 重新安装依赖")
        say("  9) 悬浮窗预览（单看悬浮窗长什么样）")
        say("  0) 退出")
        say()
        try:
            c = input("选一个数字后回车：").strip()
        except (EOFError, KeyboardInterrupt):
            return

        if c == "1":
            cfg = read_config()
            say("\n开始前确认：")
            say("  1. 游戏已经打开、调成【窗口化】或【无边框窗口】")
            say("  2. 先把统计窗口拖到不挡操作的位置（比如右上角）\n")
            say("统计显示在哪里？")
            say("  1) 【推荐】电脑上悬浮窗 —— 一块置顶小窗压在游戏上，不用手机")
            say("  2) 手机扫码看 —— 适合有第二块屏幕，或者想拿旧手机当副屏")
            say("  3) 两种都要")
            try:
                how = (input("选一个（回车 = 1）：").strip() or "1")
            except (EOFError, KeyboardInterrupt):
                how = "1"
            if how not in ("1", "2", "3"):
                how = "1"
            mon = ask_monitor(cfg)
            cfg = read_config()
            region = ask_region(cfg)
            extra = ["--source", "screen", "--monitor", mon]
            if region:
                extra += ["--region", region]
            if how == "1":
                extra += ["--no-qr"]          # 只用悬浮窗，别刷二维码
            run_runner(extra, with_overlay=how in ("1", "3"))

        elif c == "2":
            do_preview(cfg)

        elif c == "3":
            hr("本机屏幕")
            py_run("runner.py", ["--list-monitors"])
            say("\n记下游戏窗口从哪开始、多大，去菜单 2 里填。")
            pause()

        elif c == "9":
            hr("悬浮窗预览（没在打游戏也能看效果）")
            stats = ROOT / "out" / "live" / "stats.json"
            if not stats.exists():
                say(f"还没数据（{stats} 不存在）。")
                say("先跑一次「1) 实时统计」，或者「5) 离线测试」生成一份。")
                pause()
                continue
            say("拖动移动 · 右键调透明度/字号 · × 关闭")
            ov = subprocess.Popen([str(PY_IN_VENV), "-u", str(ROOT / "tools" / "overlay.py"),
                                   "--stats", str(stats)], cwd=str(ROOT))
            pause("回车关掉悬浮窗…")
            if ov.poll() is None:
                ov.terminate()

        elif c == "4":
            try:
                path = input("\n把录像文件拖到这个窗口里，然后回车：").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                continue
            if not path or not Path(path).exists():
                say("没找到这个文件。")
                pause()
                continue
            try:
                rot = input("画面方向 0 还是 270？（手机竖屏录像一般 270，回车 = 270）：").strip() or "270"
            except (EOFError, KeyboardInterrupt):
                rot = "270"
            run_runner(["--source", f"video:{path}", "--rotate", rot, "--speed", "6"])

        elif c == "5":
            run_runner(["--source", "screen", "--mock"])

        elif c == "6":
            hr("验证 API key")
            py_run("check_api.py", [], {
                "VLM_API_KEY": cfg.get("vlm_api_key", ""),
                "VLM_BASE_URL": cfg.get("vlm_base_url", ""),
                "VLM_MODEL": cfg.get("vlm_model", ""),
            })
            pause()

        elif c == "7":
            cfg["vlm_api_key"] = ""
            write_config(cfg)
            ensure_config()
            pause()

        elif c == "8":
            setup_env()
            pause()

        elif c == "0":
            return
        else:
            say("没这个选项。")


def main() -> None:
    if IS_WIN:
        try:  # 让中文在 cmd 里正常显示
            subprocess.run(["chcp", "65001"], shell=True, capture_output=True)
        except Exception:  # noqa: BLE001
            pass
    say("=" * 62)
    say("  幻唐志 · 回合战斗统计工具")
    say("=" * 62)

    if not venv_ok():
        if not setup_env():
            return
    install_models()  # 换机器/新装 venv 后补放模型；已在位则跳过
    ensure_config()
    menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
