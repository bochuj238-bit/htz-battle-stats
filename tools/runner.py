#!/usr/bin/env python3
"""实时主循环：屏幕 → 触发帧 → 云端视觉模型读数 → 回合统计 → 手机面板。

这是产品的入口。设计要点（都是实测踩出来的）:
  · 本地检测只当**触发器**（召回 9/9、免费、87ms/帧），不当读数器（精确率仅 15~32%）
  · 浮动数字只活 0.6~0.8s，所以 5fps 采样 + 新颖性触发（同一个 (值, 位置) 只触发一次）
  · 云端模型一次读多帧，**并发**发送，消除单帧 1~3s 的网络延迟
  · 回合用「第N回合」OCR 切分（满帧 OCR 实测 100% 可靠）
  · 回合结束后立刻刷新面板，手机扫码即可看

用法:
    # 用录好的视频当信号源（模拟实时，也用于离线调参）
    python tools/runner.py --source video:fight.mp4 --rotate 270

    # 真实时抓屏（Windows 用 mss/dxcam，macOS 用 mss）
    python tools/runner.py --source screen --monitor 1 --fps 5

    # 云端读数（不填就用 mock 模式，只跑链路不联网）
    export VLM_API_KEY=sk-xxx
    python tools/runner.py --source video:fight.mp4 --rotate 270 \\
        --base-url https://api.siliconflow.cn/v1 --model Qwen/Qwen2.5-VL-72B-Instruct
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detect_numbers import (  # noqa: E402
    COLOR_RANGES,
    UI_ZONES,
    color_masks,
    find_tokens,
    in_ui_zone,
    merge_row,
    read_crop,
)
from extract_candidates import apply_rotate  # noqa: E402
from round_stats import HTML_TPL, dedup  # noqa: E402

VLM_PROMPT = """你在读一张回合制网络游戏《幻唐志：逍遥外传》的战斗画面截图。

本地检测器已在画面里找到下面这些**疑似数字**（位置是占画面的百分比，读数是本地 OCR 结果，可能不准）：
{cands_placeholder}

请对**每一条**给出判定，并读出该处的**准确数值**。

判定只能是下面 6 个之一：
- `damage` = 正在漂浮的**伤害数字**：橘黄色，很大（约占画面高度 5%），从单位身上飘起
- `heal`   = 正在漂浮的**治疗数字**：荧光绿色，同样很大
- `level`  = **等级前缀**：紧贴在单位名字**左边**的小白字 140 / 145 / 飞升140
- `turn`   = **出手顺序编号**：戴在菱形或圆形小底框里的小数字（1~10）
- `ui`     = 其他 UI：状态层数、头顶徽章、气血/魔法数值、金币奖励、聊天文字、金色技能名汉字
- `none`   = 那个位置没有数字（本地误检，或数字已经消失了）

判断要点：**伤害/治疗数字是从单位胸口上浮到头顶的悬空大字**；而等级前缀永远紧贴在名字左边、字号小很多。
归属单位时：单位名字在单位**脚下**（绿色=我方，金黄=敌方/怪），**以「数字与哪个单位水平方向对齐」为准**，不要取屏幕上离得最近的名字；定不下来就填 `-`。

## 输出格式（很重要）

**每条一行**，不要 JSON、不要表格、不要任何解释：

```
序号 判定 数值 归属单位 置信度
```

例如：
```
1 damage 1546 极·羊力大仙 high
2 level - - -
3 none - - -
```

规则：
- 数值填整数；判定不是 damage/heal 时，数值和单位都填 `-`
- 置信度只能是 high / medium / low
- 有几条本地候选就输出几行，一行不能少
"""


def _parse_lines(txt: str, cands: list[dict]) -> list[dict]:
    """解析逐行文本输出：`序号 判定 数值 归属单位 置信度`。

    为什么不用 JSON：实测模型输出的 JSON 有 21/38 次语法错误导致解析失败；
    逐行文本几乎不会出错，而且更短、更便宜。
    """
    nums = []
    for line in txt.splitlines():
        line = line.strip().strip("`|").strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        head = parts[0].rstrip(".、:|）)").lstrip("#")
        if not head.isdigit():
            continue
        i, kind = int(head), parts[1].lower().strip(".:|`*")
        if kind not in ("damage", "heal"):
            continue
        if not parts[2].strip("-—").isdigit():
            continue
        src = cands[i - 1] if 1 <= i <= len(cands) else {}
        # 类型由本地颜色族硬约束：绿色大数字一定是治疗，橘色一定是伤害（已标定过的事实）。
        # 不这么做的话，同一个数字在不同帧会被判成不同类型（实测 1245 一次 damage 一次 heal）。
        if src.get("family") == "green":
            kind = "heal"
        elif src.get("family") == "warm":
            kind = "damage"
        unit = parts[3] if len(parts) > 3 else "-"
        unit = unit.strip("。.,，、·-)））")  # 去掉尾部标点：否则「事事如心顺」和「事事如心顺.」会被当成两个单位
        conf = parts[4].lower() if len(parts) > 4 else "medium"
        nums.append({"value": int(parts[2]), "type": kind,
                     "unit": None if unit in ("-", "—", "null", "None") else unit,
                     "x_pct": round(src.get("x_pct", 0), 1), "y_pct": round(src.get("y_pct", 0), 1),
                     "confidence": conf if conf in ("high", "medium", "low") else "medium"})
    return nums


def downscale(frame: np.ndarray, max_w: int) -> np.ndarray:
    """发送前缩图。原图 2868px 直接送又慢又容易超请求体限制；缩到 1600px 后
    浮动数字仍有 ~30px 高，足够识别。"""
    if not max_w or frame.shape[1] <= max_w:
        return frame
    k = max_w / frame.shape[1]
    return cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def load_env() -> None:
    """从脚本旁边的 .env / config.txt 读配置（Windows 上比设环境变量方便）。

    支持两种写法（key=value，一行一条，# 开头是注释）：
        VLM_API_KEY=sk-xxxx
        base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
        model=qwen-vl-max
    """
    for name in (".env", "config.txt"):
        for p in (HERE.parent / name, HERE / name):
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip().lower(), v.strip().strip('"\'')
                if not v:
                    continue
                alias = {"vlm_api_key": "VLM_API_KEY", "api_key": "VLM_API_KEY", "key": "VLM_API_KEY",
                         "base_url": "VLM_BASE_URL", "vlm_base_url": "VLM_BASE_URL",
                         "model": "VLM_MODEL", "vlm_model": "VLM_MODEL"}
                os.environ.setdefault(alias.get(k, k), v)


def build_prompt(cands: list[dict]) -> str:
    """注意：不能用 str.format()——提示词里 JSON 的 {} 会被当成占位符（实测 KeyError）。"""
    lines = []
    for i, c in enumerate(cands, 1):
        family = "绿" if c.get("family") == "green" else "橘"
        lines.append(f'{i}. 位置 横{c["x_pct"]:.0f}% 纵{c["y_pct"]:.0f}%　本地读数 "{c.get("value", "?")}"（{family}色，高 {int(c["h"])}px）')
    return VLM_PROMPT.replace("{cands_placeholder}", "\n".join(lines) if lines else "（无）")


# ─────────────────────────── 信号源 ───────────────────────────

class VideoSource:
    """用视频文件当信号源：按原速节拍吐帧（模拟实时）。"""

    def __init__(self, path: str, rotate: int, fps: float, speed: float = 1.0, loop: bool = False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"打不开视频 {path}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.rotate = rotate
        self.step = max(1, int(round(self.src_fps / fps)))
        self.delay = self.step / self.src_fps / speed
        self.loop = loop
        self.i = 0
        self.t0 = time.time()

    def frames(self):
        while True:
            ok, fr = self.cap.read()
            if not ok:
                if not self.loop:
                    return
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.t0 = time.time()
                self.i = 0
                continue
            if self.i % self.step:
                self.i += 1
                continue
            self.i += 1
            # 节拍对齐
            target = self.t0 + (self.i / self.src_fps)
            lag = target - time.time()
            if lag > 0:
                time.sleep(lag)
            yield self.t, apply_rotate(fr, self.rotate)

    @property
    def t(self) -> float:
        return self.i / self.src_fps


def _mss_open():
    """mss 新版改用 MSS，旧名 mss() 会报警告。这里兼容两边。"""
    import mss  # type: ignore

    return (getattr(mss, "MSS", None) or mss.mss)()


class ScreenSource:
    """真实时抓屏。"""

    def __init__(self, monitor: int, fps: float, region: str = ""):
        try:
            import mss  # type: ignore
        except ImportError:
            raise SystemExit("抓屏需要 mss：./.venv/bin/pip install mss") from None
        self.sct = _mss_open()
        mons = self.sct.monitors
        self.mon = mons[monitor] if monitor < len(mons) else mons[1]
        if region:
            x, y, w, h = (int(v) for v in region.split(","))
            self.mon = {"left": x, "top": y, "width": w, "height": h}
        self.delay = 1.0 / fps
        self.n = 0

    def frames(self):
        while True:
            t0 = time.time()
            shot = np.array(self.sct.grab(self.mon))[:, :, :3]
            yield self.n / max(1, int(1 / self.delay)), shot
            self.n += 1
            lag = self.delay - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)


# ─────────────────────────── 触发器 ───────────────────────────

class Trigger:
    """本地检测 → 新颖性触发。只负责「别漏」，不负责「判真伪」。"""

    def __init__(self, ttl: float = 2.5, drop_values: set[str] | None = None, min_h_pct: float = 0.038,
                 max_cands: int = 60, min_digits: int = 3, static_after: int = 20):
        self.ttl = ttl
        self.drop = drop_values if drop_values is not None else {"140", "145"}
        self.min_h_pct = min_h_pct
        self.max_cands = max_cands
        self.min_digits = min_digits
        self.static_after = static_after
        self.seen: dict[tuple, float] = {}
        self.pos_hits: dict[tuple, int] = {}
        self.static_zones: set[tuple] = set()
        self.suppressed = 0

    @staticmethod
    def _pos_key(c: dict) -> tuple:
        """位置指纹。格子取 40px：真飘字往上升，不会长时间停在同一个格子；UI 元素位置固定。"""
        return (c["family"], round(c["cx"] / 40), round(c["cy"] / 40))

    def candidates(self, frame: np.ndarray) -> list[dict]:
        fh, fw = frame.shape[:2]
        min_h = int(self.min_h_pct * fh)
        max_h = int(0.10 * fh)
        out: list[dict] = []
        for fam, mask in color_masks(frame, ("warm", "green")).items():
            for x, y, w, h in merge_row(find_tokens(mask, min_h, max_h)):
                if w < 12 or h < min_h:
                    continue
                cx, cy = x + w / 2, y + h / 2
                if in_ui_zone(cx, cy, fw, fh):
                    continue
                out.append({"x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy, "family": fam,
                            "x_pct": 100 * cx / fw, "y_pct": 100 * cy / fh})
        return out

    def fire(self, frame: np.ndarray, t: float) -> list[dict]:
        """返回本次「值得读图」的候选（空列表 = 不触发）。"""
        cands = self.candidates(frame)
        for c in cands:
            if "value" not in c:
                txt, conf = read_crop(frame, (int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"])), pad=6)
                c["value"], c["ocr_conf"] = txt.strip(), conf
        # 在线自标定：某位置**连续**多帧都出现东西 = 固定 UI，永久排除。
        # 实测值：某 UI 元素的 x 永远钉在 1914px（66.7%），被本地 OCR 反复读成 101。
        #
        # ⚠️ 必须用「连续帧」而不是「累计次数」：累计算的话，
        # 任何一个位置在这场上百帧里都会攒够次数，会把真数字的落点也误杀
        # （实测累计版把触发从 38 掐到 15，真值全丢）。
        # 真飘字只活 0.6~0.8s（≈3~4 帧）且一直在往上飘，不可能在同一个格子里连着 12 帧。
        present = {self._pos_key(c) for c in cands}
        for k in [k for k in self.pos_hits if k not in present]:
            del self.pos_hits[k]
        for c in cands:
            k = self._pos_key(c)
            n = self.pos_hits.get(k, 0) + 1
            self.pos_hits[k] = n
            if n == self.static_after and k not in self.static_zones:
                self.static_zones.add(k)
                if len(self.static_zones) <= 12:
                    print(f"  [自标定] 横{c['x_pct']:.0f}% 纵{c['y_pct']:.0f}% 连续 {n} 帧不动 → 记为固定 UI")
        cands = [c for c in cands if self._pos_key(c) not in self.static_zones]
        # 判据一（最有用的一条）：真数字 OCR 出来是**纯数字**；
        # 汉字/字母（城镇招牌「福/心/万」、技能名、LOGO 文字）一律是误检。
        # 注：数字会因逐字浮动而被拆成几块（实测 696 → 6 + 96），这是正常的，
        # 只要有一块是纯数字就会触发，读值由云端模型负责。
        # 判据二：候选数暴增 = 城镇/结算画面（实测战斗帧 ≤ 13，城镇帧 50+）。
        if len(cands) > self.max_cands:
            self.suppressed += 1
            return []
        digit_cands = [c for c in cands
                       if c["value"].isdigit()
                       and len(c["value"]) >= self.min_digits
                       and c["value"] not in self.drop]
        novel = [c for c in digit_cands if self._novel(c, t)]
        for c in digit_cands:
            self.seen[(c["value"], round(c["cx"] / 60))] = t
        return novel

    def _novel(self, c: dict, t: float) -> bool:
        # 去重键用 (数值, 横向位置)：
        #   只用横向位置的话，同一个 x 上的误检会把真数字挤掉（实测 592 就是这样漏的）；
        #   不能带 y —— 数字是往上飘的，y 每帧都变，带上就每帧都算「新东西」。
        return self.seen.get((c["value"], round(c["cx"] / 60)), -1e9) < t - self.ttl


# ─────────────────────────── 回合识别 ───────────────────────────

class RoundTracker:
    """用「第N回合」满帧 OCR 切分回合（实测 100% 可靠）。"""

    _engine: RapidOCR | None = None

    def __init__(self, every: float = 2.0, roi: tuple[float, float, float, float] = (0.0, 0.0, 0.22, 0.12)):
        self.every = every
        self.roi = roi
        self.last = -1e9
        self.round: int | None = None

    def update(self, frame: np.ndarray, t: float) -> int | None:
        if t - self.last < self.every:
            return self.round
        self.last = t
        if RoundTracker._engine is None:
            RoundTracker._engine = RapidOCR()
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = int(self.roi[0] * w), int(self.roi[1] * h), int(self.roi[2] * w), int(self.roi[3] * h)
        res = RoundTracker._engine(frame[y0:y1, x0:x1])
        if res is None or res.txts is None:
            return self.round
        import re

        # 注意：裁小区域后检测器会把「第1 回合」拆成 ['第','1','回合'] 几个 token，
        # 所以必须**拼接全文**再匹配，逐 token 匹配永远命中不了。
        joined = "".join(res.txts)
        m = re.search(r"第\s*(\d+)\s*回合", joined)
        if m:
            self.round = int(m.group(1))
        return self.round


# ─────────────────────────── 云端读数 ───────────────────────────

def call_vlm(base_url: str, model: str, key: str, images_b64: list[str], timeout: int = 90,
             cands: list[dict] | None = None) -> dict:
    content: list[dict] = [{"type": "text", "text": build_prompt(cands or [])}]
    for b in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "temperature": 0, "max_tokens": 1500}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    return {"numbers": _parse_lines(txt, cands or []), "raw": txt[:300]}


class Reader:
    """并发读数。mock 模式下不发网络请求，只用本地候选当占位（用于跑通链路）。"""

    def __init__(self, args):
        self.args = args
        self.pool = ThreadPoolExecutor(max_workers=args.concurrency)
        self.raw: list[dict] = []
        self.lock = threading.Lock()
        self.n_calls = 0

    def submit(self, frame: np.ndarray, t: float, rnd: int | None, cands: list[dict]):
        if self.args.mock:
            with self.lock:
                self.raw.append({"frame": f"t{t:08.2f}", "t": t, "round": rnd,
                                 "numbers": [{"value": None, "type": "heal" if c["family"] == "green" else "damage",
                                              "x_pct": round(c["x_pct"], 1), "y_pct": round(c["y_pct"], 1),
                                              "unit": None, "confidence": "low"} for c in cands],
                                 "notes": "mock"})
            return
        ok, buf = cv2.imencode(".png", downscale(frame, self.args.max_width),
                               [cv2.IMWRITE_PNG_COMPRESSION, 4])
        if not ok:
            return
        b64 = base64.b64encode(buf.tobytes()).decode()
        f = self.pool.submit(call_vlm, self.args.base_url, self.args.model, self.args.api_key, [b64],
                             90, cands)
        f.add_done_callback(lambda fut, t=t, rnd=rnd: self._done(fut, t, rnd))

    def _done(self, fut, t: float, rnd: int | None):
        try:
            obj = fut.result()
        except urllib.error.HTTPError as e:
            print(f"  ! t={t:.1f} HTTP {e.code}: {e.read()[:150]!r}")
            return
        except Exception as e:  # noqa: BLE001
            print(f"  ! t={t:.1f} {type(e).__name__}: {e}")
            return
        obj.setdefault("frame", f"t{t:08.2f}")
        obj["t"] = t
        obj["round"] = obj.get("round") or rnd
        with self.lock:
            self.raw.append(obj)
            self.n_calls += 1

    def drain(self) -> list[dict]:
        with self.lock:
            out, self.raw = self.raw, []
        return out


# ─────────────────────────── 汇总 ───────────────────────────

def summarize(events: list[dict], drop_values: set[int] | None = None) -> tuple[dict, dict]:
    """把事件按 回合 → 单位 汇总。events 里每项含 round/unit/type/value/x_pct/t。"""
    drop = drop_values if drop_values is not None else {140, 145}
    rounds: dict = {}
    for e in events:
        r = e.get("round") if e.get("round") is not None else "?"
        rounds.setdefault(r, []).append(e)
    stats: dict = {}
    for r, rows in rounds.items():
        agg: dict = {}
        rows = [e for e in rows
                if isinstance(e.get("value"), int) and e["value"] > 0 and e["value"] not in drop]
        for ev in dedup(rows):
            s = agg.setdefault(ev.get("unit") or "未归属", {"damage": 0, "heal": 0, "hits": 0, "heals": 0})
            if ev.get("type") == "heal":
                s["heal"] += ev["value"]
                s["heals"] += 1
            else:
                s["damage"] += ev["value"]
                s["hits"] += 1
        stats[r] = agg
    return rounds, stats


def write_panel(path: Path, stats: dict, n_frames: int, extra: str = "") -> None:
    import datetime

    body = []
    for rnd in sorted(stats, key=lambda x: str(x)):
        rows = stats[rnd]
        if not rows:
            continue
        td = sum(v["damage"] for v in rows.values())
        th = sum(v["heal"] for v in rows.values())
        body.append(f'<div class="round">第 {rnd} 回合　<span class="dmg">伤害 {td}</span>　<span class="heal">治疗 {th}</span></div>')
        body.append('<table><tr><th class="name">单位</th><th>伤害</th><th>段</th><th>治疗</th><th>次</th></tr>')
        for unit, v in sorted(rows.items(), key=lambda kv: -kv[1]["damage"] - kv[1]["heal"]):
            d = f'<span class="dmg">{v["damage"]}</span>' if v["damage"] else '<span class="zero">–</span>'
            h = f'<span class="heal">{v["heal"]}</span>' if v["heal"] else '<span class="zero">–</span>'
            body.append(f'<tr class="unit-row"><td class="name">{unit}</td><td>{d}</td><td>{v["hits"] or ""}</td><td>{h}</td><td>{v["heals"] or ""}</td></tr>')
        body.append("</table>")
    path.write_text(HTML_TPL.format(updated=datetime.datetime.now().strftime("%H:%M:%S"),
                                    n_frames=n_frames, n_rounds=len(stats),
                                    body="\n".join(body) + extra), encoding="utf-8")

    # 顺带给悬浮窗（和别的程序）留一份机器可读的。
    # 只有一块屏幕、又要在电脑上打游戏时，看的是这个小窗，不是浏览器。
    try:
        (path.parent / "stats.json").write_text(json.dumps(
            {"updated": datetime.datetime.now().strftime("%H:%M:%S"),
             "n_frames": n_frames, "n_rounds": len(stats), "stats": stats},
            ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _local_ipv4s() -> list[str]:
    """尽力枚举本机所有 IPv4 地址（不依赖第三方库）。"""
    import socket

    found: list[str] = []
    try:  # 路由法：拿到「出口」那张网卡的地址
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            found.append(s.getsockname()[0])
        finally:
            s.close()
    except Exception:  # noqa: BLE001
        pass
    try:  # 主机名反查：能带出其它网卡
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":  # Windows 上主机名反查常常只给一个，ipconfig 更全
        try:
            import re
            import subprocess

            txt = subprocess.run(["ipconfig"], capture_output=True, text=True, timeout=15).stdout
            found += re.findall(r"IPv4[^\d]*((?:\d{1,3}\.){3}\d{1,3})", txt)
        except Exception:  # noqa: BLE001
            pass
    return list(dict.fromkeys(found))


def lan_ip() -> str:
    """挑一个「手机真能连上」的局域网地址。

    坑：直接连 8.8.8.8 拿出口地址，在开了 VPN/代理的机器上会拿到 VPN 网卡的地址
    （实测拿到 198.18.0.1 —— 保留给网络测试的段，手机永远连不上）。
    所以枚举所有网卡，按「像不像家用路由器」打分，优先 192.168.x.x。
    """
    import ipaddress

    def score(ip: str) -> int:
        try:
            a = ipaddress.ip_address(ip)
        except ValueError:
            return -999
        if a.is_loopback or a.is_link_local or a.is_multicast or a.is_unspecified:
            return -999
        if a.is_private:
            if ip.startswith("192.168."):
                return 100  # 家用路由器最标准的段
            if ip.startswith("10."):
                return 90
            if ip.startswith("172."):
                return 85
            return 80
        return 10  # 公网地址（多半是 VPN 出口）：能用，但排在私网之后

    cands = _local_ipv4s()
    if not cands:
        return "127.0.0.1"
    cands.sort(key=score, reverse=True)
    return cands[0] if score(cands[0]) > 0 else "127.0.0.1"


def print_qr(url: str, png_path: Path | None = None) -> None:
    try:
        import segno  # type: ignore
    except ImportError:
        print(f"   手机访问: {url}   （装 segno 可显示二维码: pip install segno）")
        return
    qr = segno.make(url)
    if png_path is not None:  # Windows 控制台渲染二维码不可靠，存一份图最稳
        try:
            qr.save(str(png_path), scale=6, border=2)
            print(f"   二维码图: {png_path}")
        except Exception:  # noqa: BLE001
            pass
    try:
        qr.terminal(compact=True)
    except Exception:  # noqa: BLE001
        pass


# ─────────────────────────── 主循环 ───────────────────────────

class PanelServer:
    """把输出目录用 HTTP 服务出去，手机连同一个 WiFi 扫码就能看面板。

    用标准库 http.server，不引入额外依赖（Windows 上少一个装不上的风险）。
    """

    def __init__(self, directory: Path, port: int = 8088, index: str = "panel.html"):
        import http.server
        import socketserver

        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text(
            f'<!doctype html><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="0;url=/{index}">', encoding="utf-8")
        root = str(directory)

        class Handler(http.server.SimpleHTTPRequestHandler):
            # 服务根目录必须在这里传下去。踩过的坑：写 functools.partial(..., directory=root)
            # 再去继承 handler.func，等于继承了「没带 directory 的」原始类，
            # 根目录会静默退回进程 cwd —— 表现是 / 能打开（列出的是程序目录）、
            # panel.html 却 404。继承 + 在 __init__ 里传参才对。
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=root, **kw)

            def log_message(self, *a):  # 别把日志刷满屏幕
                return

        self.httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        self.port = port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self) -> str:
        return f"http://{lan_ip()}:{self.port}/panel.html"


def list_monitors() -> None:
    try:
        import mss  # type: ignore
    except ImportError:
        print("需先安装 mss：pip install mss")
        return
    try:
        with _mss_open() as sct:
            for i, m in enumerate(sct.monitors):
                tag = "（全部屏幕拼起来）" if i == 0 else ""
                print(f"  --monitor {i}   {m['width']}x{m['height']} @ ({m['left']},{m['top']}) {tag}")
    except Exception as e:  # noqa: BLE001
        print(f"抓屏初始化失败：{e}")
        return
    print("\n游戏窗口不是全屏的话，可以用 --region x,y,w,h 只抓那块区域（更快也更准）")


def main() -> None:
    load_env()  # 必须在建 argparse 之前：下面几个默认值要从环境变量里读
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="screen", help="screen 或 video:<路径>")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="video 源用")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--region", default="", help="抓屏区域 x,y,w,h")
    ap.add_argument("--speed", type=float, default=1.0, help="video 源播放倍速，>1 加速")
    ap.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "qwen-vl-max"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    ap.add_argument("--concurrency", type=int, default=6, help="同时发几个读数请求")
    ap.add_argument("--max-width", type=int, default=1600, help="发送前把画面缩到这个宽度（0=不缩）")
    ap.add_argument("--mock", action="store_true", help="不联网，只跑链路")
    ap.add_argument("--min-digits", type=int, default=3,
                    help="触发所需的数字位数，默认 3（实测：38 次触发 / 已知真值 9/9 全中）；"
                         "调成 2 更保守但触发量翻倍")
    ap.add_argument("--refresh", type=float, default=2.0, help="面板刷新间隔(秒)")
    ap.add_argument("--out", default="out/live")
    ap.add_argument("--port", type=int, default=8088, help="手机看面板的端口")
    ap.add_argument("--no-serve", action="store_true", help="不起 HTTP 服务（只写本地文件）")
    ap.add_argument("--no-qr", action="store_true", dest="no_qr",
                    help="不打印二维码（只用悬浮窗看时，别刷屏）")
    ap.add_argument("--list-monitors", action="store_true", help="列出所有屏幕编号后退出", dest="list_monitors")
    args = ap.parse_args()

    if args.list_monitors:
        list_monitors()
        return

    if not args.mock and not args.api_key:
        raise SystemExit("缺 API key：export VLM_API_KEY=…  或加 --mock 跑离线链路")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panel = out / "panel.html"

    if args.source.startswith("video:"):
        src = VideoSource(args.source.split(":", 1)[1], args.rotate, args.fps, args.speed)
    else:
        src = ScreenSource(args.monitor, args.fps, args.region)

    trig, rt, reader = Trigger(min_digits=args.min_digits), RoundTracker(), Reader(args)
    events: list[dict] = []
    n_trig = 0
    n_frame = 0
    last_panel = 0.0

    def ingest(obj: dict) -> None:
        for n in obj.get("numbers") or []:
            events.append({"round": obj.get("round"), "unit": n.get("unit"),
                           "type": n.get("type"), "value": n.get("value"),
                           "x_pct": n.get("x_pct", 0), "conf": str(n.get("confidence", "high")),
                           "t": obj.get("t", 0)})

    print(f"开始。读数={'mock（不发网络请求）' if args.mock else args.model}")

    # 先把面板落一次盘：不然手机一扫码看到的是 404，要等两秒才有内容
    _, stats0 = summarize([])
    write_panel(panel, stats0, 0, extra='<div class="sub">等待第一个回合…</div>')

    srv = None
    if not args.no_serve:
        try:
            srv = PanelServer(out, args.port)
            print(f"面板文件 {panel.resolve()}")
            print(f"手机扫码看（需同一个 WiFi）：{srv.url()}")
            alts = [i for i in _local_ipv4s() if i != lan_ip()][:3]
            if alts and not lan_ip().startswith("192.168."):
                print("   连不上就换这几个地址试：")
                for a in alts:
                    print(f"     http://{a}:{args.port}/panel.html")
            if not args.no_qr:
                print_qr(srv.url(), out / "qrcode.png")
        except OSError as e:
            print(f"  ! 端口 {args.port} 起不来（{e}），改用本地文件模式")
    print()

    try:
        for t, frame in src.frames():
            n_frame += 1
            rnd = rt.update(frame, t)
            novel = trig.fire(frame, t)
            if novel:
                n_trig += 1
                reader.submit(frame, t, rnd, novel)
                top = sorted(novel, key=lambda c: -c["h"])[:4]
                desc = " ".join(f"{c['value']}({c['family'][:2]}{int(c['h'])}px)" for c in top)
                rnd_txt = f"第{rnd}回合" if rnd else "回合未识别"
                print(f"  t={t:7.1f}s {rnd_txt} 触发 #{n_trig:<3} {desc}")

            for obj in reader.drain():
                ingest(obj)

            if time.time() - last_panel > args.refresh:
                last_panel = time.time()
                _, stats = summarize(events)
                write_panel(panel, stats, n_frame,
                            extra=f'<div class="sub">触发 {n_trig} 次 · 读数 {reader.n_calls} 次 · 事件 {len(events)} · 抑制非战斗帧 {trig.suppressed}</div>')
    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        # 视频放完/手动停止时，还有若干请求在路上；必须等齐再汇总，
        # 否则最后几帧的读数会被丢掉（shutdown(wait=False) 会直接丢弃未完成的任务）。
        pending = reader.pool._work_queue.qsize()  # noqa: SLF001
        if pending:
            print(f"\n等待 {pending} 个读数请求返回…")
        reader.pool.shutdown(wait=True)
        time.sleep(0.3)
        for obj in reader.drain():
            ingest(obj)
        _, stats = summarize(events)
        write_panel(panel, stats, n_frame)
        (out / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n触发 {n_trig} 次 / 读数 {reader.n_calls} 次 / 事件 {len(events)} 条")
        print(f"面板 → {panel.resolve()}")


if __name__ == "__main__":
    main()
