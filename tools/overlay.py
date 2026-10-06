#!/usr/bin/env python3
"""幻唐志战斗统计 · 悬浮窗。

为什么需要它：手机扫码的本质是「拿手机当第二块屏幕」。但如果你只有一块显示器、
又是直接在电脑上打游戏，那最合适的是**一块始终置顶的小窗，压在游戏画面上** ——
既不用腾出手看手机，也不用切窗口。

数据来源：runner.py 写的 out/live/stats.json（panel.html 是给人看的，这是给程序看的）。

技术选择：tkinter —— Windows 官方 Python 自带，不用装任何额外东西。
（注意：Homebrew 的 macOS Python 常常没编 _tkinter，但 Windows 官方安装包一定有。）

操作：
    左键拖动     移动位置
    右键         菜单（透明度 / 字号 / 显示几个单位 / 退出）
    关闭按钮 ×   退出
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # 软导入：没 tkinter 的机器上，下面的纯数据函数仍然可用（也方便测）
    import tkinter as tk
    from tkinter import font as tkfont
    HAVE_TK = True
except ImportError:
    tk = None  # type: ignore[assignment]
    tkfont = None  # type: ignore[assignment]
    HAVE_TK = False

BG = "#0f1216"
PANEL = "#161a20"
FG = "#e6edf3"
DIM = "#7d8590"
DMG = "#ffab4a"
HEAL = "#5ee08a"
ACC = "#58a6ff"

CJK_FONTS = ("Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑", "PingFang SC",
             "Heiti SC", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "SimSun")


def layout_rows(rows: dict, top_n: int) -> tuple[list[tuple[str, str, str, bool, bool]], int]:
    """把某一回合的单位数据排成要显示的行。

    返回 (行列表, 被省略的条数)。每行是 (显示名, 伤害文本, 治疗文本, 有伤害, 有治疗)。
    单独抽出来是为了能脱离 GUI 测 —— tkinter 不是到处都装得上。
    """
    ranked = sorted(rows.items(), key=lambda kv: -(kv[1]["damage"] + kv[1]["heal"]))
    hidden = max(0, len(ranked) - top_n)
    out: list[tuple[str, str, str, bool, bool]] = []
    for unit, v in ranked[:top_n]:
        name = str(unit)
        if len(name) > 8:
            name = name[:7] + "…"
        out.append((name,
                    str(v["damage"]) if v["damage"] else "–",
                    f"+{v['heal']}" if v["heal"] else "",
                    bool(v["damage"]), bool(v["heal"])))
    return out, hidden


def pick_family(root) -> str:
    if not HAVE_TK:
        return "TkDefaultFont"
    have = set(tkfont.families(root))
    for n in CJK_FONTS:
        if n in have:
            return n
    return "TkDefaultFont"


def load_stats(path: Path) -> dict | None:
    """读 stats.json；文件还没生成/正写到一半都算「没数据」，不抛异常。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def latest_round(stats: dict) -> tuple[str, dict] | None:
    """取最新回合。回合号可能是 int，也可能是 '?'（还没识别出来）。"""

    def key(k: str):
        try:
            return (1, int(k))
        except (TypeError, ValueError):
            return (0, 0)

    keys = [k for k in stats if stats[k]]
    if not keys:
        return None
    k = max(keys, key=key)
    return k, stats[k]


class Overlay:
    def __init__(self, root: tk.Tk, stats_path: Path, top_n: int, alpha: float,
                 interval_ms: int = 1000):
        self.root = root
        self.stats_path = stats_path
        self.top_n = top_n
        self.interval = interval_ms
        self.fam = pick_family(root)
        self._drag = (0, 0)
        self._photo = None  # 防 GC

        root.overrideredirect(True)          # 去掉标题栏
        root.attributes("-topmost", True)    # 永远压在游戏上面
        root.attributes("-alpha", alpha)
        root.configure(bg=PANEL)

        # 外框：一条细边，免得在深色游戏画面上糊成一片
        outer = tk.Frame(root, bg=ACC, bd=0)
        outer.pack(fill="both", expand=True, padx=1, pady=1)
        self.box = tk.Frame(outer, bg=PANEL, bd=0)
        self.box.pack(fill="both", expand=True, padx=1, pady=1)

        # ── 标题栏 ──
        bar = tk.Frame(self.box, bg=PANEL)
        bar.pack(fill="x", padx=8, pady=(6, 2))
        self.title = tk.Label(bar, text="幻唐志统计  等待数据…", bg=PANEL, fg=ACC,
                              font=(self.fam, 10, "bold"), anchor="w")
        self.title.pack(side="left")
        close = tk.Label(bar, text="×", bg=PANEL, fg=DIM, font=(self.fam, 12), cursor="hand2")
        close.pack(side="right")
        close.bind("<Button-1>", lambda _e: root.destroy())

        # ── 正文（每次刷新重建）──
        self.body = tk.Frame(self.box, bg=PANEL)
        self.body.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self.foot = tk.Label(self.box, text="", bg=PANEL, fg=DIM,
                             font=(self.fam, 8), anchor="w")
        self.foot.pack(fill="x", padx=8, pady=(0, 6))

        # ── 拖动 ──
        for w in (root, outer, self.box, bar, self.title, self.foot, self.body):
            w.bind("<Button-1>", self._press)
            w.bind("<B1-Motion>", self._motion)

        # ── 右键菜单 ──
        self.menu = tk.Menu(root, tearoff=0)
        self.menu.add_command(label="置顶开关", command=self.toggle_top)
        sub = tk.Menu(self.menu, tearoff=0)
        for a in (0.95, 0.85, 0.75, 0.6, 0.45):
            sub.add_command(label=f"{int(a * 100)}%",
                            command=lambda v=a: root.attributes("-alpha", v))
        self.menu.add_cascade(label="透明度", menu=sub)
        sub2 = tk.Menu(self.menu, tearoff=0)
        for n in (5, 8, 10, 14, 20):
            sub2.add_command(label=f"{n} 个", command=lambda v=n: setattr(self, "top_n", v))
        self.menu.add_cascade(label="显示单位数", menu=sub2)
        self.menu.add_separator()
        self.menu.add_command(label="退出", command=root.destroy)
        for w in (root, outer, self.box, bar, self.title, self.body, self.foot):
            w.bind("<Button-3>", self._menu)

        root.geometry("250x170+40+40")  # 默认靠左上角，免得挡战斗画面
        self.refresh()

    # ── 交互 ──

    def _press(self, e):
        self._drag = (e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y())

    def _motion(self, e):
        self.root.geometry(f"+{e.x_root - self._drag[0]}+{e.y_root - self._drag[1]}")

    def _menu(self, e):
        try:
            self.menu.tk_popup(e.x_root, e.y_root)
        finally:
            self.menu.grab_release()

    def toggle_top(self):
        cur = bool(self.root.attributes("-topmost"))
        self.root.attributes("-topmost", not cur)

    # ── 渲染 ──

    def refresh(self):
        data = load_stats(self.stats_path)
        for w in self.body.winfo_children():
            w.destroy()

        got = latest_round((data or {}).get("stats") or {})
        if not got:
            self.title.config(text="幻唐志统计  等待数据…")
            tk.Label(self.body, text="还没抓到数字。\n战斗开始后会自动出现。",
                     bg=PANEL, fg=DIM, font=(self.fam, 9), justify="left",
                     anchor="w").pack(fill="x")
            self.foot.config(text=f"数据: {self.stats_path.name}")
            self.root.after(self.interval, self.refresh)
            return

        rnd, rows = got
        rnd_txt = f"第 {rnd} 回合" if str(rnd).isdigit() else "回合未识别"
        self.title.config(text=f"幻唐志统计   {rnd_txt}")

        ranked = sorted(rows.items(), key=lambda kv: -(kv[1]["damage"] + kv[1]["heal"]))
        shown, hidden = layout_rows(rows, self.top_n)
        grid = tk.Frame(self.body, bg=PANEL)
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1)

        for i, (name, d, h, has_d, has_h) in enumerate(shown):
            tk.Label(grid, text=name, bg=PANEL, fg=FG, font=(self.fam, 9),
                     anchor="w").grid(row=i, column=0, sticky="w")
            tk.Label(grid, text=d, bg=PANEL, fg=DMG if has_d else DIM,
                     font=(self.fam, 9, "bold"), anchor="e", width=6).grid(row=i, column=1, sticky="e")
            tk.Label(grid, text=h, bg=PANEL, fg=HEAL if has_h else DIM,
                     font=(self.fam, 9), anchor="e", width=6).grid(row=i, column=2, sticky="e")

        if hidden:
            tk.Label(grid, text=f"…还有 {hidden} 个", bg=PANEL, fg=DIM,
                     font=(self.fam, 8), anchor="w").grid(row=len(shown), column=0, sticky="w")

        td = sum(v["damage"] for v in rows.values())
        th = sum(v["heal"] for v in rows.values())
        self.foot.config(text=f"合计  伤害 {td}  治疗 {th}　·　{(data or {}).get('updated', '')}")
        self.root.after(self.interval, self.refresh)


def main() -> int:
    ap = argparse.ArgumentParser(description="幻唐志战斗统计 · 置顶悬浮窗")
    ap.add_argument("--stats", default="out/live/stats.json", help="runner 写的 stats.json")
    ap.add_argument("--top-n", type=int, default=8, help="一屏显示几个单位")
    ap.add_argument("--alpha", type=float, default=0.85, help="不透明度 0~1")
    ap.add_argument("--interval", type=int, default=1000, help="刷新间隔（毫秒）")
    args = ap.parse_args()

    try:
        if not HAVE_TK:
            raise ImportError("no tkinter")
        root = tk.Tk()
    except Exception as e:  # noqa: BLE001
        print(f"打不开悬浮窗（{e}）。")
        print("这通常是因为你的 Python 没带 tkinter（Windows 官方安装包自带）。")
        print("退路：直接双击 out/live/panel.html 用浏览器看，")
        print("     或者用 Windows PowerToys 的「始终置顶」(Win+Ctrl+T) 把它钉在游戏上。")
        return 1

    Overlay(root, Path(args.stats), args.top_n, args.alpha, args.interval)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
