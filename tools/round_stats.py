#!/usr/bin/env python3
"""把视觉模型读出的帧级数字，聚合成「每回合 × 每单位」的伤害/治疗统计表。

输入: vlm_reader.py 产出的 vlm_raw.json
输出: 控制台表格 + CSV + 手机友好的 HTML 面板（供指挥官扫码看）

去重规则（回合制里同一个数字会在连续几帧都在飘）:
  同一个 (类型, 单位, 数值) 且横向位置差 < x_tol，且出现时间间隔 < gap 秒 → 算同一次
  间隔 >= gap 秒 → 算作新的一次（多段连击/连续两回合）

用法:
  python tools/round_stats.py out/vlm/vlm_raw.json --out out/stats
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HTML_TPL = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>幻唐志 战斗统计</title>
<style>
 body{{margin:0;padding:12px;background:#12151c;color:#e8ecf4;
      font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
 h1{{font-size:17px;margin:0 0 10px}} .sub{{color:#8892a6;font-size:12px;margin-bottom:12px}}
 table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
 th,td{{padding:6px 8px;text-align:right;border-bottom:1px solid #232936}}
 th{{color:#8892a6;font-weight:500;font-size:12px;position:sticky;top:0;background:#12151c}}
 td.name,th.name{{text-align:left}} .dmg{{color:#ffb454}} .heal{{color:#5ee08a}}
 .round{{margin:16px 0 6px;font-weight:600;color:#a9b6cc}}
 .unit-row td{{font-size:13px}} .zero{{color:#4a5364}}
</style></head><body>
<h1>幻唐志 · 回合伤害/治疗统计</h1>
<div class="sub">更新 {updated} · 数据源：视觉模型整帧读数 · 共 {n_frames} 帧 / {n_rounds} 回合</div>
{body}
</body></html>
"""


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def t_of(frame: str) -> float:
    """从 frame_t0068.70.png 这种文件名里取时间。"""
    import re

    m = re.search(r"t(\d+)\.?(\d*)", frame)
    return float(f"{m.group(1)}.{m.group(2) or 0}") if m else 0.0


def dedup(nums: list[dict], x_tol: float = 6.0, gap: float = 4.0) -> list[dict]:
    """跨帧去重。

    规则一（同一次飘字的多帧）：同类型、同单位、同数值、横向重合、间隔 < gap 秒 → 合并。

    规则二（同一次飘字的不同读数）：**同类型、同单位、横向重合、间隔 < gap 秒，
    但数值不同** —— 这是同一个数字在动画过程中被读了两次（一次读全、一次读残，
    实测 1245 被读成 124、592 被读成 592 同时还有一条 51952）。
    这种情况保留**位数更多**的那个读数，而不是当成两次伤害。
    """
    keep: list[dict] = []
    for n in sorted(nums, key=lambda r: r["t"]):
        dup = False
        for k in keep:
            if k["type"] != n["type"] or k["unit"] != n["unit"]:
                continue
            if abs(k["x_pct"] - n["x_pct"]) > x_tol or (n["t"] - k["t"]) >= gap:
                continue
            # 规则一：数值也对得上 → 同一次
            if abs(k["value"] - n["value"]) <= max(1, 0.02 * n["value"]):
                dup = True
                k["seen"] += 1
                break
            # 规则二：同一数字的不同读数。只在**数值有包含关系**时才合并：
            #   124 是 1245 的前缀（读残了）→ 留长的；
            #   592 是 51952 的后缀（前面多了杂点）→ 留短的。
            # 不这么限的话，会把两次真实连击（实测 186 和 2037 相隔 2.8 秒）也吞掉。
            a, b = str(k["value"]), str(n["value"])
            long_v, short_v = (n, k) if len(b) > len(a) else (k, n)
            if (n["t"] - k["t"]) < 1.5 and len(str(long_v["value"])) > len(str(short_v["value"])):
                ls, ss = str(long_v["value"]), str(short_v["value"])
                if ls.endswith(ss):      # 多余的在前 → 留短的
                    k["value"], k["seen"] = short_v["value"], k["seen"] + 1
                    dup = True
                    break
                if ls.startswith(ss):    # 读残的在后 → 留长的
                    k["value"], k["seen"] = long_v["value"], k["seen"] + 1
                    dup = True
                    break
            continue
        if not dup:
            keep.append({**n, "seen": 1})
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="out/stats")
    ap.add_argument("--x-tol", type=float, default=6.0)
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--low-conf-counts", action="store_true", help="把 confidence=low 的也计入")
    args = ap.parse_args()

    frames = load(Path(args.src))
    recs: list[dict] = []
    for f in frames:
        t = t_of(f.get("frame", ""))
        for n in f.get("numbers", []) or []:
            conf = str(n.get("confidence", "high")).lower()
            if conf == "low" and not args.low_conf_counts:
                continue
            unit = n.get("unit") or "未归属"
            recs.append(
                {
                    "t": t,
                    "frame": f.get("frame"),
                    "round": f.get("round"),
                    "type": n.get("type"),
                    "value": int(n.get("value", 0)),
                    "unit": unit,
                    "x_pct": float(n.get("x_pct", 0.0)),
                    "conf": conf,
                }
            )

    # 回合号：优先用模型读的；读不到就用「帧序号」当兜底回合
    if any(r["round"] is None for r in recs):
        unknown = sorted({r["frame"] for r in recs if r["round"] is None})
        for r in recs:
            if r["round"] is None:
                r["round"] = f"?{unknown.index(r['frame']) + 1}"

    per_round: dict = defaultdict(list)
    for r in recs:
        per_round[r["round"]].append(r)

    stats: dict = defaultdict(lambda: defaultdict(lambda: {"damage": 0, "heal": 0, "hits": 0, "heals": 0}))
    for rnd, rows in sorted(per_round.items(), key=lambda kv: str(kv[0])):
        for e in dedup(rows, args.x_tol, args.gap):
            s = stats[rnd][e["unit"]]
            if e["type"] == "heal":
                s["heal"] += e["value"]
                s["heals"] += 1
            else:
                s["damage"] += e["value"]
                s["hits"] += 1

    # 控制台表格
    for rnd in sorted(per_round, key=lambda x: str(x)):
        rows = stats[rnd]
        if not rows:
            continue
        td = sum(v["damage"] for v in rows.values())
        th = sum(v["heal"] for v in rows.values())
        print(f"\n第 {rnd} 回合   伤害合计 {td}   治疗合计 {th}")
        print(f"  {'单位':<14}{'伤害':>8}{'段数':>5}{'治疗':>8}{'次数':>5}")
        for unit, v in sorted(rows.items(), key=lambda kv: -kv[1]["damage"] - kv[1]["heal"]):
            print(f"  {unit:<14}{v['damage']:>8}{v['hits']:>5}{v['heal']:>8}{v['heals']:>5}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "round_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "unit", "damage", "hits", "heal", "heals"])
        for rnd in sorted(per_round, key=lambda x: str(x)):
            for unit, v in sorted(stats[rnd].items(), key=lambda kv: -kv[1]["damage"] - kv[1]["heal"]):
                w.writerow([rnd, unit, v["damage"], v["hits"], v["heal"], v["heals"]])

    with open(out / "events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "frame", "t", "unit", "type", "value", "x_pct", "conf"])
        for rnd in sorted(per_round, key=lambda x: str(x)):
            for e in dedup(per_round[rnd], args.x_tol, args.gap):
                w.writerow([rnd, e["frame"], f"{e['t']:.2f}", e["unit"], e["type"], e["value"], f"{e['x_pct']:.1f}", e["conf"]])

    # HTML 面板
    import datetime

    body = []
    for rnd in sorted(per_round, key=lambda x: str(x)):
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
    (out / "panel.html").write_text(
        HTML_TPL.format(
            updated=datetime.datetime.now().strftime("%H:%M:%S"),
            n_frames=len(frames),
            n_rounds=len(per_round),
            body="\n".join(body),
        ),
        encoding="utf-8",
    )
    print(f"\n  CSV → {out}/round_stats.csv")
    print(f"  事件明细 → {out}/events.csv")
    print(f"  手机面板 → {out}/panel.html")


if __name__ == "__main__":
    main()
