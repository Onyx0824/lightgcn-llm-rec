"""
Week 4/6 - 增強比例（20% / 50% / 100%）效益彙整。

做三件事：
1. 每個比例：α=1.0 相對 α=0 的配對增益（Recall / NDCG），含 95% 信賴區間
2. 相鄰比例之間（20%→50%、50%→100%）的配對差，檢驗「多生成是否真的有額外好處」
3. 從 results/week4/timing.jsonl 加總各比例的累計生成耗時，輸出 CSV 供 Week 6 畫效益曲線

用法（專案根目錄）：
    python scripts/compare_ratios.py
    python scripts/compare_ratios.py --ratios 0.2 0.5 1.0 --alpha 1.0
    # item-only 消融：α=1.0 用 itemonly 的結果、α=0 控制組沿用主實驗，成本只算 item 生成時間
    python scripts/compare_ratios.py --prefix week4fe_aug_itemonly --control_prefix week4fe_aug \
        --cost_side items --out_csv results/week6_tier_table_itemonly.csv
輸出：終端機表格 + results/week6_tier_table.csv
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats


def load(results_dir, prefix, ratio, alpha):
    out = {}
    for f in glob.glob(str(results_dir / f"{prefix}_r{ratio}_a{alpha}_s*.json")):
        m = re.search(r"_s(\d+)\.json$", f)
        if m:
            with open(f, encoding="utf-8") as fh:
                out[int(m.group(1))] = json.load(fh)
    return out


def ci95(d):
    n = len(d)
    sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return d.mean() - half, d.mean() + half


def cumulative_hours(timing_path, tier):
    users = items = 0.0
    if not timing_path.exists():
        return None, None
    with open(timing_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["max_ratio"] <= tier + 1e-9:
                if r["kind"] == "users":
                    users += r["wall_sec"]
                elif r["kind"] == "items":
                    items += r["wall_sec"]
    return users / 3600, items / 3600


def pick(uh, ih, side):
    return uh + ih if side == "total" else (uh if side == "users" else ih)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--prefix", default="week4fe_aug")
    ap.add_argument("--control_prefix", default=None, help="α=0 控制組的檔名前綴，預設同 --prefix")
    ap.add_argument("--cost_side", choices=["total", "users", "items"], default="total",
                    help="成本只計哪一端的生成時間（item-only 消融用 items）")
    ap.add_argument("--ratios", nargs="+", default=["0.2", "0.5", "1.0"])
    ap.add_argument("--a0", default="0.0")
    ap.add_argument("--a1", default="1.0")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--timing", default="results/week4/timing.jsonl")
    ap.add_argument("--out_csv", default="results/week6_tier_table.csv")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    metrics = [f"recall@{args.k}", f"ndcg@{args.k}"]
    tiers = {}
    rows = []

    print(f"== 各比例：α={args.a1} 相對 α={args.a0} 的配對增益（全體 test user，95% CI）==")
    for r in args.ratios:
        a0 = load(rdir, args.control_prefix or args.prefix, r, args.a0)
        a1 = load(rdir, args.prefix, r, args.a1)
        seeds = sorted(set(a0) & set(a1))
        if len(seeds) < 3:
            print(f"ratio={r}: 配對 seed 不足（{len(seeds)}），略過")
            continue
        tiers[r] = (a1, seeds)
        uh, ih = cumulative_hours(Path(args.timing), float(r))
        row = {"ratio": r, "n_seeds": len(seeds)}
        if uh is not None:
            row.update(gen_hours_users=round(uh, 2), gen_hours_items=round(ih, 2),
                       gen_hours_total=round(uh + ih, 2),
                       gen_hours_used=round(pick(uh, ih, args.cost_side), 2), cost_side=args.cost_side)
        line = f"ratio={r} (n={len(seeds)}"
        if uh is not None:
            line += f", 累計生成 {pick(uh, ih, args.cost_side):.1f} 小時[{args.cost_side}]"
        print(line + ")")
        for m in metrics:
            c = np.array([a0[s]["test_metrics"][m] for s in seeds])
            a = np.array([a1[s]["test_metrics"][m] for s in seeds])
            d = a - c
            lo, hi = ci95(d)
            t, p = stats.ttest_rel(a, c)
            print(f"  {m}: 控制 {c.mean():.4f} → 增強 {a.mean():.4f} | 增益 {d.mean():+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] (相對 {d.mean() / c.mean() * 100:+.1f}%)  p={p:.4f}")
            key = m.split("@")[0]
            row.update({f"{key}_ctrl": round(c.mean(), 5), f"{key}_aug": round(a.mean(), 5),
                        f"{key}_gain": round(d.mean(), 5), f"{key}_gain_ci_lo": round(lo, 5),
                        f"{key}_gain_ci_hi": round(hi, 5)})
        rows.append(row)

    rs = [r for r in args.ratios if r in tiers]
    if len(rs) >= 2:
        print(f"\n== 相鄰比例的配對差（α={args.a1}，同 seed）：多生成有沒有額外好處？ ==")
        for r1, r2 in zip(rs[:-1], rs[1:]):
            (a1_lo, s1), (a1_hi, s2) = tiers[r1], tiers[r2]
            seeds = sorted(set(s1) & set(s2))
            u1, i1 = cumulative_hours(Path(args.timing), float(r1))
            u2, i2 = cumulative_hours(Path(args.timing), float(r2))
            extra = (f"，多花 {pick(u2, i2, args.cost_side) - pick(u1, i1, args.cost_side):.1f} 小時[{args.cost_side}]"
                     if u1 is not None else "")
            print(f"ratio {r1} → {r2}（n={len(seeds)}{extra}）")
            for m in metrics:
                d = np.array([a1_hi[s]["test_metrics"][m] - a1_lo[s]["test_metrics"][m] for s in seeds])
                lo, hi = ci95(d)
                t, p = stats.ttest_rel(
                    [a1_hi[s]["test_metrics"][m] for s in seeds],
                    [a1_lo[s]["test_metrics"][m] for s in seeds])
                print(f"  {m}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  為正 {int((d > 0).sum())}/{len(d)}  p={p:.4f}")

    if rows:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "ratio", k))
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\n已輸出 {out}（Week 6 效益曲線用）")


if __name__ == "__main__":
    main()
