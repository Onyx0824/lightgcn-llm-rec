"""Step C summary: popularity-first vs random-order item augmentation at the same coverage.

For keep = 10/20/30/50% of items (item side only, alpha=1.0, fixed 25 epochs) it reports
  - Recall@20 / NDCG@20 of popularity-first (popXX) and random-order (itemonly_r0.X)
  - paired gain over alpha=0 (95% CI, t distribution) for both
  - paired difference popularity-first minus random (the key test), with p value and #seeds positive
  - share of train interactions on the augmented items, exact LLM generation hours of the selected items
  - share of the item-100% gain retained
Writes results/stepC/summary.csv and results/stepC/pop_vs_random.png.

Usage (project root):  python scripts/stepC_summary.py --seeds 0,1,2,3,4
Seeds are paired across conditions; only seeds present in ALL compared files are used.
"""
import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

K = 20


def load(prefix, ratio, alpha):
    out = {}
    for f in glob.glob(f"results/{prefix}_r{ratio}_a{alpha}_s*.json"):
        m = re.search(r"_s(\d+)\.json$", f)
        if m:
            with open(f, encoding="utf-8") as fh:
                out[int(m.group(1))] = json.load(fh)["test_metrics"]
    return out


def paired(a, b, seeds, metric):
    """b - a over seeds -> mean, ci_lo, ci_hi, p, n_pos"""
    x = np.array([a[s][metric] for s in seeds])
    y = np.array([b[s][metric] for s in seeds])
    d = y - x
    n = len(d)
    if n < 2:
        return d.mean() if n else float("nan"), float("nan"), float("nan"), float("nan"), int((d > 0).sum())
    se = d.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = stats.ttest_rel(y, x).pvalue
    return d.mean(), d.mean() - h, d.mean() + h, p, int((d > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--keeps", default="10,20,30,50")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--src_dir", default="results/week4")
    args = ap.parse_args()
    want = [int(s) for s in args.seeds.split(",")]
    keeps = [int(k) for k in args.keeps.split(",")]

    a0 = load("week4fe_aug", "1.0", "0.0")
    full = load("week4fe_aug_itemonly", "1.0", "1.0")

    # exact generation hours and interaction share
    elapsed = {}
    with open(Path(args.src_dir) / "items.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            elapsed[r["item_idx"]] = r.get("elapsed_sec_total", 0.0)
    with open(Path(args.src_dir) / "order.json", encoding="utf-8") as f:
        item_order = json.load(f)["item_order"]
    train = pd.read_csv(args.train)
    n_items = len(np.load(Path(args.src_dir) / "item_has.npy"))
    deg = np.bincount(train["item"].values, minlength=n_items).astype(float)
    total_hours_full = sum(elapsed.get(i, 0.0) for i in item_order) / 3600

    rows = []
    for kp in keeps:
        pop = load(f"week4fe_aug_itempop{kp}", "1.0", "1.0")
        rnd = load("week4fe_aug_itemonly", str(kp / 100), "1.0")
        seeds = sorted(set(want) & set(a0) & set(pop) & set(rnd) & set(full))
        if len(seeds) < 2:
            print(f"keep {kp}%: not enough paired seeds (pop {sorted(pop)}, rand {sorted(rnd)})")
            continue
        has = np.load(f"results/stepC/pop{kp}/item_has.npy").astype(bool)
        sel_pop = np.flatnonzero(has)
        sel_rnd = item_order[: math.ceil(kp / 100 * len(item_order))]
        row = {"keep_pct": kp, "n_seeds": len(seeds),
               "n_items_pop": int(has.sum()), "n_items_rand": len(sel_rnd),
               "interaction_share_pop": deg[sel_pop].sum() / deg.sum(),
               "interaction_share_rand": deg[sel_rnd].sum() / deg.sum(),
               "gen_hours_pop": sum(elapsed.get(i, 0.0) for i in sel_pop) / 3600,
               "gen_hours_rand": sum(elapsed.get(i, 0.0) for i in sel_rnd) / 3600}
        for m, mn in ((f"recall@{K}", "recall"), (f"ndcg@{K}", "ndcg")):
            row[f"{mn}_pop"] = float(np.mean([pop[s][m] for s in seeds]))
            row[f"{mn}_rand"] = float(np.mean([rnd[s][m] for s in seeds]))
            for name, src in (("pop", pop), ("rand", rnd), ("full", full)):
                g, lo, hi, p, npos = paired(a0, src, seeds, m)
                row[f"{mn}_gain_{name}"], row[f"{mn}_gain_{name}_lo"], row[f"{mn}_gain_{name}_hi"] = g, lo, hi
                row[f"{mn}_gain_{name}_p"], row[f"{mn}_gain_{name}_npos"] = p, npos
            d, lo, hi, p, npos = paired(rnd, pop, seeds, m)
            row[f"{mn}_pop_minus_rand"], row[f"{mn}_pop_minus_rand_lo"], row[f"{mn}_pop_minus_rand_hi"] = d, lo, hi
            row[f"{mn}_pop_minus_rand_p"], row[f"{mn}_pop_minus_rand_npos"] = p, npos
            row[f"{mn}_retained_pop"] = row[f"{mn}_gain_pop"] / row[f"{mn}_gain_full"]
            row[f"{mn}_retained_rand"] = row[f"{mn}_gain_rand"] / row[f"{mn}_gain_full"]
        rows.append(row)

    if not rows:
        print("no complete results yet")
        return
    df = pd.DataFrame(rows)
    Path("results/stepC").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/stepC/summary.csv", index=False)

    print(f"\n(item side only, alpha=1.0, fixed 25 epochs; full item-only generation = {total_hours_full:.2f} h)")
    for r in rows:
        print(f"\n===== keep top {r['keep_pct']}% of items (n_seeds={r['n_seeds']}) =====")
        print(f"items: pop {r['n_items_pop']} / rand {r['n_items_rand']} | interaction share: pop {r['interaction_share_pop']:.1%} / rand {r['interaction_share_rand']:.1%} | "
              f"generation hours (exact): pop {r['gen_hours_pop']:.2f} / rand {r['gen_hours_rand']:.2f}")
        for mn in ("recall", "ndcg"):
            print(f"[{mn}@20] abs: pop {r[f'{mn}_pop']:.4f} | rand {r[f'{mn}_rand']:.4f}")
            for name in ("pop", "rand", "full"):
                print(f"   gain vs alpha=0, {name:4s}: {r[f'{mn}_gain_{name}']:+.4f} "
                      f"[{r[f'{mn}_gain_{name}_lo']:+.4f}, {r[f'{mn}_gain_{name}_hi']:+.4f}] p={r[f'{mn}_gain_{name}_p']:.4f} "
                      f"({r[f'{mn}_gain_{name}_npos']}/{r['n_seeds']} positive)")
            print(f"   >> pop - rand: {r[f'{mn}_pop_minus_rand']:+.4f} [{r[f'{mn}_pop_minus_rand_lo']:+.4f}, {r[f'{mn}_pop_minus_rand_hi']:+.4f}] "
                  f"p={r[f'{mn}_pop_minus_rand_p']:.4f} ({r[f'{mn}_pop_minus_rand_npos']}/{r['n_seeds']} positive) | "
                  f"share of item-100% gain retained: pop {r[f'{mn}_retained_pop']:.0%} / rand {r[f'{mn}_retained_rand']:.0%}")

    # figure (English labels to avoid CJK font issues, same as plot_cost_benefit.py)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mn, title in ((axes[0], "recall", "Recall@20 gain over alpha=0"), (axes[1], "ndcg", "NDCG@20 gain over alpha=0")):
        for name, c, mk, lab in (("rand", "#1f77b4", "o", "Random order"), ("pop", "#d62728", "s", "Popularity first")):
            xs = [0] + list(df[f"gen_hours_{name}"])
            ys = [0] + list(df[f"{mn}_gain_{name}"])
            lo = [0] + list(df[f"{mn}_gain_{name}"] - df[f"{mn}_gain_{name}_lo"])
            hi = [0] + list(df[f"{mn}_gain_{name}_hi"] - df[f"{mn}_gain_{name}"])
            ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, color=c, capsize=3, label=lab)
            for kp, xv, yv in zip(df["keep_pct"], df[f"gen_hours_{name}"], df[f"{mn}_gain_{name}"]):
                ax.annotate(f"{kp}%", (xv, yv), textcoords="offset points", xytext=(5, -12), fontsize=8, color=c)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("Cumulative generation time of augmented items (hours)")
        ax.set_ylabel("Gain (absolute)")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    n = int(df["n_seeds"].min())
    fig.suptitle(f"Item-side augmentation: popularity-first vs random order (alpha=1.0, {n} seeds, 95% CI)", fontsize=11)
    fig.tight_layout(); fig.savefig("results/stepC/pop_vs_random.png", dpi=160)
    print("\nsaved results/stepC/summary.csv and results/stepC/pop_vs_random.png")


if __name__ == "__main__":
    main()
