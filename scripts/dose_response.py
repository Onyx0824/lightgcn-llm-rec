"""Week 5 summary: dose-response of item-augmentation pruning.

x = share of TRAIN interactions that fall on items whose LLM augmentation was removed
    (0% = unfiltered item-only, 100% = no augmentation = alpha 0 control)
y = share of the item-only gain (vs. no augmentation) that is retained, per seed, paired:
    retained = (metric_variant - metric_noaug) / mean(metric_itemonly - metric_noaug)

Question answered: do PageRank-selected / semantic-score-selected drops behave differently from
random drops that touch the SAME share of interactions, or is everything just "how many interactions
were touched" (popularity)?

Reads  : results/week4/item_has.npy, results/week5/<dir>/item_has.npy, data/processed/train.csv,
         results/week4fe_aug_<tag>_r1.0_a1.0_s*.json   (variants), results/week4fe_aug_itemonly_r1.0_a1.0_s*.json,
         results/week4fe_aug_r1.0_a0.0_s*.json         (no-augmentation control)
Writes : results/week5/dose_response_<metric>.csv / .png   (metric "ndcg@20" -> ndcg20)

Usage (project root):
    python scripts/dose_response.py
    python scripts/dose_response.py --metric recall@20
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# (folder under results/week5, result-file tag, group label)
VARIANTS = [
    ("prune10", "itemprune10", "semantic score"), ("prune20", "itemprune20", "semantic score"),
    ("rand10", "itemrand10", "random"), ("rand20", "itemrand20", "random"),
    ("rand30", "itemrand30", "random"), ("rand50", "itemrand50", "random"),
    ("rand80", "itemrand80", "random"), ("rand90", "itemrand90", "random"),
    ("prlow30", "itemprlow30", "PageRank low"), ("prlow50", "itemprlow50", "PageRank low"),
    ("prhigh30", "itemprhigh30", "PageRank high"), ("prhigh50", "itemprhigh50", "PageRank high"),
]
STYLE = {"random": ("tab:gray", "o"), "semantic score": ("tab:green", "s"),
         "PageRank low": ("tab:blue", "^"), "PageRank high": ("tab:red", "v")}


def load_runs(prefix, metric):
    out = {}
    for f in glob.glob(f"results/{prefix}_s*.json"):
        m = re.search(r"_s(\d+)\.json$", f)
        if m:
            with open(f, encoding="utf-8") as fh:
                out[int(m.group(1))] = json.load(fh)["test_metrics"][metric]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--metric", default="ndcg@20")
    ap.add_argument("--out_dir", default="results/week5")
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    has0 = np.load("results/week4/item_has.npy").astype(bool)
    deg = np.bincount(df["item"].to_numpy(), minlength=len(has0)).astype(float)

    item_only = load_runs("week4fe_aug_itemonly_r1.0_a1.0", args.metric)
    no_aug = load_runs("week4fe_aug_r1.0_a0.0", args.metric)
    print(f"item-only seeds: {sorted(item_only)} | no-aug seeds: {sorted(no_aug)}")

    rows = []
    for folder, tag, group in VARIANTS:
        mask_path = Path("results/week5") / folder / "item_has.npy"
        res = load_runs(f"week4fe_aug_{tag}_r1.0_a1.0", args.metric)
        if not mask_path.exists() or not res:
            print(f"skip {folder}: mask exists={mask_path.exists()}, result files={len(res)}")
            continue
        has = np.load(mask_path).astype(bool)
        dropped = has0 & ~has
        share = deg[dropped].sum() / deg[has0].sum()
        seeds = sorted(set(res) & set(item_only) & set(no_aug))
        if len(seeds) < 2:
            print(f"skip {folder}: only {len(seeds)} paired seeds")
            continue
        gain = np.mean([item_only[s] - no_aug[s] for s in seeds])
        retained = np.array([(res[s] - no_aug[s]) / gain for s in seeds])
        rows.append({
            "variant": folder, "group": group, "n_dropped": int(dropped.sum()),
            "share_interactions_dropped": round(float(share), 4), "n_seeds": len(seeds),
            f"{args.metric}_mean": round(float(np.mean([res[s] for s in seeds])), 5),
            "retained_gain_mean": round(float(retained.mean()), 4),
            "retained_gain_se": round(float(retained.std(ddof=1) / np.sqrt(len(seeds))), 4),
        })
    if not rows:
        print("no variants found - run the training scripts first")
        return
    tab = pd.DataFrame(rows).sort_values(["share_interactions_dropped"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "dose_response_" + args.metric.replace("@", "")
    tab.to_csv(out / f"{stem}.csv", index=False)
    print("\n", tab.to_string(index=False))

    # random-drop reference curve: the random points connected in order of share (NOT a linear fit:
    # the dose-response is saturating, so a straight line is misleading). Outside the range of the random
    # points the curve is clamped to the nearest random point. The SE below ignores the noise of the curve itself.
    rnd = tab[tab["group"] == "random"].sort_values("share_interactions_dropped")
    ref = None
    if len(rnd) >= 3:
        rx = rnd["share_interactions_dropped"].to_numpy()
        ry = rnd["retained_gain_mean"].to_numpy()
        ref = (rx, ry)
        print(f"\nresidual vs random-drop curve (random points {rx.min():.0%}-{rx.max():.0%} of interactions, clamped outside):")
        for _, r in tab[tab["group"] != "random"].iterrows():
            pred = float(np.interp(r["share_interactions_dropped"], rx, ry))
            print(f"  {r['variant']:9s} share {r['share_interactions_dropped']:.0%}: observed {r['retained_gain_mean']:+.3f} vs random-curve {pred:+.3f} "
                  f"-> residual {r['retained_gain_mean'] - pred:+.3f} (SE {r['retained_gain_se']:.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipped figure (pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.linspace(0, 1, 50)
    ax.plot(xs * 100, (1 - xs) * 100, "k--", lw=1, label="gain proportional to interactions kept")
    if ref:
        ax.plot(ref[0] * 100, ref[1] * 100, color="tab:gray", lw=1, alpha=.6, label="random-drop curve")
    for g, (c, m) in STYLE.items():
        s = tab[tab["group"] == g]
        if len(s):
            ax.errorbar(s["share_interactions_dropped"] * 100, s["retained_gain_mean"] * 100,
                        yerr=s["retained_gain_se"] * 100, fmt=m, color=c, capsize=3, label=g)
            for _, r in s.iterrows():
                ax.annotate(r["variant"], (r["share_interactions_dropped"] * 100, r["retained_gain_mean"] * 100),
                            textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.axhline(100, color="k", lw=.5)
    ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel("train interactions on items whose augmentation was removed (%)")
    ax.set_ylabel(f"item-only gain retained (%), {args.metric}")
    ax.set_title("Dose-response of item-augmentation pruning (mean +/- SE over seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=160)
    print(f"saved {out}/{stem}.csv and {stem}.png")


if __name__ == "__main__":
    main()
