"""Week 7: stratified analysis of LLM augmentation (alpha=0 control vs augmented), paired over seeds.

--by item : tiers = item popularity (train degree quartiles). Metric per tier = micro Recall@20
            (share of test positives in that tier that appear in the user's top-20), plus
            how the top-20 recommendation slots and catalogue coverage are distributed over tiers.
--by user : tiers = user activity (train degree quartiles among users that have test positives).
            Metric per tier = mean per-user Recall@20.

Needs the top-20 files written by `train_augmented.py --save_emb` (results/week7/emb/*_top20.npy).
Usage (project root):
    python scripts/popularity_strata.py --by item
    python scripts/popularity_strata.py --by user
    python scripts/popularity_strata.py --by item --n_tiers 3
Outputs: results/week7/strata_<by>.md / .csv / .png
p-values are exploratory (uncorrected); they only reflect training randomness (test users are fixed).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.stats_utils import paired_ci  # noqa: E402
from src.analysis.stratify import hit_vector, load_top, quantile_tiers, tier_labels, user_recall  # noqa: E402


def fmt_rel(x):
    return "n/a" if x is None or np.isnan(x) else f"{x:+.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["item", "user"], default="item")
    ap.add_argument("--n_tiers", type=int, default=4)
    ap.add_argument("--ratio", default="1.0")
    ap.add_argument("--ctrl_prefix", default="week4fe_aug_w7")
    ap.add_argument("--ctrl_alpha", default="0.0")
    ap.add_argument("--treat_prefix", default="week4fe_aug_w7item")
    ap.add_argument("--treat_alpha", default="1.0")
    ap.add_argument("--emb_dir", default="results/week7/emb")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="results/week7")
    args = ap.parse_args()

    dd = Path(args.data_dir)
    train = pd.read_csv(dd / "train.csv")
    test = pd.read_csv(dd / "test.csv")
    with open(dd / "id_maps.json", encoding="utf-8") as f:
        im = json.load(f)
    n_users, n_items = len(im["user2idx"]), len(im["item2idx"])
    tu, ti = test["user"].to_numpy(), test["item"].to_numpy()

    ctrl = load_top(args.emb_dir, args.ctrl_prefix, args.ratio, args.ctrl_alpha)
    treat = load_top(args.emb_dir, args.treat_prefix, args.ratio, args.treat_alpha)
    seeds = sorted(set(ctrl) & set(treat))
    print(f"paired seeds: {seeds}")
    if len(seeds) < 2:
        raise SystemExit("need >= 2 paired seeds (run scripts/run_week7.ps1 first)")

    deg_i = np.bincount(train["item"].to_numpy(), minlength=n_items)
    deg_u = np.bincount(train["user"].to_numpy(), minlength=n_users)
    nt = args.n_tiers
    labels = tier_labels(nt)

    if args.by == "item":
        pool = np.flatnonzero(deg_i > 0)
        tier_of = quantile_tiers(deg_i, pool, nt, n_items)
        row_tier = tier_of[ti]  # tier of each test positive

        def metric(top):
            hit = hit_vector(top, tu, ti)
            return np.array([hit[row_tier == k].mean() if (row_tier == k).any() else np.nan for k in range(nt)])

        n_unit = [int((tier_of == k).sum()) for k in range(nt)]
        share_train = [float(deg_i[tier_of == k].sum() / deg_i.sum()) for k in range(nt)]
        n_test = [int((row_tier == k).sum()) for k in range(nt)]
        unit_name, metric_name = "items", "Recall@20 (micro, per tier)"
    else:
        users = np.unique(tu)
        tier_of = quantile_tiers(deg_u, users, nt, n_users)
        n_pos = np.bincount(tu, minlength=n_users)

        def metric(top):
            rec = user_recall(hit_vector(top, tu, ti), tu, n_users, n_pos)
            return np.array([rec[tier_of == k].mean() for k in range(nt)])

        n_unit = [int((tier_of == k).sum()) for k in range(nt)]
        share_train = [float(deg_u[tier_of == k].sum() / deg_u[users].sum()) for k in range(nt)]
        n_test = [int(n_pos[tier_of == k].sum()) for k in range(nt)]
        unit_name, metric_name = "users", "mean per-user Recall@20"

    C = np.array([metric(ctrl[s]) for s in seeds])  # (seeds, tiers)
    A = np.array([metric(treat[s]) for s in seeds])

    unit_key = f"n_{unit_name}"
    rows = []
    for k in range(nt):
        d, lo, hi, p = paired_ci(C[:, k], A[:, k])
        base = C[:, k].mean()
        rows.append({
            "tier": labels[k], unit_key: n_unit[k], "share_of_train_interactions": round(share_train[k], 4),
            "n_test_positives": n_test[k],
            "ctrl": round(base, 5), "aug": round(A[:, k].mean(), 5),
            "diff": round(d, 5), "ci_lo": round(lo, 5), "ci_hi": round(hi, 5),
            "rel_pct": round(d / base * 100, 2) if base else np.nan,
            "p": round(p, 4) if not np.isnan(p) else np.nan,
            "n_pos_seeds": int(((A[:, k] - C[:, k]) > 0).sum()), "n_seeds": len(seeds),
        })
    tab = pd.DataFrame(rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / f"strata_{args.by}.csv", index=False)

    md = [f"## Stratified by {args.by} ({'item popularity' if args.by == 'item' else 'user activity'}; "
          f"{nt} tiers by train degree, T1 = lowest) - {metric_name}",
          "", f"alpha={args.ctrl_alpha} ({args.ctrl_prefix}) vs alpha={args.treat_alpha} ({args.treat_prefix}), "
          f"ratio {args.ratio}, paired over seeds {seeds}. p-values uncorrected (exploratory).", "",
          f"| tier | #{unit_name} | share of train interactions | #test positives | ctrl | aug | diff [95% CI] | relative | p | positive seeds |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tier']} | {r[unit_key]} | {r['share_of_train_interactions']:.1%} | "
                  f"{r['n_test_positives']} | {r['ctrl']:.4f} | {r['aug']:.4f} | "
                  f"{r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {fmt_rel(r['rel_pct'])} | {r['p']} | "
                  f"{r['n_pos_seeds']}/{r['n_seeds']} |")

    if args.by == "item":
        # where do the 20 recommendation slots go, and how many distinct items are ever recommended?
        tu_u = np.unique(tu)

        def exposure(top):
            rec = top[tu_u].ravel()
            t = tier_of[rec]
            return np.array([(t == k).mean() for k in range(nt)] + [len(np.unique(rec)) / len(pool)])

        EC = np.array([exposure(ctrl[s]) for s in seeds])
        EA = np.array([exposure(treat[s]) for s in seeds])
        md += ["", "### Where the top-20 slots go (test users) and catalogue coverage", "",
               "| tier | slot share ctrl | slot share aug | diff [95% CI] | p |", "|---|---|---|---|---|"]
        for k in range(nt):
            d, lo, hi, p = paired_ci(EC[:, k], EA[:, k])
            md.append(f"| {labels[k]} | {EC[:, k].mean():.2%} | {EA[:, k].mean():.2%} | "
                      f"{d:+.2%} [{lo:+.2%}, {hi:+.2%}] | {p:.4f} |")
        d, lo, hi, p = paired_ci(EC[:, nt], EA[:, nt])
        md.append(f"| catalogue coverage (distinct items recommended / items) | {EC[:, nt].mean():.2%} | "
                  f"{EA[:, nt].mean():.2%} | {d:+.2%} [{lo:+.2%}, {hi:+.2%}] | {p:.4f} |")

    md += ["", "> Reading guide: a positive diff concentrated in T1/T2 means augmentation mostly helps long-tail "
           "(cold / sparse) cases; flat or negative diff in the hottest tier means popular items were already "
           "well served by the interaction graph."]
    (out / f"strata_{args.by}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: skipped figure")
        return
    x = np.arange(nt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    d_ = tab["diff"].to_numpy()
    lo_, hi_ = tab["ci_lo"].to_numpy(), tab["ci_hi"].to_numpy()
    base_ = tab["ctrl"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(base_ > 0, d_ / base_ * 100, np.nan)
        rel_err = [np.where(base_ > 0, (d_ - lo_) / base_ * 100, 0), np.where(base_ > 0, (hi_ - d_) / base_ * 100, 0)]
    for ax, v, err, title in [(axes[0], d_, [d_ - lo_, hi_ - d_], "Absolute gain (aug - ctrl)"),
                              (axes[1], np.nan_to_num(rel), rel_err, "Relative gain (%) (n/a where ctrl = 0)")]:
        ax.bar(x, v, yerr=err, capsize=3, color="#1f77b4")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{k + 1}" for k in range(nt)])
        ax.set_xlabel(f"{args.by} tier (T1 = least {'popular' if args.by == 'item' else 'active'})")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Augmentation gain by {args.by} tier ({metric_name}; {len(seeds)} seeds, 95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"strata_{args.by}.png", dpi=160)
    print(f"saved {out}/strata_{args.by}.md/.csv/.png")


if __name__ == "__main__":
    main()
