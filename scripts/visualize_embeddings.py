"""Week 7: item embedding visualisation (PCA / t-SNE) + quantitative genre-cohesion check.

Three spaces are compared: raw LLM text embedding (bge, 512-d), final LightGCN item embedding with alpha=0,
final LightGCN item embedding with item-side augmentation. Plots show seed `--plot_seed`; the numbers
(kNN genre purity, silhouette by primary genre, Spearman(embedding norm, popularity)) use ALL paired seeds,
because t-SNE pictures alone are easy to over-read (axes are not comparable across runs).
Items are labelled by primary genre = first genre listed in MovieLens; only the 8 most frequent genres are used.

Needs results/week7/emb/*_item.npy (train_augmented.py --save_emb) and results/week4/item_emb.npy, items.jsonl.
Usage (project root):
    python scripts/visualize_embeddings.py
Outputs: results/week7/embeddings_pca.png, embeddings_tsne.png, embedding_cohesion.md
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.embedding_metrics import knn_purity, l2  # noqa: E402


def load_emb(emb_dir, prefix, ratio, alpha):
    out = {}
    for f in glob.glob(str(Path(emb_dir) / f"{prefix}_r{ratio}_a{alpha}_s*_item.npy")):
        m = re.search(r"_s(\d+)_item\.npy$", f)
        if m:
            out[int(m.group(1))] = np.load(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", default="1.0")
    ap.add_argument("--ctrl_prefix", default="week4fe_aug_w7")
    ap.add_argument("--ctrl_alpha", default="0.0")
    ap.add_argument("--treat_prefix", default="week4fe_aug_w7item")
    ap.add_argument("--treat_alpha", default="1.0")
    ap.add_argument("--emb_dir", default="results/week7/emb")
    ap.add_argument("--gen_dir", default="results/week4")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--out_dir", default="results/week7")
    ap.add_argument("--plot_seed", type=int, default=0)
    ap.add_argument("--top_genres", type=int, default=8)
    args = ap.parse_args()

    ctrl = load_emb(args.emb_dir, args.ctrl_prefix, args.ratio, args.ctrl_alpha)
    treat = load_emb(args.emb_dir, args.treat_prefix, args.ratio, args.treat_alpha)
    seeds = sorted(set(ctrl) & set(treat))
    if not seeds:
        raise SystemExit("no paired item embeddings found (run scripts/run_week7.ps1 first)")
    plot_seed = args.plot_seed if args.plot_seed in seeds else seeds[0]
    n_items = ctrl[seeds[0]].shape[0]

    llm = np.load(Path(args.gen_dir) / "item_emb.npy")
    has = np.load(Path(args.gen_dir) / "item_has.npy").astype(bool)
    primary = {}
    with open(Path(args.gen_dir) / "items.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                g = [x for x in str(r.get("genres", "")).split("|") if x]
                if g:
                    primary[r["item_idx"]] = g[0]
    top = [g for g, _ in Counter(primary.values()).most_common(args.top_genres)]
    ids = np.array([i for i in range(n_items) if primary.get(i) in top and has[i]])
    lab_str = np.array([primary[i] for i in ids])
    lab = np.array([top.index(g) for g in lab_str])
    deg = np.bincount(pd.read_csv(args.train)["item"].to_numpy(), minlength=n_items)

    # ---- numbers over all paired seeds ----
    def stats_for(x):
        xs = x[ids]
        return (knn_purity(xs, lab), float(silhouette_score(l2(xs), lab, metric="cosine")),
                float(stats.spearmanr(np.linalg.norm(x[ids], axis=1), deg[ids])[0]))

    rows = {"LLM text embedding (raw, 512-d)": [stats_for(llm)]}
    pur_c, pur_a = [], []
    ctrl_stats, treat_stats = [], []
    for s in seeds:
        c, a = stats_for(ctrl[s]), stats_for(treat[s])
        ctrl_stats.append(c)
        treat_stats.append(a)
        pur_c.append(c[0])
        pur_a.append(a[0])
    rows["LightGCN alpha=0"] = ctrl_stats
    rows["LightGCN item-only augmented"] = treat_stats

    md = [f"## Item embedding genre cohesion ({len(ids)} items with a top-{args.top_genres} primary genre; "
          f"seeds {seeds})", "",
          "| space | kNN genre purity (k=10) | silhouette (cosine, by genre) | Spearman(norm, train degree) |",
          "|---|---|---|---|"]
    for name, vals in rows.items():
        v = np.array(vals)
        sd = v.std(axis=0, ddof=1) if len(v) > 1 else np.zeros(3)
        md.append(f"| {name} | {v[:, 0].mean():.3f} +/- {sd[0]:.3f} | {v[:, 1].mean():.3f} +/- {sd[1]:.3f} | "
                  f"{v[:, 2].mean():.3f} +/- {sd[2]:.3f} |")
    if len(seeds) > 1:
        d = np.array(pur_a) - np.array(pur_c)
        md += ["", f"Paired kNN-purity difference (aug - alpha0): {d.mean():+.4f}, positive in "
               f"{int((d > 0).sum())}/{len(d)} seeds, paired t p={stats.ttest_rel(pur_a, pur_c).pvalue:.4f} (exploratory)."]
    md += ["", "> Higher purity / silhouette = items of the same genre sit closer together. Spearman(norm, degree) "
           "shows how strongly embedding length encodes popularity. Raw LLM text embedding is a reference, not a competitor."]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "embedding_cohesion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---- figures (plot_seed only) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: skipped figures")
        return
    spaces = [("LLM text emb (bge)", llm[ids]), ("LightGCN alpha=0", ctrl[plot_seed][ids]),
              ("LightGCN + item aug", treat[plot_seed][ids])]
    for method in ("pca", "tsne"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for j, (name, x) in enumerate(spaces):
            if method == "pca":
                z = PCA(n_components=2, random_state=0).fit_transform(x)
            else:
                z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(l2(x))
            ax = axes[0, j]
            for g, gname in enumerate(top):
                m = lab == g
                ax.scatter(z[m, 0], z[m, 1], s=6, alpha=0.7, label=gname)
            ax.set_title(f"{name}\nby primary genre", fontsize=10)
            ax2 = axes[1, j]
            sc = ax2.scatter(z[:, 0], z[:, 1], s=6, c=np.log1p(deg[ids]), cmap="viridis")
            ax2.set_title(f"{name}\nby log(1+train degree)", fontsize=10)
            for a in (ax, ax2):
                a.set_xticks([])
                a.set_yticks([])
        axes[0, 0].legend(fontsize=7, markerscale=2, loc="best")
        fig.colorbar(sc, ax=axes[1, :], shrink=0.6, label="log(1+train degree)")
        fig.suptitle(f"Item embeddings ({method.upper()}, seed {plot_seed}); axes are not comparable across panels", fontsize=11)
        fig.savefig(out / f"embeddings_{method}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}/embeddings_{method}.png")


if __name__ == "__main__":
    main()
