"""Week 5 D group: semantic-outlier pruning of LLM item embeddings (+ random-drop control).

Idea: an item whose LLM-generated embedding is far from the other items of the same
MovieLens genre(s) is a "semantic outlier" (likely off-topic / low-quality generation).
Score = mean cosine(item, leave-one-out centroid of each of its genres); lower = more suspicious.
Prune the lowest q% of scored items (they fall back to ID-only embeddings in LightGCN).

--mode random drops the same number of items at random (control: is the effect from
SELECTION or merely from having fewer augmented items?).

The scoring function lives in src/analysis/pruning.py (moved out of this script; logic unchanged).

Usage (project root):
    python scripts/prune_items.py --mode score  --frac 0.10 --out_dir results/week5/prune10
    python scripts/prune_items.py --mode random --frac 0.20 --out_dir results/week5/rand20 --seed 0
Train with:  --aug_side item --aug_dir <out_dir> --tag <tag>
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.pruning import consistency_scores, genre_list  # noqa: E402


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="results/week4", help="unfiltered embeddings + items.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["score", "random"], required=True)
    ap.add_argument("--frac", type=float, required=True, help="fraction of scorable items to drop, e.g. 0.10")
    ap.add_argument("--seed", type=int, default=0, help="only used by --mode random")
    args = ap.parse_args()

    src, out = Path(args.src_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = np.load(src / "item_emb.npy")
    has = np.load(src / "item_has.npy").astype(bool)
    rows = read_jsonl(src / "items.jsonl")
    genres = [[] for _ in range(len(emb))]
    titles = {}
    for r in rows:
        genres[r["item_idx"]] = genre_list(r.get("genres"))
        titles[r["item_idx"]] = r.get("title", "")

    scores = consistency_scores(emb, has, genres)
    pool = np.flatnonzero(~np.isnan(scores))
    k = int(round(args.frac * len(pool)))
    if args.mode == "score":
        drop = pool[np.argsort(scores[pool])[:k]]          # lowest consistency first
    else:
        drop = np.random.default_rng(args.seed).choice(pool, size=k, replace=False)

    new_has = has.copy()
    new_has[drop] = False

    # train_augmented.py loads all four arrays + order.json from --aug_dir
    for name in ["user_emb.npy", "user_has.npy", "item_emb.npy", "order.json"]:
        shutil.copy(src / name, out / name)
    np.save(out / "item_has.npy", new_has)

    all_g = Counter(g for i in pool for g in genres[i])
    drop_g = Counter(g for i in drop for g in genres[i])
    stats = {
        "mode": args.mode, "frac": args.frac, "seed": args.seed if args.mode == "random" else None,
        "items_with_embedding": int(has.sum()), "scorable_pool": int(len(pool)),
        "dropped": int(len(drop)), "items_usable_after": int(new_has.sum()),
        "score_mean_all": round(float(np.nanmean(scores)), 4),
        "score_mean_dropped": round(float(np.mean(scores[drop])), 4),
        "score_cutoff": round(float(scores[drop].max()), 4) if args.mode == "score" else None,
        "genre_share_dropped_vs_all": {
            g: [round(drop_g[g] / max(len(drop), 1), 3), round(all_g[g] / len(pool), 3)]
            for g, _ in drop_g.most_common(8)
        },
        "examples_dropped": [(int(i), titles.get(int(i), "")) for i in drop[:10]],
    }
    with open(out / "prune_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
