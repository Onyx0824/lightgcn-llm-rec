"""Week 7: error analysis - pick candidate cases and attach the evidence you need to classify them by hand.

Per-user score = hits / min(n_test, 20) ("normalised hit rate", max 1), averaged over all paired seeds,
so single-seed noise does not decide which users are shown.
Groups: HURT (augmentation lowers the score most), HELPED (raises most), FAIL (lowest score under augmentation,
users with >= 20 test positives).
Each case lists: train/test size, the user's top train genres, the test items that were hit, the top-10 recommended
titles (with genres, train popularity, and [D]=dirty / [R]=needed retry flags from the LLM generation),
and the LLM tags of the first 3 recommended items (to eyeball hallucination).
Automatic tags are HINTS for your manual classification, not conclusions:
  sparse-user      user train degree below the 25th percentile of test users   -> data sparsity
  long-tail-user   >= 50% of the user's test items are below the median item degree -> sparsity / cold items
  genre-drift      < 50% of top-10 recs share a genre with the user's top-3 train genres -> semantic drift
  llm-flagged      a top-10 rec has dirty / retry>1 generation                 -> LLM noise (check mood tags)
  hot-bias         mean degree of recs > 2x mean degree of the user's test items -> popularity bias
Usage (project root):
    python scripts/error_analysis.py            # 5 cases per group -> results/week7/error_cases.md
    python scripts/error_analysis.py --n_show 8
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.stratify import hit_vector, load_top, user_hit_rate  # noqa: E402

K = 20


def user_scores(top, tu, ti, n_users, n_pos):
    return user_hit_rate(hit_vector(top, tu, ti), tu, n_users, n_pos, K)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", default="1.0")
    ap.add_argument("--ctrl_prefix", default="week4fe_aug_w7")
    ap.add_argument("--ctrl_alpha", default="0.0")
    ap.add_argument("--treat_prefix", default="week4fe_aug_w7item")
    ap.add_argument("--treat_alpha", default="1.0")
    ap.add_argument("--emb_dir", default="results/week7/emb")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--raw_dir", default="data/raw/ml-1m")
    ap.add_argument("--gen_dir", default="results/week4")
    ap.add_argument("--out", default="results/week7/error_cases.md")
    ap.add_argument("--n_show", type=int, default=5)
    args = ap.parse_args()

    dd = Path(args.data_dir)
    train, test = pd.read_csv(dd / "train.csv"), pd.read_csv(dd / "test.csv")
    with open(dd / "id_maps.json", encoding="utf-8") as f:
        im = json.load(f)
    n_users, n_items = len(im["user2idx"]), len(im["item2idx"])
    idx2raw = {v: int(k) for k, v in im["item2idx"].items()}
    movies = pd.read_csv(Path(args.raw_dir) / "movies.dat", sep="::", engine="python",
                         names=["item_raw", "title", "genres"], encoding="latin-1").set_index("item_raw")
    title = {i: movies.loc[idx2raw[i], "title"] if idx2raw[i] in movies.index else "?" for i in range(n_items)}
    genres = {i: set(movies.loc[idx2raw[i], "genres"].split("|")) if idx2raw[i] in movies.index else set()
              for i in range(n_items)}
    flags, llm = {}, {}
    ipath = Path(args.gen_dir) / "items.jsonl"
    if ipath.exists():
        for r in read_jsonl(ipath):
            f = ("D" if r.get("dirty") else "") + ("R" if r.get("attempts", 1) > 1 else "")
            flags[r["item_idx"]] = f
            llm[r["item_idx"]] = r.get("parsed") or {}

    tu, ti = test["user"].to_numpy(), test["item"].to_numpy()
    n_pos = np.bincount(tu, minlength=n_users)
    deg_i = np.bincount(train["item"].to_numpy(), minlength=n_items)
    deg_u = np.bincount(train["user"].to_numpy(), minlength=n_users)
    item_median = np.median(deg_i[deg_i > 0])
    test_users = np.flatnonzero(n_pos > 0)
    sparse_cut = np.percentile(deg_u[test_users], 25)

    ctrl = load_top(args.emb_dir, args.ctrl_prefix, args.ratio, args.ctrl_alpha)
    treat = load_top(args.emb_dir, args.treat_prefix, args.ratio, args.treat_alpha)
    seeds = sorted(set(ctrl) & set(treat))
    if not seeds:
        raise SystemExit("no paired seeds found (run scripts/run_week7.ps1 first)")
    sc = np.mean([user_scores(ctrl[s], tu, ti, n_users, n_pos) for s in seeds], axis=0)
    sa = np.mean([user_scores(treat[s], tu, ti, n_users, n_pos) for s in seeds], axis=0)
    diff = sa - sc
    show_seed = seeds[0]
    top_show = treat[show_seed]

    user_train = train.groupby("user")["item"].apply(list).to_dict()
    user_test = test.groupby("user")["item"].apply(list).to_dict()

    def describe(u):
        gc = Counter(g for it in user_train.get(u, []) for g in genres[it])
        top3 = [g for g, _ in gc.most_common(3)]
        recs = top_show[u][:10].tolist()
        match = np.mean([bool(genres[r] & set(top3)) for r in recs])
        t_items = user_test.get(u, [])
        hit_items = [it for it in t_items if it in set(top_show[u].tolist())]
        tags = []
        if deg_u[u] < sparse_cut:
            tags.append("sparse-user")
        if t_items and np.mean([deg_i[it] < item_median for it in t_items]) >= 0.5:
            tags.append("long-tail-user")
        if match < 0.5:
            tags.append("genre-drift")
        if any(flags.get(r) for r in recs):
            tags.append("llm-flagged")
        if t_items and np.mean(deg_i[recs]) > 2 * np.mean(deg_i[t_items]):
            tags.append("hot-bias")
        lines = [f"- user_idx **{u}** | train {deg_u[u]} / test {n_pos[u]} | score ctrl {sc[u]:.2f} -> aug {sa[u]:.2f} "
                 f"(diff {diff[u]:+.2f}) | auto tags: {', '.join(tags) or '-'}",
                 f"  - train top genres: {', '.join(top3) or '-'}; top-10 recs sharing them: {match:.0%}",
                 f"  - test items hit by aug top-20 (seed {show_seed}): "
                 + ("; ".join(f"{title[i]} (deg {deg_i[i]})" for i in hit_items[:5]) or "none"),
                 "  - aug top-10 recs:"]
        for r in recs:
            lines.append(f"    - {title[r]} [{'|'.join(sorted(genres[r]))}] deg={deg_i[r]} {('[' + flags[r] + ']') if flags.get(r) else ''}")
        for r in recs[:3]:
            p = llm.get(r, {})
            if p:
                lines.append(f"  - LLM tags of '{title[r]}': mood={p.get('mood_tags')}, "
                             f"supp={p.get('supplementary_genres')}, summary={str(p.get('profile_summary'))[:60]}")
        return "\n".join(lines)

    cand = test_users
    hurt = cand[np.argsort(diff[cand])][:args.n_show]
    helped = cand[np.argsort(-diff[cand])][:args.n_show]
    big = cand[n_pos[cand] >= K]
    fail = big[np.argsort(sa[big])][:args.n_show]

    md = [f"# Error analysis candidates (paired seeds {seeds}; per-user score = hits / min(n_test, {K}))", "",
          f"Overall mean score: ctrl {sc[test_users].mean():.4f}, aug {sa[test_users].mean():.4f}; "
          f"users hurt {int((diff[test_users] < 0).sum())}, helped {int((diff[test_users] > 0).sum())}, "
          f"unchanged {int((diff[test_users] == 0).sum())} (of {len(test_users)}).", "",
          "Pick 5-10 cases overall, then assign each ONE main cause by reading the evidence: "
          "data sparsity / cold-start-like user / LLM hallucination or drift / popularity bias / other. "
          "The auto tags below are only hints.", "",
          "## HURT - augmentation lowers the score the most", ""]
    md += [describe(u) + "\n" for u in hurt]
    md += ["## HELPED - augmentation raises the score the most", ""] + [describe(u) + "\n" for u in helped]
    md += [f"## FAIL - lowest score under augmentation (users with >= {K} test items)", ""] + [describe(u) + "\n" for u in fail]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {outp}  (users hurt {int((diff[test_users] < 0).sum())}, helped {int((diff[test_users] > 0).sum())})")


if __name__ == "__main__":
    main()
