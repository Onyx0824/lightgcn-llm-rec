"""Step B summary.
  python scripts/stepB_summary.py --seeds 0,1,2,3,4        # B1: early stopping vs fixed-25 protocol
  python scripts/stepB_summary.py --tune_only              # B2 stage 1: validation table of tuning variants
  python scripts/stepB_summary.py --confirm lr3e-3 --seeds 0,1,2,3,4   # B2 stage 2: LLM item-only vs alpha=0 under the tuned config
"""
import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

K = 20
M = {"recall": f"recall@{K}", "ndcg": f"ndcg@{K}"}


def load_dir(pattern):
    out = {}
    for f in glob.glob(pattern):
        m = re.search(r"_s(\d+)\.json$", f)
        if m:
            out[int(m.group(1))] = json.load(open(f, encoding="utf-8"))
    return out


def paired(a, b, seeds, key):
    """b - a: mean, lo, hi, p, n_pos"""
    d = np.array([b[s]["test_metrics"][key] - a[s]["test_metrics"][key] for s in seeds])
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = stats.ttest_rel([b[s]["test_metrics"][key] for s in seeds], [a[s]["test_metrics"][key] for s in seeds]).pvalue
    return d.mean(), d.mean() - h, d.mean() + h, p, int((d > 0).sum())


def fmt(t):
    m, lo, hi, p, npos = t
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} ({npos} positive)"


def b1(seeds_arg):
    es = {c: load_dir(f"results/stepB/es/{pat}_s*.json") for c, pat in
          (("alpha0", "week4_aug_r1.0_a0.0"), ("llm", "week4_aug_itemonly_r1.0_a1.0"), ("meta", "week4_aug_itemmeta_r1.0_a1.0"))}
    fx = {c: load_dir(f"results/{pat}_s*.json") for c, pat in
          (("alpha0", "week4fe_aug_r1.0_a0.0"), ("llm", "week4fe_aug_itemonly_r1.0_a1.0"), ("meta", "week4fe_aug_itemmeta_r1.0_a1.0"))}
    have = [c for c in ("alpha0", "llm", "meta") if es[c]]
    seeds = sorted(set(seeds_arg) & set.intersection(*[set(es[c]) for c in have]) & set.intersection(*[set(fx[c]) for c in have]))
    print(f"\n=== B1: early stopping (validation Recall, patience 5 evals) vs fixed 25 epochs | conditions {have} | paired seeds {seeds}")
    if len(seeds) < 2:
        print("not enough paired seeds yet"); return
    for c in have:
        be = [es[c][s]["best_epoch"] for s in seeds]
        print(f"[{c:6s}] ES: recall {np.mean([es[c][s]['test_metrics'][M['recall']] for s in seeds]):.4f}"
              f"+-{np.std([es[c][s]['test_metrics'][M['recall']] for s in seeds], ddof=1):.4f}  "
              f"ndcg {np.mean([es[c][s]['test_metrics'][M['ndcg']] for s in seeds]):.4f}"
              f"+-{np.std([es[c][s]['test_metrics'][M['ndcg']] for s in seeds], ddof=1):.4f} | best epochs {be}"
              f"   || fixed25: recall {np.mean([fx[c][s]['test_metrics'][M['recall']] for s in seeds]):.4f}"
              f" ndcg {np.mean([fx[c][s]['test_metrics'][M['ndcg']] for s in seeds]):.4f}")
    pairs = [("llm", "alpha0"), ("meta", "alpha0"), ("llm", "meta")]
    for a, b in pairs:
        if a in have and b in have:
            print(f"\n--- {a} minus {b} ---")
            for mn, key in M.items():
                print(f"  {mn:6s} early-stop: {fmt(paired(es[b], es[a], seeds, key))}")
                print(f"  {mn:6s} fixed-25  : {fmt(paired(fx[b], fx[a], seeds, key))}")
    print("\nHow to read: if the SIGNS and significance of 'llm - alpha0' and 'llm - meta' match between the two protocols, the conclusions do not depend on the stopping rule.")


def tune_table():
    base = load_dir("results/stepB/es/week4_aug_r1.0_a0.0_s*.json")
    rows = []
    for d in sorted(glob.glob("results/stepB/tune/*")):
        r = load_dir(f"{d}/week4_aug_r1.0_a0.0_s*.json")
        if r:
            rows.append((Path(d).name, r))
    seeds = sorted(set.intersection(*[set(r) for _, r in rows])) if rows else []
    if base:
        seeds = sorted(set(seeds) & set(base)) if seeds else []
    print(f"\n=== B2 stage 1: baseline tuning screen (alpha=0, early stopping), VALIDATION Recall@20, seeds {seeds} ===")
    if not seeds:
        print("no complete results yet"); return
    dflt = [base[s]["best_valid_recall"] for s in seeds] if base else None
    if dflt:
        print(f"default (lr1e-3, wd1e-4, 3 layers, dim64): valid recall {np.mean(dflt):.4f} (per seed {[round(x,4) for x in dflt]}), best epochs {[base[s]['best_epoch'] for s in seeds]}")
    for name, r in rows:
        v = [r[s]["best_valid_recall"] for s in seeds]
        flag = ""
        if dflt and np.mean(v) >= np.mean(dflt) + 0.001 and min(v) > max(dflt):
            flag = "   <-- meets the pre-set rule (mean >= default + 0.001 AND every seed above the default's best seed)"
        print(f"{name:8s}: valid recall {np.mean(v):.4f} (per seed {[round(x,4) for x in v]}), best epochs {[r[s]['best_epoch'] for s in seeds]}, train sec {np.mean([r[s]['train_sec'] for s in seeds]):.0f}{flag}")
    print("\nRule fixed in advance: adopt a variant only if it meets the flagged rule; otherwise report that the default is not clearly under-tuned. Test metrics are NOT used for selection.")


def confirm(name, seeds_arg):
    d = f"results/stepB/tune/{name}"
    a0, ll = load_dir(f"{d}/week4_aug_r1.0_a0.0_s*.json"), load_dir(f"{d}/week4_aug_itemonly_r1.0_a1.0_s*.json")
    seeds = sorted(set(seeds_arg) & set(a0) & set(ll))
    print(f"\n=== B2 stage 2: tuned config '{name}', early stopping, seeds {seeds} ===")
    if len(seeds) < 2:
        print("not enough paired seeds"); return
    for mn, key in M.items():
        print(f"  LLM item-only - alpha0 ({mn}): {fmt(paired(a0, ll, seeds, key))}")
        print(f"  abs: alpha0 {np.mean([a0[s]['test_metrics'][key] for s in seeds]):.4f} | llm {np.mean([ll[s]['test_metrics'][key] for s in seeds]):.4f}")
    es0 = load_dir("results/stepB/es/week4_aug_r1.0_a0.0_s*.json")
    ss = [s for s in seeds if s in es0]
    if ss:
        print(f"  (default-config alpha0, same seeds, early stopping: recall {np.mean([es0[s]['test_metrics'][M['recall']] for s in ss]):.4f}, ndcg {np.mean([es0[s]['test_metrics'][M['ndcg']] for s in ss]):.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tune_only", action="store_true")
    ap.add_argument("--confirm", default="")
    a = ap.parse_args()
    sd = [int(x) for x in a.seeds.split(",")]
    if a.tune_only:
        tune_table()
    elif a.confirm:
        confirm(a.confirm, sd)
    else:
        b1(sd)
