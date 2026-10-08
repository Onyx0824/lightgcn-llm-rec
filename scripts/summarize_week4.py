"""
彙整 Week 4 多 seed 結果：各 alpha 的 mean ± std，並對 alpha=1.0 vs alpha=0 做「依 seed 配對」的比較。

用法（專案根目錄）：
    python scripts/summarize_week4.py --ratio 0.2                       # 固定 epoch 版本（week4fe_aug_*）
    python scripts/summarize_week4.py --ratio 0.2 --prefix week4_aug    # early stopping 版本（無分群欄位）
"""
import argparse
import glob
import json
import re

import numpy as np
from scipy import stats


def load(prefix, ratio, alpha):
    out = {}
    for f in glob.glob(f"results/{prefix}_r{ratio}_a{alpha}_s*.json"):
        m = re.search(r"_s(\d+)\.json$", f)
        if m:
            with open(f, encoding="utf-8") as fh:
                out[int(m.group(1))] = json.load(fh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", default="0.2")
    ap.add_argument("--prefix", default="week4fe_aug")
    ap.add_argument("--a0", default="0.0")
    ap.add_argument("--a1", default="1.0")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    r0, r1 = load(args.prefix, args.ratio, args.a0), load(args.prefix, args.ratio, args.a1)
    seeds = sorted(set(r0) & set(r1))
    print(f"prefix={args.prefix} ratio={args.ratio}  配對 seed 數 n={len(seeds)}: {seeds}")
    if len(seeds) < 2:
        print("seed 不足，無法比較")
        return

    for group, label in [("test_metrics", "全體 test user"),
                         ("test_aug_users", "有增強的 test user"),
                         ("test_nonaug_users", "無增強的 test user")]:
        if any(group not in r0[s] or group not in r1[s] for s in seeds):
            continue
        n_users = r1[seeds[0]][group].get("n_users")
        print(f"\n== {label}" + (f"（每次約 {n_users} 位）" if n_users is not None else "") + " ==")
        for metric in (f"recall@{args.k}", f"ndcg@{args.k}"):
            a = np.array([r0[s][group][metric] for s in seeds])
            b = np.array([r1[s][group][metric] for s in seeds])
            d = b - a
            t, p = stats.ttest_rel(b, a)
            print(f"{metric}: alpha={args.a0} {a.mean():.4f}±{a.std(ddof=1):.4f} | "
                  f"alpha={args.a1} {b.mean():.4f}±{b.std(ddof=1):.4f} | "
                  f"配對差 {d.mean():+.4f} (相對 {d.mean() / a.mean() * 100:+.1f}%)  "
                  f"為正 {int((d > 0).sum())}/{len(d)}  paired t={t:.2f} p={p:.3f}")


if __name__ == "__main__":
    main()
