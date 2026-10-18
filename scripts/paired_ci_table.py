"""Paired differences with 95% CI and p-values for all reported comparisons.
Usage (project root):  python scripts/paired_ci_table.py
Writes results/paired_ci_table.md (UTF-8). Two-sided paired t-test across seeds; CI = t-based 95% CI of the mean paired difference.
"""
import glob
import json
import numpy as np
from scipy import stats


def load(prefix, ratio, alpha):
    out = {}
    for f in glob.glob(f"results/{prefix}_r{ratio}_a{alpha}_s*.json"):
        s = int(f.rsplit("_s", 1)[1].split(".")[0])
        m = json.load(open(f, encoding="utf-8"))["test_metrics"]
        out[s] = (m["recall@20"], m["ndcg@20"])
    return out


def cell(treat, ctrl, j):
    seeds = sorted(set(treat) & set(ctrl))
    n = len(seeds)
    a = np.array([treat[s][j] for s in seeds])
    b = np.array([ctrl[s][j] for s in seeds])
    d = a - b
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_rel(a, b).pvalue
    return n, f"{d.mean():+.4f} [{d.mean()-h:+.4f}, {d.mean()+h:+.4f}]", f"{p:.3f}" if p >= 0.001 else "<0.001", int((d > 0).sum())


def row(label, treat, ctrl):
    n, r, pr, posr = cell(treat, ctrl, 0)
    _, g, pg, posg = cell(treat, ctrl, 1)
    return f"| {label} | {n} | {r} | {pr} | {posr}/{n} | {g} | {pg} | {posg}/{n} |"


HEAD = ("| 比較（處理 − 控制） | n | Recall@20 配對差 [95% CI] | p | 為正 | NDCG@20 配對差 [95% CI] | p | 為正 |\n"
        "|---|---|---|---|---|---|---|---|")


def main():
    lines = ["## 主要比較：增強 vs α=0（α=1.0；12 個檢定，Bonferroni 門檻 0.05/12≈0.004）", "", HEAD]
    for label, tp in [("兩端", "week4fe_aug"), ("只 item", "week4fe_aug_itemonly")]:
        for r in ("0.2", "0.5", "1.0"):
            lines.append(row(f"{label} {int(float(r)*100)}% − α=0", load(tp, r, "1.0"), load("week4fe_aug", r, "0.0")))
    lines += ["", "## 探索性比較：user-only 與 Week 5 去噪（p 值未經校正；Week 5 共 16 個檢定，Bonferroni 門檻≈0.003）", "", HEAD]
    lines.append(row("user-only 100% − α=0", load("week4fe_aug_useronly", "1.0", "1.0"), load("week4fe_aug", "1.0", "0.0")))
    ctrl = load("week4fe_aug_itemonly", "1.0", "1.0")
    for label, tag in [("A drop_dirty", "itemdropdirty"), ("B drop_retry", "itemdropretry"),
                       ("prune10", "itemprune10"), ("prune20", "itemprune20"),
                       ("rand10", "itemrand10"), ("rand20", "itemrand20")]:
        lines.append(row(f"{label} − 未過濾 item-only", load(f"week4fe_aug_{tag}", "1.0", "1.0"), ctrl))
    for p_, r_ in [("10", "10"), ("20", "20")]:
        lines.append(row(f"prune{p_} − rand{r_}", load(f"week4fe_aug_itemprune{p_}", "1.0", "1.0"),
                         load(f"week4fe_aug_itemrand{r_}", "1.0", "1.0")))
    lines += ["", "## 探索性比較：Week 5 PageRank 篩選與隨機剔除（5 seed；p 值未經校正；PageRank 組共 28 個檢定，Bonferroni 門檻 0.05/28≈0.0018）", "", HEAD]
    pr_tags = [("prlow30", "itemprlow30"), ("prlow50", "itemprlow50"), ("prhigh30", "itemprhigh30"), ("prhigh50", "itemprhigh50"),
               ("rand30", "itemrand30"), ("rand50", "itemrand50"), ("rand80", "itemrand80"), ("rand90", "itemrand90")]
    for label, tag in pr_tags:
        lines.append(row(f"{label} − 未過濾 item-only", load(f"week4fe_aug_{tag}", "1.0", "1.0"), ctrl))
    for a, b in [("prlow30", "rand30"), ("prlow50", "rand50"), ("prhigh30", "rand30"), ("prhigh50", "rand50"),
                 ("prhigh30", "rand80"), ("prhigh50", "rand90")]:
        lines.append(row(f"{a} − {b}", load(f"week4fe_aug_item{a}", "1.0", "1.0"), load(f"week4fe_aug_item{b}", "1.0", "1.0")))
    with open("results/paired_ci_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/paired_ci_table.md")


if __name__ == "__main__":
    main()
