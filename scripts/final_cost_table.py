"""Week 6 final cost-benefit table + ratio-based dual-axis figure.
Usage (project root): python scripts/final_cost_table.py
Outputs: results/week6/cost_benefit_table.md, results/week6/cost_benefit_dual_axis.png
Gains are absolute test-metric differences vs alpha=0 (paired over 10 seeds); generation hours are cumulative measured time.
"""
import glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

both = pd.read_csv("results/week6_tier_table.csv").set_index("ratio")
item = pd.read_csv("results/week6_tier_table_itemonly.csv").set_index("ratio")

def train_sec(pat):
    return np.mean([json.load(open(f, encoding="utf-8"))["train_sec"] for f in glob.glob(pat)])

rows = []
full_r, full_n = both.loc[1.0, "recall_gain"], both.loc[1.0, "ndcg_gain"]
for side, df, hcol, pat in [("兩端", both, "gen_hours_total", "results/week4fe_aug_r{r}_a1.0_s*.json"),
                            ("只 item", item, "gen_hours_items", "results/week4fe_aug_itemonly_r{r}_a1.0_s*.json")]:
    for r in (0.2, 0.5, 1.0):
        x = df.loc[r]
        rows.append((side, int(r * 100), x[hcol], x["recall_gain"], x["recall_gain_ci_lo"], x["recall_gain_ci_hi"],
                     x["ndcg_gain"], x["ndcg_gain_ci_lo"], x["ndcg_gain_ci_hi"],
                     x["recall_gain"] / full_r, x["ndcg_gain"] / full_n, train_sec(pat.format(r=r))))

L = ["## Week 6 效益總表（α=1.0，10 seed；增益為相對 α=0 的配對差，括號為 95% CI）", "",
     "| 增強端 | 生成比例 | 累計生成時間（小時） | Recall@20 增益 | NDCG@20 增益 | 占「兩端 100%」Recall 增益 | 占「兩端 100%」NDCG 增益 | 每次訓練時間（秒） |",
     "|---|---|---|---|---|---|---|---|"]
for s, r, h, rg, rl, rh, ng, nl, nh, rp, np_, ts in rows:
    L.append(f"| {s} | {r}% | {h:.2f} | {rg:+.4f} [{rl:+.4f}, {rh:+.4f}] | {ng:+.4f} [{nl:+.4f}, {nh:+.4f}] | {rp:.0%} | {np_:.0%} | {ts:.0f} |")
L += ["", "> 基準：α=0 訓練時間約 265 秒；增強不增加訓練時間（各組 258–271 秒），成本幾乎全來自 LLM 生成。",
      "> 「占兩端 100%」＝該組增益 ÷ 兩端 100% 的增益（Recall +0.0037、NDCG +0.0048），僅為點估計比例；CI 很寬，不宜當精確數字。"]
open("results/week6/cost_benefit_table.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))

# Dual-axis: x = generation ratio; left axis = gain (bars/lines); right axis = cumulative hours
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
xs = [20, 50, 100]
for ax, m, title in [(axes[0], "recall", "Recall@20"), (axes[1], "ndcg", "NDCG@20")]:
    for df, lab, c, mk in [(both, "Users + items", "#1f77b4", "o"), (item, "Items only", "#d62728", "s")]:
        g = df[f"{m}_gain"].values
        e = [g - df[f"{m}_gain_ci_lo"].values, df[f"{m}_gain_ci_hi"].values - g]
        ax.errorbar(xs, g, yerr=e, marker=mk, color=c, capsize=3, label=f"{lab} gain")
    ax.set_xticks(xs); ax.set_xlabel("Generation ratio (%)"); ax.set_ylabel("Gain over alpha=0"); ax.set_title(title)
    ax.axhline(0, color="gray", lw=0.8); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xs, both["gen_hours_total"].values, "--", color="#1f77b4", alpha=0.6, label="Users + items hours")
    ax2.plot(xs, item["gen_hours_items"].values, "--", color="#d62728", alpha=0.6, label="Items only hours")
    ax2.set_ylabel("Cumulative generation time (hours)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
fig.suptitle("Gain (solid, left) and generation time (dashed, right) vs generation ratio", fontsize=11)
fig.tight_layout(); fig.savefig("results/week6/cost_benefit_dual_axis.png", dpi=160)
