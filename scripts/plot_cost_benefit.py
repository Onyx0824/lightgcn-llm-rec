"""Week 6 cost-benefit curves: test gain over alpha=0 vs cumulative LLM generation hours.
Usage (project root): python scripts/plot_cost_benefit.py -> results/week6/cost_benefit.png
Inputs: results/week6_tier_table.csv (both sides), results/week6_tier_table_itemonly.csv (item-only).
Error bars = 95% CI of paired difference over 10 seeds. Labels are English to avoid CJK font issues.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

both = pd.read_csv("results/week6_tier_table.csv")
item = pd.read_csv("results/week6_tier_table_itemonly.csv")
os.makedirs("results/week6", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, m, title in [(axes[0], "recall", "Recall@20 gain over alpha=0"), (axes[1], "ndcg", "NDCG@20 gain over alpha=0")]:
    for df, x, lab, c, mk in [(both, "gen_hours_total", "Users + items", "#1f77b4", "o"),
                              (item, "gen_hours_items", "Items only", "#d62728", "s")]:
        xs = [0] + list(df[x]); ys = [0] + list(df[f"{m}_gain"])
        lo = [0] + list(df[f"{m}_gain"] - df[f"{m}_gain_ci_lo"]); hi = [0] + list(df[f"{m}_gain_ci_hi"] - df[f"{m}_gain"])
        ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, color=c, capsize=3, label=lab)
        for r, xv, yv in zip(df["ratio"], df[x], df[f"{m}_gain"]):
            ax.annotate(f"{int(r*100)}%", (xv, yv), textcoords="offset points", xytext=(5, -12), fontsize=8, color=c)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Cumulative generation time (hours)"); ax.set_ylabel("Gain (absolute)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
fig.suptitle("LLM augmentation: gain vs generation cost (alpha=1.0, 10 seeds, 95% CI)", fontsize=11)
fig.tight_layout(); fig.savefig("results/week6/cost_benefit.png", dpi=160)
print("saved results/week6/cost_benefit.png")
