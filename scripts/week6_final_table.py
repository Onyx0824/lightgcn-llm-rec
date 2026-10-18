"""Week 6 report-ready cost-benefit table. Usage (project root): python scripts/week6_final_table.py
Reads results/week6_tier_table.csv and results/week6_tier_table_itemonly.csv; writes results/week6/final_table.md.
Gain = test gain over alpha=0 (10 seeds, 95% CI of paired difference). 'Share of full gain' is relative to users+items 100%.
"""
import pandas as pd
both = pd.read_csv("results/week6_tier_table.csv"); item = pd.read_csv("results/week6_tier_table_itemonly.csv")
full_r, full_n = both.loc[both.ratio == 1.0, "recall_gain"].item(), both.loc[both.ratio == 1.0, "ndcg_gain"].item()
full_h = both.loc[both.ratio == 1.0, "gen_hours_total"].item()
rows = [(f"只 item {int(r.ratio*100)}%", r.gen_hours_items, r) for r in item.itertuples()] + \
       [(f"兩端 {int(r.ratio*100)}%", r.gen_hours_total, r) for r in both.itertuples()]
L = ["| 方案 | 累計生成時間（小時） | 佔兩端100%時間 | Recall@20 增益 [95% CI] | 佔兩端100%增益 | NDCG@20 增益 [95% CI] | 佔兩端100%增益 |", "|---|---|---|---|---|---|---|"]
for lab, h, r in rows:
    L.append(f"| {lab} | {h:.1f} | {h/full_h:.0%} | {r.recall_gain:+.4f} [{r.recall_gain_ci_lo:+.4f}, {r.recall_gain_ci_hi:+.4f}] | {r.recall_gain/full_r:.0%} | "
             f"{r.ndcg_gain:+.4f} [{r.ndcg_gain_ci_lo:+.4f}, {r.ndcg_gain_ci_hi:+.4f}] | {r.ndcg_gain/full_n:.0%} |")
open("results/week6/final_table.md", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
