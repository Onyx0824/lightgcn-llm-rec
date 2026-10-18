"""Three-way comparison of Week 5 denoising families (format filter / semantic prune / PageRank)
at matched numbers of removed items, with a same-count random control. Seeds 0-4 only (the common seed set).
Usage (project root):  python scripts/three_way_table.py   -> results/week5/three_way_comparison.md
"""
import glob, json
import numpy as np
from scipy import stats

SEEDS = range(5)
N_ITEMS = 3408
# share of train interactions touched by the removed items (from Week 5 logs; see experiment-log (14))
INTER = {"unfiltered": 0.0, "random10": 9.7, "random20": 19.4, "prune10": 12.3, "prune20": 23.0,
         "prlow30": 2.6, "random30": 30.5, "prlow50": 9.0, "random50": 51.5}

def load(tag):
    out = {}
    for s in SEEDS:
        fs = glob.glob(f"results/week4fe_aug_{tag}_r1.0_a1.0_s{s}.json")
        if fs:
            j = json.load(open(fs[0], encoding="utf-8"))
            out[s] = (j["test_metrics"]["recall@20"], j["test_metrics"]["ndcg@20"], N_ITEMS - j["n_aug_items"])
    return out

def diff(a, b, k):
    seeds = sorted(set(a) & set(b)); x = np.array([a[s][k] for s in seeds]); y = np.array([b[s][k] for s in seeds])
    d = x - y; p = stats.ttest_rel(x, y).pvalue
    return f"{d.mean():+.4f} (p={p:.3f})" if p >= 0.001 else f"{d.mean():+.4f} (p<0.001)"

# (family, label, tag, matched random tag)
ROWS = [
    ("格式過濾", "A drop_dirty", "itemdropdirty", None),
    ("格式過濾", "B drop_retry", "itemdropretry", "itemrand10"),
    ("語意剪枝", "prune10", "itemprune10", "itemrand10"),
    ("語意剪枝", "prune20", "itemprune20", "itemrand20"),
    ("PageRank（熱門度）", "prlow30", "itemprlow30", "itemrand30"),
    ("PageRank（熱門度）", "prlow50", "itemprlow50", "itemrand50"),
]
ctrl = load("itemonly"); alpha0 = {s: v for s, v in
    ((s, (lambda j: (j["test_metrics"]["recall@20"], j["test_metrics"]["ndcg@20"], 0))(json.load(open(f"results/week4fe_aug_r1.0_a0.0_s{s}.json", encoding="utf-8")))) for s in SEEDS)}
def ms(d, k): v = [d[s][k] for s in SEEDS if s in d]; return f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"

L = ["## Week 5 三方對照：格式過濾 vs 語意剪枝 vs PageRank（item-only、ratio 1.0、α=1.0、seed 0–4）", "",
     "控制組為同 seed 的未過濾 item-only；「同數量隨機」為剔除 item 數最接近的 rand 組（單一剔除名單 `--seed 0`）。p 值為配對 t 檢定、未經校正。", "",
     "| 做法 | 變體 | 剔除 item 數 | 動到互動 | Recall@20 | NDCG@20 | vs 未過濾：Recall | vs 未過濾：NDCG | vs 同數量隨機：Recall | vs 同數量隨機：NDCG |",
     "|---|---|---|---|---|---|---|---|---|---|",
     f"| 無增強 | α=0 | — | — | {ms(alpha0,0)} | {ms(alpha0,1)} | {diff(alpha0,ctrl,0)} | {diff(alpha0,ctrl,1)} | — | — |",
     f"| 未過濾 | item-only | 0 | 0% | {ms(ctrl,0)} | {ms(ctrl,1)} | — | — | — | — |"]
for fam, lab, tag, rnd in ROWS:
    d = load(tag); r = load(rnd) if rnd else None
    removed = int(np.mean([d[s][2] for s in d])); share = INTER.get(tag.replace("item", ""), None)
    share_s = f"{share:.1f}%" if share is not None else "—"
    if tag in ("itemdropdirty", "itemdropretry"): share_s = "未計算"
    vs = (diff(d, r, 0), diff(d, r, 1)) if r else ("—", "—")
    L.append(f"| {fam} | {lab}（對照 {rnd.replace('item','') if rnd else '—'}） | {removed} | {share_s} | {ms(d,0)} | {ms(d,1)} | {diff(d,ctrl,0)} | {diff(d,ctrl,1)} | {vs[0]} | {vs[1]} |")
L += ["", "> 讀法：三種做法相對未過濾版都沒有 Recall 的穩健進步（最多 prune10 的 +0.0008）；NDCG 一律持平或下降。同數量隨機對照下，只有 prlow50 − rand50 的 NDCG 為正，但 prlow50 只動到 9.0% 互動、rand50 動到 51.5%，差異可由「增強覆蓋互動比例」解釋（見 experiment-log (14)），不代表選擇本身較佳。",
      "> 備註：A、B 在 seed 0–4 的子集上數字會與 10 seed 版（experiment-log (12)）略有不同；rand10 對照 B 時兩者剔除數相近（341 vs 342）但 A 沒有同數量隨機組。"]
open("results/week5/three_way_comparison.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
