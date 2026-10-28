## Stratified by user (user activity; 4 tiers by train degree, T1 = lowest) - mean per-user Recall@20

alpha=0.0 (week4fe_aug_w7) vs alpha=1.0 (week4fe_aug_w7item), ratio 1.0, paired over seeds [0, 1, 2, 3, 4]. p-values uncorrected (exploratory).

| tier | #users | share of train interactions | #test positives | ctrl | aug | diff [95% CI] | relative | p | positive seeds |
|---|---|---|---|---|---|---|---|---|---|
| T1 (least active) | 244 | 3.3% | 24279 | 0.0897 | 0.0948 | +0.0051 [+0.0038, +0.0064] | +5.7% | 0.0004 | 5/5 |
| T2 (low-mid) | 243 | 11.1% | 18440 | 0.0899 | 0.0936 | +0.0037 [+0.0017, +0.0057] | +4.1% | 0.0065 | 5/5 |
| T3 (high-mid) | 243 | 24.8% | 17324 | 0.0737 | 0.0753 | +0.0016 [+0.0006, +0.0027] | +2.2% | 0.0124 | 5/5 |
| T4 (most active) | 243 | 60.9% | 24719 | 0.0606 | 0.0621 | +0.0015 [-0.0007, +0.0037] | +2.4% | 0.138 | 4/5 |

> Reading guide: a positive diff concentrated in T1/T2 means augmentation mostly helps long-tail (cold / sparse) cases; flat or negative diff in the hottest tier means popular items were already well served by the interaction graph.
