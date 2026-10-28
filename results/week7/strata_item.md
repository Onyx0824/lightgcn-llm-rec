## Stratified by item (item popularity; 4 tiers by train degree, T1 = lowest) - Recall@20 (micro, per tier)

alpha=0.0 (week4fe_aug_w7) vs alpha=1.0 (week4fe_aug_w7item), ratio 1.0, paired over seeds [0, 1, 2, 3, 4]. p-values uncorrected (exploratory).

| tier | #items | share of train interactions | #test positives | ctrl | aug | diff [95% CI] | relative | p | positive seeds |
|---|---|---|---|---|---|---|---|---|---|
| T1 (coldest) | 852 | 1.7% | 3281 | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | n/a | nan | 0/5 |
| T2 (cool) | 852 | 7.2% | 9224 | 0.0007 | 0.0005 | -0.0002 [-0.0004, -0.0000] | -32.3% | 0.0474 | 0/5 |
| T3 (warm) | 852 | 20.3% | 20582 | 0.0022 | 0.0025 | +0.0003 [+0.0002, +0.0005] | +16.2% | 0.0061 | 5/5 |
| T4 (hottest) | 852 | 70.7% | 51675 | 0.0902 | 0.0921 | +0.0019 [+0.0012, +0.0027] | +2.2% | 0.002 | 5/5 |

### Where the top-20 slots go (test users) and catalogue coverage

| tier | slot share ctrl | slot share aug | diff [95% CI] | p |
|---|---|---|---|---|
| T1 (coldest) | 0.00% | 0.00% | +0.00% [+0.00%, +0.00%] | nan |
| T2 (cool) | 0.08% | 0.08% | +0.00% [-0.01%, +0.01%] | 0.6483 |
| T3 (warm) | 0.98% | 1.33% | +0.35% [+0.29%, +0.42%] | 0.0001 |
| T4 (hottest) | 98.95% | 98.59% | -0.36% [-0.43%, -0.29%] | 0.0001 |
| catalogue coverage (distinct items recommended / items) | 24.64% | 27.13% | +2.49% [+2.08%, +2.91%] | 0.0001 |

> Reading guide: a positive diff concentrated in T1/T2 means augmentation mostly helps long-tail (cold / sparse) cases; flat or negative diff in the hottest tier means popular items were already well served by the interaction graph.
