## Item embedding genre cohesion (3037 items with a top-8 primary genre; seeds [0, 1, 2, 3, 4])

| space | kNN genre purity (k=10) | silhouette (cosine, by genre) | Spearman(norm, train degree) |
|---|---|---|---|
| LLM text embedding (raw, 512-d) | 0.784 +/- 0.000 | 0.049 +/- 0.000 | -0.009 +/- 0.000 |
| LightGCN alpha=0 | 0.656 +/- 0.007 | 0.006 +/- 0.004 | 0.892 +/- 0.002 |
| LightGCN item-only augmented | 0.826 +/- 0.001 | 0.183 +/- 0.002 | 0.932 +/- 0.002 |

Paired kNN-purity difference (aug - alpha0): +0.1699, positive in 5/5 seeds, paired t p=0.0000 (exploratory).

> Higher purity / silhouette = items of the same genre sit closer together. Spearman(norm, degree) shows how strongly embedding length encodes popularity. Raw LLM text embedding is a reference, not a competitor.
