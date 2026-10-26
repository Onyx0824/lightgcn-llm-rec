# 報告大綱（對應證據檔）

> 敘事重心：挖到什麼知識、對企業的建議；工程細節放 repo，不放正文。每段結論都要有一個數字 + 出處。

## 1. 商業問題
- 推薦系統的稀疏／冷啟動：時間切分後 valid 73.7%、test 57.6% 的互動來自 train 沒看過的 user/item 而被移除（`docs/experiment-log.md` Week 2）。
- 研究問題：用本地 LLM 補充 user/item 屬性，值不值得？補多少？要不要清洗？

## 2. 方法（附一張架構圖，README 的流程圖可直接改）
- LightGCN baseline → LLM 生成（user profiling / item attribute）→ bge-small-zh embedding → 線性投影後以 α 加到 ID embedding → 去噪／剪枝分析 → 效益分析。
- 防洩漏設計（user 歷史只用 train）、固定 25 epoch + 10 seed 配對檢定、α=1.0 事前指定。

## 3. 實驗設置
- MovieLens-1M；6,040 users / 3,416 items；test 約 973 位 user；Recall@20 / NDCG@20；Qwen2.5-7B q4_K_M；RTX 4060。

## 4. 發現（每點一張圖或表）
| # | 發現 | 證據檔 |
|---|---|---|
| 1 | 增強有效但幅度小：兩端 100% Recall +4.7%、NDCG +1.8% | `results/paired_ci_table.md` |
| 2 | 增益主要來自 item 端；user 端偵測不到 | experiment-log Week 4 (6)(8) |
| 3 | Recall 約 50% 飽和，NDCG 到 100% 仍上升 | `results/week6/cost_benefit_dual_axis.png` |
| 4 | 成本效益：item 50% 約 1.9 小時 ≈ 兩端 100% 成本的 19%，取得 88% Recall 增益 | `results/week6/cost_benefit_table.md` |
| 5 | 去噪沒有額外益處（格式過濾、語意剪枝、PageRank 皆然） | `results/week5/three_way_comparison.md` |
| 6 | PageRank ≈ 熱門度（Spearman 0.9989）；效益隨「被增強 item 覆蓋的互動比例」飽和 | `results/week5/dose_response_ndcg20.png` |
| 7 | 增益落在熱門 item、對低活躍 user 較大；推薦 98.6%–99.0% 集中在最熱門四分之一 item | `results/week7/strata_item.md`, `strata_user.md` |
| 8 | 增強使 item embedding 類型凝聚度由 0.656 升至 0.826，熱門度仍為主軸（類型凝聚部分承襲 prompt 內的既有類型） | `results/week7/embeddings_tsne.png`, `embedding_cohesion.md` |
| 9 | 錯誤案例分類（熱門度偏誤、語意擴散、稀疏 user 受惠、兩模型皆失敗）；未找到明確的 LLM 幻覺案例 | `results/week7/error_cases.md`、README「錯誤分析」表 |

## 5. 商業建議（只寫有數字支撐的）
- 預算有限：只對 item 生成、約 50%、優先熱門 item。
- 重視排序品質（NDCG）：item 做到 100%（3.8 小時）。
- user profiling 不優先；不需要額外清洗 LLM 輸出。
- 增強主要強化熱門 item，不會把推薦推向長尾；對低活躍 user 的相對增益較大。若目標是長尾曝光或冷啟動，需要別的手段（例如 inductive 模型或針對冷門 item 的專門策略）。

## 6. 限制與未來方向
- test 只有 973 位重度使用者、冷啟動 user/item 被移除，因此「增強能否解決冷啟動」尚未直接檢驗。
- 單一資料集、單一 LLM、單一 prompt；p 值只反映訓練隨機性；去噪只檢驗兩種操作型定義。
- PageRank 只做 item 層級篩選，未改傳播權重；隨機對照只用一份剔除名單。
- 未來：inductive 模型（真正處理冷啟動）、多資料集、更大 LLM、PageRank 調權。
