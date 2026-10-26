# LightGCN + 本地LLM增強推薦系統：去噪與效益分析

> 中央大學資管所 碩一上「商業智慧」期中報告專案
> 以 0 成本、全本地運算的方式，探討 LLM 生成的輔助特徵能否提升推薦系統效果、增益落在哪些 user / item、值不值得為此付出生成成本，以及過量／低品質增強資訊是否真的帶來噪音。

## 專案背景

推薦系統長期面臨資料稀疏與冷啟動問題。近年研究（LLMRec、RPP、GAECL、LightGNN）嘗試用 LLM 生成的 user/item 屬性文字來補充圖結構資訊，但這類增強是否「越多越好」、低品質或過量的生成內容會不會反而傷害效果，仍需要實證。

本專案在 **LightGCN baseline** 之上，加入**本地端 LLM 生成的 user profiling / item attribute 增強特徵**，並實作**去噪／剪枝分析**與**分層分析**，量化增強帶來的效果提升、噪音的影響、增益落在哪一層使用者與物品，最後產出**效益分析**（效果 vs. 運算成本），轉化為具體的商業建議。

## 方法架構

```
Raw Interaction Data (MovieLens-1M)
        │
        ▼
 ┌─────────────┐      ┌──────────────────────┐
 │  LightGCN   │◄─────│ LLM 生成的 user/item  │
 │  Baseline   │      │ 屬性文字 → Embedding  │
 └─────────────┘      └──────────────────────┘
        │                       │
        ▼                       ▼
   Recall@20 /            去噪/剪枝分析
   NDCG@20 對照          (格式過濾、語意離群剪枝、
                          PageRank／熱門度篩選)
        │                       │
        └───────────┬───────────┘
                     ▼
        效益分析（效果 vs. 運算時間）＋ 分層分析（item 熱門度、user 活躍度）
                     ▼
              商業建議：何時值得用 LLM 增強、增強多少、補在哪裡
```

## 技術棧（全程 0 成本，本地運算）

| 用途 | 工具 |
|---|---|
| GNN 推薦模型 | PyTorch（自行實作 LightGCN） |
| LLM 推論 | Ollama + `qwen2.5:7b-instruct-q4_K_M`（量化版） |
| 文字轉 Embedding | sentence-transformers（`BAAI/bge-small-zh-v1.5`，512 維） |
| 去噪/剪枝 | 格式過濾（`--drop_dirty` / `--drop_retry`）、embedding 語意離群剪枝、二部圖 PageRank 篩選 |
| 分析與視覺化 | numpy / pandas / scipy / scikit-learn / matplotlib（PCA、t-SNE） |
| 硬體 | RTX 4060 Laptop（8GB VRAM）／16GB RAM |

## 專案結構

> 標註「不進版控」的資料夾由程式產生，clone 後需依「重現實驗」重建。

```
.
├── README.md                         # 本檔：專案說明、主要結果、重現步驟
├── LICENSE                           # MIT 授權
├── requirements.txt                  # 套件清單（第一行已含 PyTorch CUDA 12.6 的 extra-index-url）
├── .gitignore                        # 排除 data/raw、*.npy、checkpoints、venv 等
│
├── configs/
│   └── baseline.yaml                 # LightGCN 超參數（embed_dim、層數、lr、batch、epoch、K…），所有訓練腳本共用
│
├── data/
│   ├── README.md                     # 資料集來源與下載說明
│   ├── raw/ml-1m/                    # MovieLens-1M 原始檔 ratings/movies/users.dat（不進版控）
│   └── processed/                    # 前處理輸出：train/valid/test.csv、id_maps.json、stats.json（不進版控）
│
├── src/                              # 可重用的函式庫程式碼（被 scripts/ 引用）
│   ├── data_processing/
│   │   └── preprocess.py             # 過濾低頻 user/item、id 映射、時間排序 70/10/20 切分、移除冷啟動
│   ├── model/
│   │   ├── lightgcn.py               # LightGCN（鄰居聚合 + BPR loss）、對稱正規化鄰接矩陣
│   │   └── lightgcn_aug.py           # LLM 增強版：ID embedding + α·mask·W·LLM embedding（α=0 與 baseline 等價）
│   ├── llm_augmentation/
│   │   ├── prompts.py                # user profiling / item attribute 兩支 prompt、必要欄位與值域檢查
│   │   ├── ollama_client.py          # 呼叫本地 Ollama + JSON parser + retry（含值域檢查）
│   │   ├── test_ollama.py            # Week 1：Ollama 推論速度與 VRAM 測試
│   │   ├── run_week3_pilot.py        # Week 3：100 user / 100 item 小規模驗證（僅供 prompt 設計，不用於評估）
│   │   ├── run_week4_generate.py     # Week 4：全量批次生成（防洩漏、可續跑、20/50/100% 嵌套順序、耗時記錄）
│   │   └── build_embeddings.py       # 生成文字 → bge embedding；支援 --drop_dirty / --drop_retry 去噪版
│   ├── analysis/                     # 分析用的共用函式（腳本只負責 CLI 與輸出）
│   │   ├── stats_utils.py            # paired_ci：依 seed 配對差、t 分佈 95% CI、paired t 檢定
│   │   ├── pagerank.py               # 二部圖 PageRank（power iteration，與 networkx 差距 < 1e-11）
│   │   ├── pruning.py                # 語意離群分數：與同類型 leave-one-out 中心的平均 cosine
│   │   ├── stratify.py               # 讀取 top-20 清單、hit 向量、度數分位分層、user 層級 recall / hit rate
│   │   └── embedding_metrics.py      # L2 正規化、kNN 類型純度
│   └── evaluation/
│       └── metrics.py                # Recall@K、NDCG@K、全量 ranking 評估（排除已看過項目）
│
├── scripts/                          # 可直接執行的實驗與分析腳本（從專案根目錄執行）
│   ├── train_baseline.py             # 訓練 LightGCN 控制組（early stopping）並存 checkpoint
│   ├── train_augmented.py            # 訓練增強版：--ratio --alpha --seed --aug_side --fixed_epochs --tag --save_emb
│   ├── compare_ratios.py             # 20/50/100% 各比例增益與相鄰比例配對差、累計生成耗時 → CSV
│   ├── summarize_week4.py            # 任兩組結果（--prefix / --control_prefix）的配對比較
│   ├── paired_ci_table.py            # 所有報告用比較的配對差 + 95% CI + p → results/paired_ci_table.md
│   ├── inspect_dirty.py              # 盤點 dirty items 由哪些字元觸發（Week 5）
│   ├── prune_items.py                # 語意離群剪枝 / 隨機剔除對照（Week 5）
│   ├── prune_pagerank.py             # 依 PageRank 剔除 item 增強（Week 5）
│   ├── dose_response.py              # 劑量反應：被剔除互動比例 vs 保留的增益（Week 5）
│   ├── three_way_table.py            # 格式過濾 / 語意剪枝 / PageRank 並排對照表（Week 5）
│   ├── plot_cost_benefit.py          # 效益曲線：增益 vs 累計生成時間（Week 6）
│   ├── final_cost_table.py           # 最終成本效益表與「比例對增益／耗時」疊圖（Week 6）
│   ├── popularity_strata.py          # item 熱門度 / user 活躍度分層分析（Week 7）
│   ├── error_analysis.py             # 挑錯誤分析候選案例並附證據（Week 7）
│   ├── visualize_embeddings.py       # item embedding 的 PCA / t-SNE 與類型凝聚度數字（Week 7）
│   ├── run_week5_denoise.ps1         # Week 5 格式過濾去噪一鍵執行（10 seed）
│   ├── run_week5_prune.ps1           # Week 5 語意剪枝 + 隨機對照一鍵執行（5 seed）
│   ├── run_week5_pagerank.ps1        # Week 5 PageRank + rand30/50/80/90 對照一鍵執行（5 seed）
│   └── run_week7.ps1                 # Week 7：訓練 w7 對照組/處理組並存 embedding，接著跑全部分析
│
├── tests/                            # 單元測試（python -m pytest tests -q）
│   ├── test_embedding.py             # Week 1：sentence-transformers 環境煙霧測試
│   ├── test_metrics.py               # Recall@K / NDCG@K 以手算值驗證
│   └── test_lightgcn_aug.py          # 驗證 α=0 的增強版與 baseline 數學等價（控制組假設）
│
├── results/                          # 實驗結果（JSON/CSV/MD/圖進版控；*.npy 不進版控）
│   ├── baseline_results.json         # Week 2 baseline（early stopping，僅供參考）
│   ├── week3_pilot_*.json            # Week 3 小規模驗證結果與彙總
│   ├── week4/                        # LLM 生成原始輸出（users/items.jsonl）、order.json、timing.jsonl、embedding（.npy 不進版控）
│   ├── week4fe_aug_*.json            # 各組訓練結果（檔名含 ratio / alpha / seed / tag）
│   ├── week5/                        # 去噪變體的 item_has 遮罩與統計、劑量反應圖表、三方對照表
│   ├── week6/                        # 效益曲線圖、疊圖、最終成本效益表
│   ├── week6_tier_table*.csv         # 各比例增益與生成耗時（compare_ratios.py 輸出）
│   ├── week7/                        # 分層分析、錯誤案例、embedding 視覺化；emb/ 子資料夾存 .npy（不進版控）
│   └── paired_ci_table.md            # 全部比較的配對差、95% CI、p 值
│
├── checkpoints/                      # baseline 模型權重（不進版控）
├── notebooks/                        # 探索性分析（目前空）
└── docs/
    ├── project-plan.md               # 原始專案計畫（摘要版）
    ├── project-plan-detailed.md      # 逐週詳細計畫（9/22–11/30）
    ├── experiment-log.md             # 每週實驗紀錄與完整統計（報告數字的來源）
    └── report-outline.md             # 報告大綱：各發現對應的證據檔
```

## 環境安裝

```bash
# 1. 建立虛擬環境（Python 3.11）
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# 2. 安裝套件（requirements.txt 第一行已指定 PyTorch CUDA 12.6 的下載來源，torch 會一併裝好）
pip install -r requirements.txt

# 3. 安裝 Ollama，下載量化模型
# https://ollama.com
ollama pull qwen2.5:7b-instruct-q4_K_M

# 4. 驗證環境
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests -q
```

> Windows 若出現 `WinError 4551`（應用程式控制原則封鎖 DLL）：`Get-ChildItem -Recurse .\venv | Unblock-File`。
> 若 `.ps1` 被執行原則擋住：`Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`，再對該檔 `Unblock-File`。

## 目前進度

- [x] Python 3.11 虛擬環境、PyTorch（CUDA 12.6）、Ollama、sentence-transformers 環境建置
- [x] 資料集選定（MovieLens-1M）
- [x] LightGCN baseline 實作與訓練（詳見 [`docs/experiment-log.md`](./docs/experiment-log.md)）
- [x] LLM 增強 prompt 設計與小規模驗證（user profiling + item attribute，含 JSON 值域檢查與 retry 機制，n=100/100，詳見 experiment-log.md Week 3）
- [x] LLM 增強全量生成（20% / 50% / 100%）→ embedding 整合 → 接入 LightGCN
- [x] 增強效益分析：生成比例、α 敏感度、user 端 vs item 端消融（experiment-log.md Week 4）
- [x] 去噪/剪枝分析（Week 5）：格式過濾、語意離群剪枝、PageRank（item 層級）篩選、隨機對照與劑量反應；結論為無額外益處，PageRank 在二部圖上等同熱門度
- [x] 效益分析圖表與最終表格（Week 6：`results/week6/`）
- [x] item 熱門度 / user 活躍度分層分析（Week 7：`results/week7/strata_item.md`、`strata_user.md`）
- [x] 錯誤分析（Week 7：`scripts/error_analysis.py` 產生候選案例，已人工挑 9 筆分類，見下方「錯誤分析」與 experiment-log Week 7）
- [x] embedding 視覺化與類型凝聚度（Week 7：`results/week7/embeddings_*.png`、`embedding_cohesion.md`）
- [ ] 報告撰寫（大綱見 [`docs/report-outline.md`](./docs/report-outline.md)）

> 完整時程規劃見 [`docs/project-plan-detailed.md`](./docs/project-plan-detailed.md)（9/22–11/30，共 10 週）。

## 實驗設定

- **資料**：MovieLens-1M，過濾互動 <5 次的 user/item 後為 6,040 users / 3,416 items / 999,611 interactions
- **切分**：依時間排序 70/10/20；valid/test 中出現冷啟動 user/item 的互動予以移除（valid 73.7%、test 57.6%）；test 評估時排除 train ∪ valid 已互動項目（test 集約 973 位 user）。**因此評估只涵蓋 train 看過的 user 與 item，冷啟動本身未被直接評估。**
- **LLM 生成**：Ollama `qwen2.5:7b-instruct-q4_K_M`，輸出繁體中文 JSON，含值域檢查與 retry；user 歷史只取 train 內互動（防資料洩漏）
- **文字 embedding**：`BAAI/bge-small-zh-v1.5`（512 維）
- **比較方式**：控制組為 α=0（與 baseline 數學等價），固定 25 epoch，依 seed 做配對檢定；主實驗 10 seed（0–9），Week 5 語意剪枝與 PageRank 組、Week 7 分層分析為 5 seed（0–4）
- **α=1.0 為事前指定**；α 掃描（0.3 / 1.0 / 3.0）僅作敏感度分析，避免用 test 調參
- **統計方法**：每個條件以多個 seed 各訓練一次；同一 seed 的兩個條件互相配對，對配對差做雙尾 paired t 檢定，並報告配對差的 95% 信賴區間（t 分佈）。p 值只反映訓練隨機性（初始化與抽樣），不代表推廣到其他使用者群體，因為 test 固定為同一批約 973 位 user。完整的配對差、信賴區間與 p 值見 [`results/paired_ci_table.md`](./results/paired_ci_table.md)（由 `scripts/paired_ci_table.py` 產生）。
- **多重比較**：主要結論只依賴下列直接對應研究問題的比較——增強 vs α=0（兩端與只 item，20% / 50% / 100%，共 12 個檢定，Bonferroni 門檻 0.05/12≈0.004）。其餘比較（user-only、α 掃描、Week 5 去噪變體與隨機對照、Week 7 分層）視為探索性，p 值未經校正，僅作方向性證據。邊緣顯著的結果（例如兩端 20% 的 NDCG，p=0.028）不視為主要發現。

## 主要結果

### 增強比例與 user/item 消融（10 seed）

| 模型 | Recall@20 | NDCG@20 | 備註 |
|---|---|---|---|
| LightGCN（控制組，固定 25 epoch，10 seed） | 0.0786 ± 0.0006 | 0.2714 ± 0.0008 | 對照基準 |
| + LLM 增強 兩端 20%（α=1.0） | 0.0804 ± 0.0010 | 0.2722 ± 0.0008 | 累計生成 1.9 小時 |
| + LLM 增強 兩端 50%（α=1.0） | 0.0819 ± 0.0009 | 0.2740 ± 0.0021 | 累計生成 4.8 小時 |
| + LLM 增強 兩端 100%（α=1.0） | 0.0822 ± 0.0017 | 0.2762 ± 0.0026 | 累計生成 9.9 小時 |
| + LLM 增強 只 item 20%（α=1.0） | 0.0804 ± 0.0010 | 0.2719 ± 0.0008 | 累計生成 0.8 小時 |
| + LLM 增強 只 item 50%（α=1.0） | 0.0818 ± 0.0007 | 0.2737 ± 0.0011 | 累計生成 1.9 小時 |
| + LLM 增強 只 item 100%（α=1.0） | 0.0812 ± 0.0007 | 0.2756 ± 0.0015 | 累計生成 3.8 小時 |

### 去噪（item 端 100%、α=1.0；Week 5）

格式過濾（10 seed）：

| 模型 | 剔除 item 數 | Recall@20 | NDCG@20 | vs 未過濾 item-only |
|---|---|---|---|---|
| 未過濾 item-only | 0 | 0.0812 ± 0.0007 | 0.2756 ± 0.0015 | — |
| A：drop_dirty（簡體殘留） | 72 | 0.0814 ± 0.0008 | 0.2751 ± 0.0012 | Recall 無差異；NDCG −0.0005（p=0.018） |
| B：drop_retry（重試 >1 次） | 342 | 0.0814 ± 0.0011 | 0.2749 ± 0.0015 | Recall 無差異；NDCG −0.0007（p=0.007） |

語意離群剪枝與同數量隨機剔除（5 seed，控制組為同 seed 的未過濾 item-only：Recall 0.0815 ± 0.0007、NDCG 0.2762 ± 0.0012）：

| 模型 | Recall@20 | NDCG@20 | vs 未過濾 item-only |
|---|---|---|---|
| prune10（剔除最離群 10%） | 0.0822 ± 0.0011 | 0.2741 ± 0.0013 | Recall +0.0008（p=0.019）；NDCG −0.0021（p=0.008） |
| prune20 | 0.0815 ± 0.0013 | 0.2736 ± 0.0012 | Recall 無差異；NDCG −0.0026（p=0.001） |
| rand10（隨機剔除同數量） | 0.0813 ± 0.0005 | 0.2747 ± 0.0009 | Recall 無差異；NDCG −0.0015（p=0.023） |
| rand20 | 0.0813 ± 0.0011 | 0.2754 ± 0.0014 | Recall 無差異；NDCG −0.0008（p=0.082） |

PageRank 篩選與隨機剔除（5 seed，同上控制組；PageRank 與 item degree 的 Spearman = 0.9989，實質為熱門度篩選）：

| 模型 | 剔除 item 數 | 被剔除 item 佔 train 互動 | Recall@20 | NDCG@20 | vs 未過濾 item-only |
|---|---|---|---|---|---|
| prlow30（剔除 PageRank 最低 30%） | 1022 | 2.6% | 0.0820 ± 0.0013 | 0.2752 ± 0.0013 | Recall 無差異；NDCG −0.0010（p=0.033） |
| prlow50 | 1704 | 9.0% | 0.0817 ± 0.0004 | 0.2752 ± 0.0006 | Recall 無差異；NDCG −0.0010（p=0.064） |
| rand50（隨機剔除同數量） | 1704 | 51.5% | 0.0819 ± 0.0011 | 0.2730 ± 0.0008 | Recall 無差異；NDCG −0.0032（p<0.001） |
| prhigh30（剔除 PageRank 最高 30%） | 1022 | 76.5% | 0.0796 ± 0.0007 | 0.2722 ± 0.0005 | Recall −0.0019（p=0.002）；NDCG −0.0040（p<0.001） |
| rand80 | 2726 | 80.2% | 0.0801 ± 0.0003 | 0.2720 ± 0.0005 | Recall −0.0014（p=0.016）；NDCG −0.0042（p=0.005） |
| rand90 | 3067 | 90.6% | 0.0784 ± 0.0006 | 0.2703 ± 0.0006 | Recall −0.0031（p<0.001）；NDCG −0.0059（p=0.001） |
| prhigh50 | 1704 | 91.1% | 0.0796 ± 0.0006 | 0.2726 ± 0.0009 | Recall −0.0019（p=0.021）；NDCG −0.0036（p=0.011） |

三種去噪做法（格式過濾／語意剪枝／PageRank）在相近剔除量下的並排對照見 [`results/week5/three_way_comparison.md`](./results/week5/three_way_comparison.md)（`scripts/three_way_table.py`）。（rand30 等其餘變體與配對比較見 [`docs/experiment-log.md`](./docs/experiment-log.md) Week 5 (14)(15)；劑量反應圖見 `results/week5/dose_response_ndcg20.png`、`dose_response_recall20.png`。）

### 分層分析（item-only 100%、α=1.0 vs α=0，5 seed，探索性；Week 7）

**重現性檢查**：Week 7 另以 `w7` 標籤重新訓練（存下 embedding 與 top-20）。5 seed 結果 α=0 為 Recall 0.0785 / NDCG 0.2714，item-only 為 0.0815 / 0.2762，配對差 Recall +0.0030（+3.8%，5/5，p=0.001）、NDCG +0.0048（+1.8%，5/5，p=0.005），與主實驗、Week 5 的未過濾 item-only 控制組數字一致。

**依 item 熱門度**（train 度數四等分，每層 852 個 item；指標為各層 test 正樣本的 Recall@20，micro）：

| item 熱門度 | 佔 train 互動 | test 正樣本數 | α=0 | item-only | 配對差 [95% CI] | 相對 | p | 為正 |
|---|---|---|---|---|---|---|---|---|
| T1（最冷門） | 1.7% | 3,281 | 0.0000 | 0.0000 | 0.0000 | n/a | — | 0/5 |
| T2 | 7.2% | 9,224 | 0.0007 | 0.0005 | −0.0002 [−0.0004, −0.0000] | −32% | 0.047 | 0/5 |
| T3 | 20.3% | 20,582 | 0.0022 | 0.0025 | +0.0003 [+0.0002, +0.0005] | +16% | 0.006 | 5/5 |
| T4（最熱門） | 70.7% | 51,675 | 0.0902 | 0.0921 | +0.0019 [+0.0012, +0.0027] | +2.2% | 0.002 | 5/5 |

top-20 推薦名額 98.95%（α=0）／98.59%（增強）給最熱門的四分之一 item；被推薦過的不同 item 佔目錄比例由 24.6% 增至 27.1%（+2.5 個百分點，p=0.0001），名額從 T4 微幅移向 T3（+0.35 個百分點）。

**依 user 活躍度**（train 度數四等分，每層約 243 位；指標為各層平均 per-user Recall@20）：

| user 活躍度 | 佔 train 互動 | α=0 | item-only | 配對差 [95% CI] | 相對 | p | 為正 |
|---|---|---|---|---|---|---|---|
| T1（最不活躍） | 3.3% | 0.0897 | 0.0948 | +0.0051 [+0.0038, +0.0064] | +5.7% | 0.0004 | 5/5 |
| T2 | 11.1% | 0.0899 | 0.0936 | +0.0037 [+0.0017, +0.0057] | +4.1% | 0.007 | 5/5 |
| T3 | 24.8% | 0.0737 | 0.0753 | +0.0016 [+0.0006, +0.0027] | +2.2% | 0.012 | 5/5 |
| T4（最活躍） | 60.9% | 0.0606 | 0.0621 | +0.0015 [−0.0007, +0.0037] | +2.4% | 0.138 | 4/5 |

### item embedding 結構（item-only 100%、α=1.0 vs α=0，5 seed；Week 7，探索性）

只用 8 個最常見的主類型（第一個列出的 MovieLens 類型）共 3,037 個 item；各空間中「最近鄰（cosine, k=10）與自己同類型的比例」與輪廓係數：

| 空間 | kNN 類型純度 | 輪廓係數（依類型） | Spearman(embedding 長度, train 度數) |
|---|---|---|---|
| LLM 文字 embedding（bge，原始 512 維，僅供參考） | 0.784 | 0.049 | −0.009 |
| LightGCN α=0 | 0.656 ± 0.007 | 0.006 ± 0.004 | 0.892 ± 0.002 |
| LightGCN item-only 增強 | 0.826 ± 0.001 | 0.183 ± 0.002 | 0.932 ± 0.002 |

配對 kNN 純度差（增強 − α=0）+0.170，5/5 seed 為正（p<0.0001，探索性）。圖見 `results/week7/embeddings_tsne.png`、`embeddings_pca.png`（t-SNE 軸線無法跨面板比較，結論以上表數字為準）。

### 錯誤分析（Week 7；seed 0–4 平均後挑選，原始證據見 `results/week7/error_cases.md`）

973 位 test user 中，增強後個人分數（命中數 / min(test 數, 20)）下降 301 位、上升 400 位、不變 272 位；平均分數 0.2569 → 0.2638。依「下降最多／上升最多／增強後最低」各挑代表，並人工判斷主因：

| user | train / test | 分數 α=0 → 增強 | 組別 | 主因判斷 | 證據 |
|---|---|---|---|---|---|
| 2790 | 38 / 10 | 0.60 → 0.20 | 下降 | 稀疏 user ＋ 語意擴散（推測） | 偏好 Drama/Western/Crime，推薦擴散到戰爭／史詩經典（Kwai、Lawrence、Killing Fields）；這些 item 的 LLM 標籤共通出現「深刻」「成年觀眾」 |
| 1698 | 184 / 164 | 0.65 → 0.29 | 下降 | 無可辨識錯誤（排序微擾） | 推薦與偏好類型 100% 吻合且仍命中多部，找不到語意問題 |
| 1210 | 111 / 223 | 0.53 → 0.28 | 下降 | 熱門度偏誤 | 推薦為 Star Wars、Raiders、Back to the Future 等大片，與其 train 類型（Drama/Comedy/Western）有距離 |
| 2513 | 28 / 3 | 0.00 → 0.67 | 上升 | 稀疏 user 受惠於 item 語意 | 推薦 100% 吻合 Mystery/Thriller，命中 Vertigo、Rear Window（test 僅 3 筆，分數粒度粗） |
| 5000 | 46 / 179 | 0.23 → 0.56 | 上升 | LLM 補充屬性帶來新資訊 | test 命中 Mary Poppins、Fantasia、Snow White 等家庭片，不在其 train 前三類型；這些 item 的 `supplementary_genres` 為 "Family" |
| 3273 | 13 / 184 | 0.57 → 0.86 | 上升 | 稀疏 user ＋ 熱門科幻 | train 僅 13 筆，推薦熱門科幻動作片並大量命中 |
| 1284 | 385 / 166 | 0.00 → 0.00 | 最低 | 兩模型皆失敗（時間漂移，推測） | 推薦與 train 類型 100% 吻合但完全未命中，α=0 同樣為 0 |
| 1361 | 77 / 27 | 0.00 → 0.00 | 最低 | 兩模型皆失敗（時間漂移，推測） | 推薦 1999 年熱門片，α=0 同樣為 0 |
| 1500 | 98 / 44 | 0.10 → 0.00 | 最低 | 兩模型皆失敗（時間漂移，推測） | 推薦 100% 為其偏好的喜劇／愛情，仍未命中 |

觀察：（1）LLM 輸出的簡體殘留／重試標記（[D]、[R]）同時出現在「下降」與「上升」兩組的推薦中（例如 Rear Window 同時出現在 2790 的錯誤推薦與 2513 的命中推薦），看不出噪音是誤差來源，與 Week 5 去噪無益一致；（2）「上升」組 5 位中 4 位 train 少於 50 筆，與 user 分層結果一致；（3）「最低」組在 α=0 同樣失敗，這類錯誤不是增強造成，推測來自時間切分下 user 興趣漂移，本專案尚未直接檢驗；（4）依「變化最大」挑選會偏向 test 筆數很少的 user（如 2513 僅 3 筆、4804 僅 2 筆），單筆命中就造成大幅變動，案例只用於說明機制，不作為效果證據。

### 關鍵發現

1. **增益主要來自 item 端**：item-only（100%）Recall +0.0027（10/10 seed 為正，p<0.001）；user-only 不顯著（Recall +0.0003，p=0.23）。
2. **Recall 在 50% 即飽和，NDCG 到 100% 仍上升**：50%→100% Recall +0.0003（p=0.41）、NDCG +0.0022（p=0.0013）。
3. **成本效益**：item 端 50% 累計生成約 1.9 小時，為兩端 100%（9.9 小時）的 19%，卻取得約 88% 的 Recall 增益。
4. **α 存在平台區**：α 在 0.3–1.0 表現穩定；α=3.0 過大，NDCG 增益消失。
5. **LLM 輸出噪音低**（100% 生成）：dirty users 7 / items 72（items 的 2.1%）；retry>1 users 155 / items 342（items 的 10.0%）。
6. **去噪沒有額外益處**：格式過濾與語意剪枝都沒有提升 Recall，NDCG 反而小幅下降（−0.0005～−0.0026），連隨機剔除同數量 item 也一樣，與「被增強的 item 變少」一致；語意剪枝在 NDCG 上並未優於隨機剔除（prune20 − rand20：−0.0018）。所有去噪版本相對 α=0 的增益（Recall +3.6%、NDCG +1.3%～+1.4%）都保留。建議直接保留全部增強特徵，不需額外清洗。
7. **PageRank 篩選等同熱門度篩選；增強效益隨「被增強 item 覆蓋的互動比例」飽和**：二部圖上 PageRank 與 degree 幾乎相同（Spearman 0.9989）。Recall 增益在被增強 item 覆蓋約 48% 以上的互動時完整保留，覆蓋約 20–24% 時剩約一半，約 9–10% 時消失；在覆蓋相近時，剔除高 PageRank item 與隨機剔除沒有可靠差異（prhigh30 vs rand80：Recall p=0.12、NDCG p=0.64），因此 PageRank 沒有帶來超越熱門度的去噪效果。例外是 prhigh50 對 rand90（兩者都動到約 91% 互動，前者較好），原因未釐清。**成本效益（推算）**：只對熱門度／PageRank 前 50% 的 item 生成（prlow50）約可省一半 item 端生成量，Recall 不變、NDCG 約低 0.4%（p=0.064）；相同生成量下隨機選 50% 的 NDCG 低 1.2%（p<0.001）。
8. **增益落在熱門 item，而不是長尾 item**（Week 7，探索性）：item-only 增強帶來的額外命中幾乎全來自最熱門的四分之一 item（T4 Recall +0.0019，5/5 seed 為正）；依各層 test 正樣本數粗估，T4 貢獻約九成以上的額外命中。最冷門兩層（T1、T2，合計佔 test 正樣本約 15%）的 Recall 幾乎為 0，增強沒有改善、T2 還略降（−0.0002，p=0.047，未校正）。模型的 top-20 有 98.6%～99.0% 都推薦給最熱門的四分之一 item，增強只讓目錄覆蓋率微增 2.5 個百分點。**換言之，單靠 item 端 LLM 屬性增強並沒有把推薦推向長尾；與 Week 5「保留熱門 item 的增強就保留大部分增益」一致。**
9. **增益對低活躍度 user 較大**（Week 7，探索性）：雖然只增強 item，最不活躍的兩個四分位 user 的 Recall 相對增益為 +5.7%、+4.1%（各 5/5 seed 為正），最活躍的四分位為 +2.4% 且 CI 含 0。各層的 CI 有重疊，「增益隨活躍度遞減」是趨勢描述，未做層間差異的直接檢定；各層平均 recall 的基準值不同（重度 user 的 test 正樣本多，Recall@20 上限較低），因此宜看相對增益。
10. **增強讓 item embedding 的類型結構更清楚，但熱門度仍是主軸**（Week 7，探索性）：item-only 增強使 item embedding 的 kNN 類型純度由 0.656 升到 0.826（5/5 seed，輪廓係數 0.006 → 0.183），甚至高於原始 LLM 文字 embedding 本身（0.784）；同時 embedding 長度與 item 熱門度的 Spearman 由 0.892 升到 0.932，顯示熱門度仍是 embedding 的主要組織方向，與發現 8（推薦集中熱門 item）一致。**注意**：LLM 生成時的 prompt 含 MovieLens 既有類型，因此類型凝聚度有一部分是承襲輸入標籤，並非 LLM 獨立產生的新知識；這項數字說明「增強把語意結構注入 embedding」，不能單獨證明它造成了推薦效果。
11. **錯誤分析**（Week 7，見上表）：沒有找到「LLM 幻覺」造成的明確錯誤；增強後變差的案例主要是熱門度偏誤、語意擴散（推測）或單純排序微擾，變好的案例多為稀疏 user 與 LLM 補充屬性（如 Family）帶來的新資訊，兩模型都失敗的案例推測來自興趣隨時間漂移。LLM 簡體殘留／重試標記在變好與變差的案例中都出現，看不出噪音是主因。
12. **限制**：相對增益約 +2%～+5%（絕對值 +0.0003～+0.0048），test 僅約 973 位 user，且已排除冷啟動 user/item；p 值只反映訓練隨機性；語意剪枝、PageRank 與分層分析僅 5 seed、隨機對照只用一份剔除名單；PageRank 只做 item 層級篩選（未改圖傳播權重）；噪音只以「格式」與「類型內語意一致性」兩種操作型定義檢驗；結論限於 MovieLens-1M、本 prompt 與 `qwen2.5:7b-instruct-q4_K_M`。結論請以配對差與 95% CI（`results/paired_ci_table.md`）為準，不宜過度外推。

完整表格見 `results/week6_tier_table.csv`，每週詳細統計見 [`docs/experiment-log.md`](./docs/experiment-log.md)。

## 重現實驗

在專案根目錄、已啟動虛擬環境後。

### 1. 從頭建立（資料 → baseline → 生成 → 增強訓練）

```powershell
# 下載 MovieLens-1M 解壓到 data/raw/ml-1m/（見 data/README.md）
python src/data_processing/preprocess.py                          # → data/processed/
python scripts/train_baseline.py --config configs/baseline.yaml   # → results/baseline_results.json（約 8 分鐘）

ollama serve                                                      # 另開一個終端機常駐
python src/llm_augmentation/run_week4_generate.py --max_ratio 1.0   # 全量生成約 9.9 小時，可中斷續跑；--max_ratio 0.2 / 0.5 可分階段（務必在專案根目錄執行）
python src/llm_augmentation/build_embeddings.py                   # → results/week4/*.npy

# 主實驗（固定 25 epoch；α=0 為控制組；每次約 4.4 分鐘，seed 0–9）
python scripts/train_augmented.py --ratio 1.0 --alpha 0   --seed 0 --fixed_epochs 25
python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed 0 --fixed_epochs 25
python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed 0 --aug_side item --fixed_epochs 25   # 只 item

python scripts/compare_ratios.py                                  # 各比例增益與耗時 → results/week6_tier_table.csv
python scripts/paired_ci_table.py                                 # → results/paired_ci_table.md
```

### 2. Week 5 去噪

```powershell
.\scripts\run_week5_denoise.ps1     # 格式過濾 A/B（10 seed）
.\scripts\run_week5_prune.ps1       # 語意剪枝 + 隨機對照（5 seed）
.\scripts\run_week5_pagerank.ps1    # PageRank + 隨機對照（5 seed）
python scripts/dose_response.py     # 劑量反應圖表
python scripts/three_way_table.py   # 三方對照表
```

單一變體手動執行範例：

```powershell
python scripts/prune_items.py --mode score --frac 0.10 --out_dir results/week5/prune10
python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed 0 --aug_side item --fixed_epochs 25 --aug_dir results/week5/prune10 --tag itemprune10
python scripts/summarize_week4.py --ratio 1.0 --prefix week4fe_aug_itemprune10 --control_prefix week4fe_aug_itemonly --a0 1.0 --a1 1.0
python scripts/prune_pagerank.py --drop low --frac 0.50 --out_dir results/week5/prlow50 --verify   # --verify 與 networkx 比對
```

### 3. Week 6 效益圖表

```powershell
python scripts/plot_cost_benefit.py   # → results/week6/cost_benefit.png
python scripts/final_cost_table.py    # → results/week6/cost_benefit_table.md、cost_benefit_dual_axis.png
```

### 4. Week 7 分層、錯誤分析、embedding 視覺化

```powershell
.\scripts\run_week7.ps1               # 訓練 w7 對照組/處理組（seed 0–4，約 45 分鐘）並依序執行下面四支分析
# 也可單獨重跑分析（訓練完成後秒級～數十秒）：
python scripts/popularity_strata.py --by item     # → results/week7/strata_item.md/.csv/.png
python scripts/popularity_strata.py --by user     # → results/week7/strata_user.md/.csv/.png
python scripts/error_analysis.py                  # → results/week7/error_cases.md（候選案例，需人工分類）
python scripts/visualize_embeddings.py            # → embeddings_pca.png、embeddings_tsne.png、embedding_cohesion.md
```

`w7` 標籤的結果檔（`week4fe_aug_w7*`）與主實驗分開存放，不會覆蓋 `week4fe_aug_*`。

### 5. 測試

```powershell
python -m pytest tests -q
```

## 授權

本專案採 MIT License，詳見 [LICENSE](./LICENSE)。
