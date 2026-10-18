# LightGCN + 本地LLM增強推薦系統：去噪與效益分析

> 中央大學資管所 碩一上「商業智慧」期中報告專案
> 以 0 成本、全本地運算的方式，探討 LLM 生成的輔助特徵能否提升推薦系統效果，以及過量/低品質增強資訊帶來的噪音問題。

## 專案背景

推薦系統長期面臨資料稀疏與冷啟動問題。近年研究（LLMRec、RPP、GAECL、LightGNN）嘗試用 LLM 生成的 user/item 屬性文字來補充圖結構資訊，但這類增強並非「越多越好」——低品質或過量的生成內容可能引入噪音，反而傷害效果。

本專案在 **LightGCN baseline** 之上，加入**本地端 LLM 生成的 user profiling / item attribute 增強特徵**，並實作**去噪/剪枝分析**，量化增強帶來的效果提升與其中的噪音比例，最後產出**效益分析**（效果 vs. 運算成本），轉化為具體的商業建議。

## 方法架構

```
Raw Interaction Data
        │
        ▼
 ┌─────────────┐      ┌──────────────────────┐
 │  LightGCN   │◄─────│ LLM 生成的 user/item  │
 │  Baseline   │      │ 屬性文字 → Embedding  │
 └─────────────┘      └──────────────────────┘
        │                       │
        ▼                       ▼
   Recall@20 /            去噪/剪枝分析
   NDCG@20 對照          (格式過濾、embedding
                          語意離群剪枝)
        │                       │
        └───────────┬───────────┘
                     ▼
              效益分析（效果 vs. 運算時間）
                     ▼
              商業建議：何時值得用 LLM 增強
```

## 技術棧（全程 0 成本，本地運算）

| 用途 | 工具 |
|---|---|
| GNN 推薦模型 | PyTorch（自行實作 LightGCN） |
| LLM 推論 | Ollama + `qwen2.5:7b-instruct-q4_K_M`（量化版） |
| 文字轉 Embedding | sentence-transformers（`BAAI/bge-small-zh-v1.5`，512 維） |
| 去噪/剪枝 | 格式過濾（`--drop_dirty` / `--drop_retry`）、embedding 語意離群剪枝（`scripts/prune_items.py`） |
| 硬體 | RTX 4060 Laptop（8GB VRAM）／16GB RAM |

## 專案結構

```
.
├── data/                   # 資料集與前處理輸出
│   └── README.md           # 說明資料集來源、下載方式
├── src/
│   ├── data_processing/    # 讀取、過濾、train/valid/test split
│   ├── model/               # LightGCN 實作
│   ├── llm_augmentation/    # Prompt 設計、批次生成、embedding 轉換（build_embeddings.py）
│   ├── analysis/            # 去噪/剪枝、PageRank、t-SNE 視覺化
│   └── evaluation/          # Recall@20、NDCG@20
├── notebooks/               # 探索性分析、圖表產出
├── configs/                 # 超參數設定檔（yaml/json），方便重現實驗
├── results/                 # 實驗結果表格與圖表（進版控，體積小；*.npy 不進版控）
├── scripts/
│   ├── train_baseline.py    # 訓練 LightGCN 控制組 (Baseline) 模型與評估
│   ├── train_augmented.py   # 執行加入本地 LLM 增強特徵的 LightGCN 訓練與評估
│   ├── compare_ratios.py    # 比較不同增強比例（如 20%、50%、100%）與雙端/單端的效果差異
│   ├── summarize_week4.py   # 配對檢定彙整：任兩組結果（--prefix / --control_prefix）的差異
│   ├── inspect_dirty.py     # 盤點 dirty items 由哪些字元觸發（Week 5）
│   ├── three_way_table.py   # Week 5 三種去噪做法並排對照表
│   ├── prune_items.py       # embedding 語意離群剪枝 / 隨機剔除對照（Week 5）
│   ├── prune_pagerank.py    # 二部圖 PageRank 計算與依 PageRank 剔除 item 增強（Week 5）
│   ├── dose_response.py     # 劑量反應：被剔除互動比例 vs 保留的增益（Week 5）
│   ├── run_week5_denoise.ps1  # Week 5 格式過濾去噪（drop_dirty、drop_retry）一鍵執行
│   ├── run_week5_prune.ps1    # Week 5 語意剪枝與隨機對照一鍵執行
│   └── run_week5_pagerank.ps1 # Week 5 PageRank 篩選與 rand30/50/80/90 對照一鍵執行
├── checkpoints/             # 訓練好的模型權重
├── docs/                    # 規劃文件、報告草稿、架構圖
│   ├── project-plan-detailed.md
│   └── experiment-log.md    # 每週實驗紀錄與完整統計
├── tests/                   # 基本單元測試
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## 環境安裝

```bash
# 1. 建立虛擬環境（Python 3.11）
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# 2. 安裝 PyTorch（CUDA 版）
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. 安裝其餘套件
pip install -r requirements.txt

# 4. 安裝 Ollama，下載量化模型
# https://ollama.com
ollama pull qwen2.5:7b-instruct-q4_K_M

# 5. 驗證環境
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 目前進度

- [x] Python 3.11 虛擬環境、PyTorch（CUDA 12.6）、Ollama、sentence-transformers 環境建置
- [x] 資料集選定（MovieLens-1M）
- [x] LightGCN baseline 實作與訓練（詳見 [`docs/experiment-log.md`](./docs/experiment-log.md)）
- [x] LLM 增強 prompt 設計與小規模驗證（user profiling + item attribute，含 JSON 值域檢查與 retry 機制，n=100/100，詳見 experiment-log.md Week3）
- [x] LLM 增強全量生成（20% / 50% / 100%）→ embedding 整合 → 接入 LightGCN
- [x] 增強效益分析：生成比例、α 敏感度、user 端 vs item 端消融（experiment-log.md Week 4）
- [x] 去噪/剪枝分析（Week 5）：格式過濾（drop_dirty、drop_retry）、embedding 語意離群剪枝與隨機剔除對照，結果為無額外益處；另做 PageRank（item 層級）篩選與劑量反應分析，結果顯示 PageRank 在二部圖上等同熱門度（見下方「主要結果」與 experiment-log.md Week 5）
- [x] 效益分析圖表與最終表格（Week 6：`results/week6/cost_benefit.png`、`results/week6/final_table.md`）
- [ ] item 熱門度分層分析、錯誤分析、embedding 視覺化（Week 7）
- [ ] 報告撰寫

> 完整時程規劃見 [`project-plan-detailed.md`](./docs/project-plan-detailed.md)（9/22–11/30，共 10 週）。

## 實驗設定

- **資料**：MovieLens-1M，過濾互動 <5 次的 user/item 後為 6,040 users / 3,416 items / 999,611 interactions
- **切分**：依時間排序 70/10/20；valid/test 中出現冷啟動 user/item 的互動予以移除；test 評估時排除 train ∪ valid 已互動項目（test 集約 973 位 user）
- **LLM 生成**：Ollama `qwen2.5:7b-instruct-q4_K_M`，輸出繁體中文 JSON，含值域檢查與 retry
- **文字 embedding**：`BAAI/bge-small-zh-v1.5`（512 維）
- **比較方式**：控制組為 α=0（與 baseline 數學等價），固定 25 epoch，依 seed 做配對檢定；主實驗 10 seed（0–9），Week 5 語意剪枝與 PageRank 組 5 seed（0–4）
- **α=1.0 為事前指定**；α 掃描（0.3 / 1.0 / 3.0）僅作敏感度分析，避免用 test 調參
- **統計方法**：每個條件以多個 seed 各訓練一次（主實驗 seed 0–9；Week 5 語意剪枝、PageRank 組與隨機對照為 seed 0–4）；同一 seed 的兩個條件互相配對，對配對差做雙尾 paired t 檢定，並報告配對差的 95% 信賴區間（t 分佈）。p 值只反映訓練隨機性（初始化與抽樣），不代表推廣到其他使用者群體，因為 test 固定為同一批約 973 位 user。完整的配對差、信賴區間與 p 值見 [`results/paired_ci_table.md`](./results/paired_ci_table.md)（由 `scripts/paired_ci_table.py` 產生）。
- **多重比較**：主要結論只依賴下列直接對應研究問題的比較——增強 vs α=0（兩端與只 item，20% / 50% / 100%，共 12 個檢定，Bonferroni 門檻 0.05/12≈0.004）。其餘比較（user-only、α 掃描、Week 5 的各種去噪變體與隨機對照）視為探索性，p 值未經校正，僅作方向性證據。邊緣顯著的結果（例如兩端 20% 的 NDCG，p=0.028）不視為主要發現。

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

三種去噪做法（格式過濾／語意剪枝／PageRank）在相近剔除量下的並排對照見 [`results/week5/three_way_comparison.md`](./results/week5/three_way_comparison.md)（`scripts/three_way_table.py`）。

（rand30 等其餘變體與配對比較見 [`docs/experiment-log.md`](./docs/experiment-log.md) Week 5 (14)(15)；劑量反應圖見 `results/week5/dose_response_ndcg20.png`、`dose_response_recall20.png`。）

### 關鍵發現

1. **增益主要來自 item 端**：item-only（100%）Recall +0.0027（10/10 seed 為正，p<0.001）；user-only 不顯著（Recall +0.0003，p=0.23）。
2. **Recall 在 50% 即飽和，NDCG 到 100% 仍上升**：50%→100% Recall +0.0003（p=0.41）、NDCG +0.0022（p=0.0013）。
3. **成本效益**：item 端 50% 累計生成約 1.9 小時，為兩端 100%（9.9 小時）的 19%，卻取得約 88% 的 Recall 增益。
4. **α 存在平台區**：α 在 0.3–1.0 表現穩定；α=3.0 過大，NDCG 增益消失。
5. **LLM 輸出噪音低**（100% 生成）：dirty users 7 / items 72（items 的 2.1%）；retry>1 users 155 / items 342（items 的 10.0%）。
6. **去噪沒有額外益處**：格式過濾與語意剪枝都沒有提升 Recall，NDCG 反而小幅下降（−0.0005～−0.0026），連隨機剔除同數量 item 也一樣，與「被增強的 item 變少」一致；語意剪枝在 NDCG 上並未優於隨機剔除（prune20 − rand20：−0.0018）。所有去噪版本相對 α=0 的增益（Recall +3.6%、NDCG +1.3%～+1.4%）都保留。建議直接保留全部增強特徵，不需額外清洗。
7. **PageRank 篩選等同熱門度篩選；增強效益隨「被增強 item 覆蓋的互動比例」飽和**：二部圖上 PageRank 與 degree 幾乎相同（Spearman 0.9989）。Recall 增益在被增強 item 覆蓋約 48% 以上的互動時完整保留，覆蓋約 20–24% 時剩約一半，約 9–10% 時消失；在覆蓋相近時，剔除高 PageRank item 與隨機剔除沒有可靠差異（prhigh30 vs rand80：Recall p=0.12、NDCG p=0.64），因此 PageRank 沒有帶來超越熱門度的去噪效果。例外是 prhigh50 對 rand90（兩者都動到約 91% 互動，前者較好），原因未釐清。**成本效益（推算）**：只對熱門度／PageRank 前 50% 的 item 生成（prlow50）約可省一半 item 端生成量，Recall 不變、NDCG 約低 0.4%（p=0.064）；相同生成量下隨機選 50% 的 NDCG 低 1.2%（p<0.001）。
8. **限制**：相對增益約 +2%～+5%（絕對值 +0.0003～+0.0048），test 僅約 973 位 user；p 值只反映訓練隨機性；語意剪枝與 PageRank 組僅 5 seed、隨機對照只用一份剔除名單；PageRank 只做 item 層級篩選（未改圖傳播權重）；噪音只以「格式」與「類型內語意一致性」兩種操作型定義檢驗。結論請以配對差與 95% CI（`results/paired_ci_table.md`）為準，不宜過度外推。

完整表格見 `results/week6_tier_table.csv`，每週詳細統計見 [`docs/experiment-log.md`](./docs/experiment-log.md)。

## 重現實驗

在專案根目錄、已啟動虛擬環境後（以下為 item 端 100%、α=1.0、固定 25 epoch 的範例）：

```powershell
# 1. 產生去噪版 embedding（輸出到 results/week5/<name>/，不覆蓋 results/week4）
python src/llm_augmentation/build_embeddings.py --drop_dirty --out_dir results/week5/drop_dirty
python scripts/prune_items.py --mode score --frac 0.10 --out_dir results/week5/prune10

# 2. 訓練（--tag 決定結果檔名，--aug_dir 指向去噪版 embedding）
python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed 0 --aug_side item --fixed_epochs 25 --aug_dir results/week5/prune10 --tag itemprune10

# 3. 與未過濾 item-only 做配對比較
python scripts/summarize_week4.py --ratio 1.0 --prefix week4fe_aug_itemprune10 --control_prefix week4fe_aug_itemonly --a0 1.0 --a1 1.0
```

PageRank 組與劑量反應：

```powershell
python scripts/prune_pagerank.py --drop low  --frac 0.50 --out_dir results/week5/prlow50 --verify   # --verify 需 pip install networkx
python scripts/prune_items.py --mode random --frac 0.50 --out_dir results/week5/rand50 --seed 0
python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed 0 --aug_side item --fixed_epochs 25 --aug_dir results/week5/prlow50 --tag itemprlow50
python scripts/dose_response.py                      # NDCG@20（輸出 results/week5/dose_response_ndcg20.csv / .png）
python scripts/dose_response.py --metric recall@20
```

整套流程可直接用 `scripts/run_week5_denoise.ps1`、`scripts/run_week5_prune.ps1` 與 `scripts/run_week5_pagerank.ps1`（Windows PowerShell；已完成的結果檔會自動跳過；若被執行原則擋住，先 `Unblock-File` 該檔）。

## 授權

本專案採 MIT License，詳見 [LICENSE](./LICENSE)。
