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
   NDCG@20 對照          (PageRank centrality
                          或 embedding pruning)
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
| LLM 推論 | Ollama + Qwen2.5-7B-Instruct / Llama3-8B-Instruct（量化版） |
| 文字轉 Embedding | sentence-transformers（`all-MiniLM-L6-v2` / `bge-small-zh`） |
| 去噪/剪枝 | 自行實作 PageRank centrality 或 embedding pruning |
| 硬體 | RTX 4060 Laptop（8GB VRAM）／16GB RAM |

## 專案結構

```
.
├── data/                   # 資料集與前處理輸出（不進版控，.gitignore 排除）
│   └── README.md           # 說明資料集來源、下載方式
├── src/
│   ├── data_processing/    # 讀取、過濾、train/valid/test split
│   ├── model/               # LightGCN 實作
│   ├── llm_augmentation/    # Prompt 設計、批次生成、embedding 轉換
│   ├── analysis/            # 去噪/剪枝、PageRank、t-SNE 視覺化
│   └── evaluation/          # Recall@20、NDCG@20
├── notebooks/               # 探索性分析、圖表產出
├── configs/                 # 超參數設定檔（yaml/json），方便重現實驗
├── results/                 # 實驗結果表格與圖表（進版控，體積小）
├── checkpoints/             # 訓練好的模型權重（不進版控，體積大）
├── docs/                    # 規劃文件、報告草稿、架構圖
│   └── project-plan-detailed.md
├── tests/                   # 基本單元測試（選配，非必要但加分）
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

- [x] Python 3.11 虛擬環境建立完成
- [x] PyTorch（CUDA 12.6）安裝並驗證 GPU 可用（RTX 4060 Laptop）
- [x] Ollama 安裝與量化模型推論測試
- [x] sentence-transformers 安裝與測試
- [x] 資料集選定（MovieLens-1M）
- [x] LightGCN baseline 實作與訓練（詳見 [`docs/experiment-log.md`](./docs/experiment-log.md)）
- [ ] LLM 增強模組（prompt 設計 → 全量生成 → embedding 整合）
- [ ] 去噪/剪枝分析
- [ ] 效益分析
- [ ] 報告撰寫

> 完整時程規劃見 [`project-plan-detailed.md`](./docs/project-plan-detailed.md)（9/22–11/30，共 10 週）。

## 主要結果

| 模型 | Recall@20 | NDCG@20 | 備註 |
|---|---|---|---|
| LightGCN (baseline) | 0.0740 | 0.2535 | epoch20 early stop（Week 2） |
| + LLM 增強（未去噪） | TBD | TBD | |
| + LLM 增強（去噪後） | TBD | TBD | |

## 授權

本專案採 MIT License，詳見 [LICENSE](./LICENSE)。
