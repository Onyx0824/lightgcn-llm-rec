# 商業智慧期中報告 — Project Plan（0成本版）

**繳交期限**：11月底前上傳至 ee-class 討論區
**核心題目**：LightGCN baseline + 本地LLM增強（user profiling / item attribute augmentation）+ 去噪/剪枝分析 + 效益分析
**目標**：
1. 讓授課老師（同時是lab指導教授）看到你對暑假讀過的四篇論文（LLMRec、RPP、GAECL、LightGNN）有整合性的理解，而不只是重現單一篇
2. 產出乾淨的 GitHub repo，作為未來求職（推薦系統/資料探勘方向）的作品集

---

## 硬體規格
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU（約8GB VRAM）
- RAM: 16GB
- 結論：可本地跑 LightGCN 訓練 + 7B等級量化LLM推論，0金錢成本

---

## 技術棧（0成本）

| 用途 | 工具 | 備註 |
|---|---|---|
| GNN 推薦模型 | PyTorch + (可選) torch-geometric | 自己實作 LightGCN，理解每一行 |
| 生成 user/item 屬性文字 | Ollama + Qwen2.5-7B-Instruct 或 Llama3-8B-Instruct（量化版） | 本地跑，無API費用；若太慢可備案用 Groq/Gemini 免費額度 |
| 文字轉 embedding | sentence-transformers（`all-MiniLM-L6-v2` 或中文 `bge-small-zh`） | 免費開源 |
| 去噪/剪枝分析 | 自行實作簡化版 PageRank centrality 或 embedding pruning | 參考 GAECL / LightGNN 的做法，不需完整重現 |
| 版本控制 | GitHub | 從第一週就建立 repo，養成 commit 習慣 |

---

## Timeline（9/22 – 11/30，共10週）

### 第1週（9/22–9/28）：環境建置 + 題目確認
- [ ] 跟指導教授確認題目方向、避開lab既有研究撞題
- [ ] 安裝 Python venv、PyTorch（CUDA版）、torch-geometric（如需要）
- [ ] 安裝 Ollama，下載並測試 Qwen2.5-7B-Instruct 或 Llama3-8B-Instruct 量化版推論速度
- [ ] 安裝 sentence-transformers，測試 embedding 生成
- [ ] 建立 GitHub repo
- [ ] 選定資料集（user數1-2萬、item數1萬以內為佳，控制LLM生成規模）

### 第2週（9/29–10/5）：Baseline
- [ ] 資料前處理、train/valid/test split
- [ ] 自行實作 LightGCN baseline
- [ ] 確認在4060上訓練速度無虞
- [ ] 記錄 Recall@20、NDCG@20 作為對照基準

### 第3–4週（10/6–10/19）：本地LLM增強模組（風險最高，優先驗證）
- [ ] 設計 user profiling prompt、item attribute prompt（參考 LLMRec Fig 2b/c 架構）
- [ ] 小規模測試（50–100 個 user）：人工檢查輸出品質與格式穩定性
- [ ] 調整 prompt（加 few-shot 範例），確保開源小模型輸出格式穩定
- [ ] 寫批次處理腳本，加上重試機制
- [ ] 全量生成，控制時間預算
- [ ] 將生成文字轉 embedding，接入 LightGCN（embedding 疊加到 ID embedding 上）
- [ ] 記錄增強後效果，與 baseline 比較

### 第5週（10/20–10/26）：去噪/剪枝分析
- [ ] 二選一：
  - 選項A：仿 GAECL，用 PageRank centrality 評估 node 重要性，測試移除不重要增強特徵的效果
  - 選項B：仿 LightGNN，對增強後模型做簡單剪枝，觀察效果與效率的權衡
- [ ] 目標敘事：「LLM增強不是越多越好，過量或低品質資訊會引入噪音」

### 第6週（10/27–11/2）：效益分析（以運算時間取代金錢成本）
- [ ] 記錄本地GPU訓練/推論時間
- [ ] 消融實驗：增強比例（20%/50%/100%）對效果 vs. 運算時間的影響曲線
- [ ] 繪製成本效益折線圖

### 第7週（11/3–11/9）：補充實驗與視覺化（緩衝週）
- [ ] 錯誤分析：挑幾個預測錯誤的 case，分析原因（稀疏/噪音/LLM幻覺）
- [ ] Embedding 視覺化（t-SNE / PCA）
- [ ] 若前面進度落後，本週用於補進度

### 第8週（11/10–11/16）：撰寫報告
- [ ] 報告結構：
  1. 商業問題：推薦系統稀疏/冷啟動對企業的影響
  2. 方法：baseline → LLM增強 → 去噪/剪枝分析
  3. 發現：效果提升幅度、運算成本、去噪後的變化
  4. 具體建議：什麼情境下值得用LLM增強，什麼情境不值得
- [ ] 整理圖表（效益曲線、embedding視覺化、消融表格）

### 第9週（11/17–11/23）：程式碼與 repo 整理
- [ ] 清理程式碼、撰寫 README（資料集說明、方法、重現步驟、主要結果表格）
- [ ] Push 到 GitHub，作為求職作品集連結
- [ ] 找同學或朋友 review 報告與 repo

### 第10週（11/24–11/30）：最終檢查與繳交
- [ ] 緩衝時間，處理最後修正
- [ ] 上傳至 ee-class 討論區

---

## 資料集候選
- Amazon Review 某品類子集（避開 LLMRec/GAECL 已用過的品類）
- MovieLens-1M
- 中文公開評論/推薦資料集（若想增加辨識度，搭配中文LLM如Qwen效果不錯）

---

## 風險提醒
第3–4週（本地LLM生成）是整個計畫風險最高的環節：推論速度與輸出格式穩定性都是未知數。務必在該週一開始就先小規模測試，及早發現問題以保留緩衝時間。
