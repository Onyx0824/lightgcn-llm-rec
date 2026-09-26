# 實驗紀錄 (Experiment Log)

> 每週更新一次，對應 `docs/project-plan-detailed.md` 的週次規劃。
> 目的：報告撰寫階段(第8週)不需要回頭翻git history找數字，所有關鍵設定/結果/決策都在這裡。

---

## Week 1（9/22–9/28）環境建置 + 題目確認

**完成項目**
- [x] Python 3.11 虛擬環境建立
- [x] PyTorch (CUDA 12.6) 安裝，確認 GPU 可用（RTX 4060 Laptop）
- [x] Ollama 安裝與量化模型推論測試：`qwen2.5:7b-instruct-q4_K_M`（4.7GB）
- [x] sentence-transformers 測試：`all-MiniLM-L6-v2`
- [x] 資料集選定：MovieLens-1M（`data/raw/ml-1m/`，含 movies.dat / ratings.dat / users.dat）
- [x] GitHub repo 建立

**Ollama 推論測試結果**（`src/llm_augmentation/test_ollama.py`）
- 模型：qwen2.5:7b-instruct-q4_K_M（4.7GB，4-bit量化）
- 第一次執行（含模型載入VRAM時間）：18.5 秒，53.1 tokens/sec
- 第二次執行（模型已常駐VRAM，代表實際推論速度）：7.16 秒，46.7 tokens/sec
- 輸出格式：繁體中文，符合system prompt指示

**VRAM 監控結果**（`nvidia-smi -l 1`，總容量 8,188 MiB）
- 待機：579 MiB
- 模型載入VRAM後：4,832 MiB
- 生成過程峰值：5,211 MiB（約64%，GPU-Util一度達86%）
- 餘裕：約 3GB，第3-4週同時載入sentence-transformers做embedding應該不會超出8GB

**sentence-transformers 測試結果**（`tests/test_embedding.py`）
- 模型：all-MiniLM-L6-v2
- 輸出embedding維度：(384,)，符合預期

**觀察/決策**
- 模型常駐VRAM後的生成速度（46.7 tokens/sec）比首次載入時（53.1，但含載入開銷）更能代表全量生成階段的實際速度，之後估算第3-4週生成總耗時應以此為準
- VRAM峰值5.2GB/8GB，在預算內且有餘裕，暫不需要考慮換更小模型

---

## Week 2（9/29–10/5）Baseline

**完成項目**
- [x] 資料前處理（`src/data_processing/preprocess.py`）
- [x] Train/valid/test 切分（時間排序 70/10/20）
- [x] LightGCN 實作（`src/model/lightgcn.py`）
- [x] 訓練迴圈 + early stopping（`scripts/train_baseline.py`）
- [x] 評估函式 Recall@20 / NDCG@20（`src/evaluation/metrics.py`）
- [x] 訓練並記錄 baseline 結果

**資料集統計**（`data/processed/stats.json`）

| 階段 | users | items | interactions |
|---|---|---|---|
| 原始 | 6,040 | 3,706 | 1,000,209 |
| 過濾後（≥5互動） | 6,040 | 3,416 | 999,611 |

| Split | 筆數 | 因冷啟動移除 |
|---|---|---|
| Train | 699,727 | — |
| Valid | 26,266 | 73,695 (73.7%) |
| Test | 84,762 | 115,161 (57.6%) |

**訓練設定**（`configs/baseline.yaml`）
- embed_dim=64, n_layers=3, lr=0.001, weight_decay=0.0001, batch_size=2048
- Early stopping patience=5（以每5個epoch評估一次為單位）

**訓練結果**
- 實際訓練 45 epoch 後 early stop，最佳 valid recall@20 落在 **epoch 20**
- 訓練耗時：每epoch約8–13秒，總計約6–7分鐘（符合checkpoint「分鐘等級」要求）
- **Test set：Recall@20 = 0.0740，NDCG@20 = 0.2535**

**觀察/決策**
- Valid recall@20 在 epoch20 後開始震盪、不再進步，train loss 仍持續下降 → 典型過擬合訊號，early stopping 正確地保留了 epoch20 的權重而非最終權重
- 時間切分導致 valid/test 冷啟動比例偏高（57–74%），推測原因：ML-1M 使用者常「一次性」評分一大批電影，較晚加入互動的user整批落在valid/test，訓練集完全沒看過 → 這點可以直接寫進報告「稀疏/冷啟動」商業問題的實證段落
- 此結果為第4週LLM增強效果的比較基準（baseline）

---

## Week 3（10/6–10/12）LLM增強 — Prompt設計與小規模驗證

**完成項目**
- [ ]（待填）

**觀察/決策**
- （待填：格式穩定率、few-shot是否有幫助）

---

## Week 4（10/13–10/19）LLM增強 — 全量生成與整合

**完成項目**
- [ ]（待填）

**觀察/決策**
- （待填：增強後 vs baseline 的 Recall@20/NDCG@20 比較）

---

## Week 5（10/20–10/26）去噪/剪枝分析

**完成項目**
- [ ]（待填）

**觀察/決策**
- （待填：無增強 / 增強無去噪 / 增強+去噪 三者對照表）

---

## Week 6（10/27–11/2）效益分析

**完成項目**
- [ ]（待填）

**觀察/決策**
- （待填：增強比例 vs 效果/耗時的甜蜜點）
