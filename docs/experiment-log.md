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
- 實際訓練 55 epoch 後 early stop，最佳 valid recall@20 落在 **epoch 30**
- 訓練耗時：每epoch約8.7–9.2秒，總計約8分鐘（符合checkpoint「分鐘等級」要求）
- **Test set：Recall@20 = 0.0814，NDCG@20 = 0.2721**
- 註：初次訓練誤把valid正樣本漏掉沒排除在test候選排名外，修正評估邏輯（test評估時改為排除train∪valid已互動項目）後重新訓練，數字由 Recall@20=0.0740/NDCG@20=0.2535 提升至上述修正後結果，此為最終正確版本

**觀察/決策**
- Valid recall@20 在 epoch30 後開始震盪、不再進步，train loss 仍持續下降 → 典型過擬合訊號，early stopping 正確地保留了 epoch30 的權重而非最終權重
- 時間切分導致 valid/test 冷啟動比例偏高（57–74%），推測原因：ML-1M 使用者常「一次性」評分一大批電影，較晚加入互動的user整批落在valid/test，訓練集完全沒看過 → 這點可以直接寫進報告「稀疏/冷啟動」商業問題的實證段落
- 評估時必須同時排除 train 與 valid 的已互動項目才能正確計算 test 指標，否則 valid 正樣本會佔掉 test top-K 名額、變相低估指標——這點若第4-5週要重跑evaluate要記得延用同樣邏輯
- 此結果為第4週LLM增強效果的比較基準（baseline）

---

## Week 3（10/6–10/12）LLM增強 — Prompt設計與小規模驗證

**完成項目**
- [x] User profiling prompt / item attribute prompt 設計（JSON輸出格式 + 1個few-shot範例）
- [x] 小規模驗證：n_users=100, n_items=100（`src/llm_augmentation/run_week3_pilot.py`）
- [x] Parser + retry機制（`src/llm_augmentation/ollama_client.py`）
- [x] 依驗證結果修正prompt（rating_tendency加值域約束、system prompt補繁體中文強制指示）

**第一輪驗證結果（修正前，n=100/100）**

| 任務 | 首次成功率 | Retry後成功率 | 平均耗時/筆 |
|---|---|---|---|
| User profiling | 100% | 100% | 4.33秒 |
| Item attribute | 99% | 100% | 3.64秒 |

格式穩定率（欄位是否齊全）遠超90%門檻，但人工檢查發現「欄位有填、但值不合理/不受控」的問題，被原本的成功率定義蓋掉了，因此又做了第二輪分析：

**內容合理性問題（第一輪，n=100 user profiles）**
1. `guessed_age_group`：99/100 都填`"unknown"`，只有1筆願意猜。代表這個欄位對後續embedding貢獻有限，也無法拿真實`users.dat`的Age欄位做交叉驗證——考慮Week4整合時是否保留這個欄位。
2. `rating_tendency`：雖然prompt/few-shot只示範`generous`一種用詞，實際卻跑出9種不同用詞（generous 75、moderate 7、balanced 4、mixed 4、cautious 3、neutral 2、conservative 2、critical 2、average 1），而原本的成功率算法只檢查欄位存在、不檢查值是否合法，沒能抓出這個格式漂移。
3. 用歷史評分平均值反查`rating_tendency`是否合理：8/100筆標成`"generous"`（寬容）但實際平均評分低於3分（最低到2.4分），顯示模型偶爾被少數幾筆評分帶偏、沒有正確權衡全部歷史。

**內容合理性問題（第一輪，n=100 item attributes）**
1. 格式面：100%成功（99首次+1次retry）。
2. `target_audience`為自由文字欄位，100筆跑出48種不同用詞（例如「成年觀眾」「成人」「成人觀眾」語意重複但字面不同），若之後要拿這欄位做類別特徵需要額外正規化/分群。
3. `supplementary_genres`：89/100為空陣列，代表ML-1M原生的18類標籤已經相當完整，LLM能補充的新資訊有限（少數補到的如"Family"、"Cyberpunk"算合理）。
4. 抓到1筆輸出出現簡體字（"时尚"），原因是原本的system prompt沒有像Week1 `test_ollama.py`一樣明確要求繁體中文。

**修正動作**
- `prompts.py`：`rating_tendency`欄位在system prompt中明確限制為`generous`/`critical`/`neutral`三選一，並提示模型要通盤考量全部評分紀錄再判斷；user/item兩支system prompt都補上「請全程使用繁體中文回答，絕對不可出現任何簡體字」。
- `ollama_client.py`：`query_with_retry`新增`enum_checks`參數，成功判定除了「必要欄位齊全」外，再加一層「指定欄位的值必須落在允許集合內」，值不合法會觸發retry並在prompt中點名是哪個欄位、允許值有哪些。

**第二輪驗證結果（修正後，重新跑n_users=100/n_items=100）**

| 任務 | 首次成功率 | Retry後成功率 | 平均耗時/筆 |
|---|---|---|---|
| User profiling | 100% | 100% | 4.19秒 |
| Item attribute | 100% | 100% | 3.61秒 |

修正效果驗證：
1. **`rating_tendency`值域收斂成功**：100筆全部落在`generous`/`critical`/`neutral`三選一內（neutral 54、generous 24、critical 22），不再出現moderate/mixed/cautious等9種漂移用詞，`enum_checks`確實有效。
2. **與實際平均評分的矛盾筆數大幅下降**：8/100 → **2/100**（且這2筆都是`critical`對到平均分3.53，非常貼近3.5的判定門檻，可視為邊界案例而非明顯誤判），比第一輪的「標generous但平均只有2.4分」這種明顯矛盾好上不少。
3. **繁體中文指示仍有極少數殘留**：user側1/100筆（`profile_summary`混雜簡體字，如「类型」「悬疑」「评价」）、item側2/100筆（如「Song of the South」的`时光`/`爱好者`；「Art of War」甚至出現亂碼替代字元`�`，疑似模型生成了非常規字元）。強制指示把問題從沒設下限壓到約1-2%，但沒有完全消除——如果後續要做全量生成，可以在parser裡加一層簡體字偵測+自動剔除該筆重試（作法類似`enum_checks`），或接受這個殘留率當作已知噪音來源。
4. **`guessed_age_group`（98/100 unknown）與`target_audience`高分歧（41種不同用詞）、`supplementary_genres`（89/100空）這三個第一輪就觀察到的模式，這輪沒有特別去修，結果也維持相近水準**，符合預期（這輪只鎖定rating_tendency和繁體中文兩個問題修正）。

**觀察/決策**
- 「格式穩定率」若只檢查JSON能不能parse+欄位是否存在，並不足以代表輸出品質，值域是否合法要另外檢查——這點會寫進報告的方法論限制/反思段落。
- `guessed_age_group`低利用率、`supplementary_genres`低命中率，這兩點可以直接呼應報告的核心敘事「LLM增強不是每個欄位都有效，過量/不精準的欄位設計本身就是一種噪音來源」，跟第5週去噪分析的主題互相呼應。
- 第二輪驗證後，格式穩定率與內容合理性都已達標（rating_tendency矛盾率8%→2%、值域100%收斂），殘留的1-2%簡體字/亂碼問題規模很小，決定不再花時間在Week3打磨，**帶著已知的殘留噪音進入Week4全量生成**，這個殘留率本身也可以寫進報告當作「即使有值域約束，開源小模型仍有低機率的格式/語言漂移」的量化佐證。

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
