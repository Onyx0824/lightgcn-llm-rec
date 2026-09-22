# 商業智慧期中報告 — Project Plan（0成本版，詳細步驟）

**繳交期限**：11月底前上傳至 ee-class 討論區
**核心題目**：LightGCN baseline（圖神經網路，資料探勘核心）+ 本地LLM增強（推論，輔助特徵工程）+ 去噪/剪枝分析 + 效益分析
**敘事重心**：模型學到了什麼、增強前後偏好預測的差異、資料中的噪音模式、對應的商業建議 —— 不是工程pipeline細節

---

## 硬體與技術棧
- GPU: RTX 4060 Laptop（約8GB VRAM）／RAM: 16GB
- LLM推論：Ollama + Qwen2.5-7B-Instruct 或 Llama3-8B-Instruct（GGUF 4-bit量化版）
- Embedding：sentence-transformers（`all-MiniLM-L6-v2` 或中文 `bge-small-zh`）
- GNN：PyTorch（+ torch-geometric 可選）
- 版控：GitHub（第一週建立）

---

## 第1週（9/22–9/28）：環境建置 + 題目確認

1. 跟指導教授簡短確認題目方向（LightGCN + LLM增強 + 去噪分析），問清楚lab有沒有類似進行中的研究要避開
2. 建立 Python 虛擬環境（`python -m venv venv`），安裝 PyTorch（CUDA版，用 `torch.cuda.is_available()` 確認能吃到4060）
3. 安裝 Ollama（Windows版），下載 `qwen2.5:7b-instruct-q4_K_M`（或 Llama3-8B 對應量化版）
4. 跑一個最小測試：用Ollama CLI輸入一句簡單prompt，確認能正常輸出、記下大約推論延遲（每次生成幾秒）
5. 用工作管理員/`nvidia-smi` 監控跑Ollama時的VRAM佔用，確認在8GB內
6. 安裝 `sentence-transformers`，跑一個測試句子確認能正常產生embedding向量
7. 建立GitHub repo，設定 `.gitignore`（排除資料集、模型checkpoint、venv），寫第一版空的README骨架
8. 選定資料集：
   - 下載候選資料集（MovieLens-1M 或 Amazon Review 某子品類）
   - 檢查user數、item數、互動數是否落在建議範圍（user 1-2萬、item 1萬以內）
   - 確認資料集裡有可用的側邊資訊（title、genre、review text等），這是LLM要用來生成profile/attribute的原料
9. 週末checkpoint：環境是否都跑得動？VRAM是否夠？若Ollama太慢或OOM，先評估要不要換更小的模型（如3B級）或切到Groq/Gemini免費API備案

---

## 第2週（9/29–10/5）：Baseline

1. 資料前處理腳本：
   - 讀入 raw interaction 資料，轉成 user-item稀疏矩陣或edge list
   - 過濾掉互動次數過少的user/item（常見門檻是至少5次互動）
   - 建立 user_id / item_id 到連續整數index的映射表（GNN輸入需要）
2. Train/valid/test split（依時間戳排序後切分，或用常見的比例如70/10/20）
3. 實作 LightGCN：
   - 初始化 user/item embedding table
   - 實作多層鄰居聚合（不含非線性轉換，這是LightGCN的核心簡化）
   - 實作 BPR loss（正樣本與負樣本抽樣）
4. 寫訓練迴圈：batch抽樣、forward、backward、optimizer step，加上簡單的 early stopping（依validation指標）
5. 實作評估函式：Recall@20、NDCG@20（可以先用top-K ranking的簡化版本，不用一開始就做全量all-ranking）
6. 訓練並記錄baseline結果，存成表格（之後要跟增強版比較）
7. checkpoint：確認訓練時間合理（單張4060訓練LightGCN在這個資料規模下通常是分鐘等級，不應該到小時等級——若太慢要檢查是不是資料結構效率問題）

---

## 第3週（10/6–10/12）：LLM增強 — Prompt設計與小規模驗證

1. 設計兩支prompt（可參考LLMRec論文Fig 2的格式）：
   - User profiling prompt：輸入該user的互動歷史（item title/類別等），輸出結構化的偏好摘要（例如年齡層、偏好類別、不喜歡的類別）
   - Item attribute prompt：輸入item現有資訊，輸出補充屬性（缺失的類別、風格描述等）
2. 在prompt裡加入明確的輸出格式指示（例如「請用JSON格式回答，欄位為...」），這對開源小模型特別重要，避免輸出格式亂跳
3. 抽樣50-100個user手動跑一次，人工檢查：
   - 輸出格式是否穩定（能不能穩定parse成結構化資料）
   - 內容是否合理（不是完全跟輸入無關的幻覺）
4. 根據觀察調整prompt（加入1-2個few-shot範例通常能明顯改善格式穩定性）
5. 寫一個小型的parser + retry機制：輸出格式錯誤時自動重跑一次，記錄失敗率
6. checkpoint：格式穩定率是否夠高（抓一個標準，例如90%以上一次過），若不夠要繼續調prompt，這是本週最重要的產出

---

## 第4週（10/13–10/19）：LLM增強 — 全量生成與整合

1. 全量批次生成：對全部（或抽樣比例，先抓20%/50%/100%三種比例做消融用）user與item跑生成，記錄總耗時
2. 生成結果存檔（JSON或CSV），避免重跑浪費時間
3. 用sentence-transformers把生成文字轉成embedding向量
4. 設計整合方式：把LLM增強的embedding疊加到原本的ID embedding上（可以先用簡單的相加或加權相加，附上normalize，對照LLMRec的做法）
5. 修改訓練pipeline，接入增強特徵，重新訓練模型
6. 記錄「增強後」的Recall@20、NDCG@20，跟baseline比較
7. checkpoint：效果是否有提升？若沒有提升或反而變差，先不要慌——這正好是下週去噪分析要探討的題材，記錄下來

---

## 第5週（10/20–10/26）：去噪/剪枝分析

1. 選定分析方向（擇一）：
   - **PageRank去噪路線**：計算user/item節點在互動圖中的PageRank centrality，依重要性對增強特徵做加權或篩選（重要性低的節點增強特徵權重調低/捨棄）
   - **剪枝路線**：對增強後的embedding或edge做簡單剪枝（例如捨棄絕對值最小的一定比例維度或邊），觀察效果與效率變化
2. 實作選定方法，設定2-3組不同強度的參數做對照
3. 重新訓練、記錄各組結果
4. 整理成對照表：無增強 / 增強無去噪 / 增強+去噪，三者的效果比較
5. 分析：去噪後效果是否比純增強更好？這代表原始增強資料中有多少雜訊
6. checkpoint：這週的分析結果是報告裡最核心的「洞見」段落，要確保有具體數字支撐敘事

---

## 第6週（10/27–11/2）：效益分析

1. 彙整第4週三種生成比例（20%/50%/100%）的：
   - 效果指標（Recall/NDCG提升幅度）
   - 本地生成耗時（GPU時間）
2. 畫出「增強比例 vs. 效果提升」與「增強比例 vs. 耗時」的折線圖，兩張圖疊在一起看效益曲線的轉折點
3. 寫出結論：多少比例是效益甜蜜點（例如50%增強已經拿到大部分效果，但耗時只有全量的一半）
4. 若有做去噪分析，一併呈現「加了去噪後，效益曲線是否更平滑/更划算」
5. checkpoint：確認圖表清楚、有明確的數字結論可以寫進報告的「商業建議」段落

---

## 第7週（11/3–11/9）：補充實驗與視覺化（緩衝週）

1. 錯誤分析：挑5-10個預測錯誤的case，分類原因（資料稀疏、LLM生成幻覺、去噪過度、冷啟動user等）
2. Embedding視覺化：用t-SNE或PCA把user/item embedding投影到2D，畫散點圖，比較增強前後/去噪前後的分布變化
3. 若前面幾週有進度落後，本週優先補齊，視覺化可以縮減
4. checkpoint：確認手上已經有「所有報告需要的圖表與數字」，下週開始寫報告不應該再需要跑新實驗

---

## 第8週（11/10–11/16）：撰寫報告

1. 報告大綱定稿：
   - 商業問題：推薦系統稀疏/冷啟動對企業的影響
   - 方法：baseline → LLM增強 → 去噪/剪枝分析（附架構圖）
   - 實驗設置：資料集、模型、評估指標
   - 發現：效果提升數字、去噪前後比較、效益曲線
   - 商業建議：什麼情境值得用LLM增強、增強比例的甜蜜點、去噪的必要性
   - 限制與未來方向
2. 依大綱把第5-7週的圖表分配進對應段落
3. 初稿完成後，通讀一次確認敘事重心在「挖掘出的知識與建議」而不是工程細節
4. checkpoint：報告初稿完成，留幾天給下週修訂

---

## 第9週（11/17–11/23）：程式碼與 repo 整理

1. 清理程式碼：移除debug用的print、整理成模組化的檔案結構（data processing / model / llm augmentation / analysis / evaluation）
2. 寫 `requirements.txt`（或 `environment.yml`）
3. 撰寫README：專案簡介、資料集說明、方法架構圖、如何重現（含Ollama安裝步驟）、主要結果表格
4. Push到GitHub，確認repo在別人電腦上clone後能照README跑起來（至少檢查一次）
5. 找同學或朋友review報告與repo，收集回饋
6. checkpoint：GitHub repo已經是可以放進履歷的狀態

---

## 第10週（11/24–11/30）：最終檢查與繳交

1. 依review回饋修訂報告
2. 最終檢查報告格式、圖表編號、錯字
3. 確認繳交格式符合ee-class要求
4. 上傳至ee-class討論區

---

## 風險提醒
第3-4週（本地LLM生成）風險最高，格式穩定性與推論速度是主要未知數，第3週一開始就要做小規模驗證，及早發現問題保留緩衝。
