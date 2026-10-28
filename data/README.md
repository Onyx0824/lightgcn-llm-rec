# 資料集

## MovieLens-1M
- 來源：https://grouplens.org/datasets/movielens/1m/
- 下載：從上面網址下載 `ml-1m.zip`，解壓縮後把 `movies.dat`、`ratings.dat`、`users.dat` 放到 `data/raw/ml-1m/`
  （PowerShell 範例：`Invoke-WebRequest https://files.grouplens.org/datasets/movielens/ml-1m.zip -OutFile ml-1m.zip; Expand-Archive ml-1m.zip data/raw/`，解壓後資料夾名稱為 `ml-1m`）
- 內容：movies.dat, ratings.dat, users.dat（分隔符號為 `::`，編碼 latin-1）
- 統計：6,040 users、3,706 movies、1,000,209 ratings；過濾互動 <5 次後為 6,040 users / 3,416 items / 999,611 interactions
- 前處理輸出：`python src/data_processing/preprocess.py` 會產生 `data/processed/`（train/valid/test.csv、id_maps.json、stats.json）
- `data/raw/` 與 `data/processed/` 不進版控（見 `.gitignore`）
