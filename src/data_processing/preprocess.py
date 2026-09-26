"""
Week 2 - 資料前處理
讀入 MovieLens-1M 原始互動資料，過濾低頻 user/item，建立連續整數 index 映射，
依時間戳排序後切分 train/valid/test，輸出給 LightGCN 訓練使用。

預期輸入格式（MovieLens-1M 標準格式，':: ' 分隔）：
  data/raw/ml-1m/ratings.dat  -> UserID::MovieID::Rating::Timestamp
  data/raw/ml-1m/movies.dat   -> MovieID::Title::Genres
  data/raw/ml-1m/users.dat    -> UserID::Gender::Age::Occupation::Zip-code

輸出：
  data/processed/train.csv / valid.csv / test.csv  (欄位: user, item, timestamp)
  data/processed/id_maps.json                       (原始id <-> 連續index的對照表)
  data/processed/stats.json                          (過濾前後的統計數字，寫報告會用到)
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    """讀入 ratings.dat，回傳 user/item/rating/timestamp 的 DataFrame。"""
    path = raw_dir / "ratings.dat"
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user", "item", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df


def filter_min_interactions(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """
    過濾互動次數過少的 user/item（門檻預設 5 次，對應 project-plan 第2週步驟1）。
    user 和 item 兩邊互相影響（過濾掉某個item後，某些user的互動數可能低於門檻），
    所以用迭代方式直到穩定為止，避免一次過濾不乾淨。
    """
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df["user"].value_counts()
        item_counts = df["item"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[df["user"].isin(valid_users) & df["item"].isin(valid_items)]
    return df.reset_index(drop=True)


def build_id_maps(df: pd.DataFrame) -> tuple[dict, dict]:
    """把原始 user_id / item_id 映射成從 0 開始的連續整數（GNN embedding table需要）。"""
    unique_users = sorted(df["user"].unique())
    unique_items = sorted(df["item"].unique())
    user2idx = {int(u): i for i, u in enumerate(unique_users)}
    item2idx = {int(it): i for i, it in enumerate(unique_items)}
    return user2idx, item2idx


def apply_id_maps(df: pd.DataFrame, user2idx: dict, item2idx: dict) -> pd.DataFrame:
    df = df.copy()
    df["user"] = df["user"].map(user2idx)
    df["item"] = df["item"].map(item2idx)
    return df


def time_based_split(
    df: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    依時間戳排序後切分 70/10/20（project-plan 第2週步驟2）。
    全域排序後直接切分：簡單、可重現，符合 baseline 週的需求。
    之後若要做更嚴謹的 leave-one-out per-user split，可以再擴充。
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    train_df = df_sorted.iloc[:train_end].reset_index(drop=True)
    valid_df = df_sorted.iloc[train_end:valid_end].reset_index(drop=True)
    test_df = df_sorted.iloc[valid_end:].reset_index(drop=True)
    return train_df, valid_df, test_df


def drop_cold_start_from_eval(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    valid/test 中若出現 train 沒看過的 user 或 item（時間切分後的冷啟動），
    LightGCN 這種 transductive 模型無法給出有意義的 embedding，先移除。
    移除的數量會記錄進 stats.json，是報告裡「冷啟動問題」的具體佐證數字。
    """
    train_users = set(train_df["user"])
    train_items = set(train_df["item"])
    valid_clean = valid_df[
        valid_df["user"].isin(train_users) & valid_df["item"].isin(train_items)
    ].reset_index(drop=True)
    test_clean = test_df[
        test_df["user"].isin(train_users) & test_df["item"].isin(train_items)
    ].reset_index(drop=True)
    return valid_clean, test_clean


def main():
    parser = argparse.ArgumentParser(description="MovieLens-1M 前處理 (Week 2 baseline)")
    parser.add_argument("--raw_dir", type=str, default="data/raw/ml-1m")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--min_interactions", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 讀入原始資料：{raw_dir}")
    df = load_ratings(raw_dir)
    n_raw_interactions = len(df)
    n_raw_users = df["user"].nunique()
    n_raw_items = df["item"].nunique()
    print(f"      原始：{n_raw_users} users, {n_raw_items} items, {n_raw_interactions} interactions")

    print(f"[2/5] 過濾互動數 < {args.min_interactions} 的 user/item")
    df = filter_min_interactions(df, args.min_interactions)
    n_filtered_interactions = len(df)
    n_filtered_users = df["user"].nunique()
    n_filtered_items = df["item"].nunique()
    print(f"      過濾後：{n_filtered_users} users, {n_filtered_items} items, {n_filtered_interactions} interactions")

    print("[3/5] 建立 id 映射表")
    user2idx, item2idx = build_id_maps(df)
    df = apply_id_maps(df, user2idx, item2idx)

    print(f"[4/5] 依時間排序切分 train/valid/test ({args.train_ratio}/{args.valid_ratio}/"
          f"{1 - args.train_ratio - args.valid_ratio:.2f})")
    train_df, valid_df, test_df = time_based_split(df, args.train_ratio, args.valid_ratio)
    n_valid_before = len(valid_df)
    n_test_before = len(test_df)
    valid_df, test_df = drop_cold_start_from_eval(train_df, valid_df, test_df)
    n_valid_dropped = n_valid_before - len(valid_df)
    n_test_dropped = n_test_before - len(test_df)
    print(f"      valid 中移除冷啟動: {n_valid_dropped} 筆 / test 中移除冷啟動: {n_test_dropped} 筆")

    print(f"[5/5] 輸出到 {out_dir}")
    train_df[["user", "item", "timestamp"]].to_csv(out_dir / "train.csv", index=False)
    valid_df[["user", "item", "timestamp"]].to_csv(out_dir / "valid.csv", index=False)
    test_df[["user", "item", "timestamp"]].to_csv(out_dir / "test.csv", index=False)

    with open(out_dir / "id_maps.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "user2idx": {str(k): v for k, v in user2idx.items()},
                "item2idx": {str(k): v for k, v in item2idx.items()},
            },
            f,
            ensure_ascii=False,
        )

    stats = {
        "raw": {
            "n_users": int(n_raw_users),
            "n_items": int(n_raw_items),
            "n_interactions": int(n_raw_interactions),
        },
        "filtered": {
            "n_users": int(n_filtered_users),
            "n_items": int(n_filtered_items),
            "n_interactions": int(n_filtered_interactions),
        },
        "split": {
            "n_train": int(len(train_df)),
            "n_valid": int(len(valid_df)),
            "n_test": int(len(test_df)),
            "n_valid_cold_start_dropped": int(n_valid_dropped),
            "n_test_cold_start_dropped": int(n_test_dropped),
        },
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("完成。統計數字已寫入 stats.json（報告的資料集章節可以直接引用）。")


if __name__ == "__main__":
    main()
