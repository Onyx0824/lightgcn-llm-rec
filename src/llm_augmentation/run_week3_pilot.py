"""
Week 3 - 小規模驗證主腳本（對應 project-plan-detailed.md 第3週步驟3-6）

用法：
    python src/llm_augmentation/run_week3_pilot.py --n_users 100 --n_items 100

流程：
    1. 從 data/processed/train.csv + id_maps.json 還原「連續index -> 原始ML-1M id」
    2. 從 data/raw/ml-1m/ 讀 movies.dat（片名/類型）、ratings.dat（真實評分值，train.csv只存timestamp）
    3. 隨機抽樣 N 個 user、N 個 item
    4. 對每個 user 套用 user profiling prompt、每個 item 套用 item attribute prompt
    5. 呼叫 Ollama（含parser+retry），記錄成功率、耗時
    6. 輸出：
        results/week3_pilot_users.json   - 每個user的profile + 是否成功 + 原始輸出
        results/week3_pilot_items.json   - 每個item的attribute + 是否成功 + 原始輸出
        results/week3_pilot_summary.json - 彙總統計（格式穩定率、平均耗時），這是checkpoint要看的數字

    人工檢查（步驟3-4）：
        產出後打開 week3_pilot_users.json / week3_pilot_items.json，
        任選10-20筆：
        - user: 比對 guessed_age_group 跟 data/raw/ml-1m/users.dat 的真實Age是否合理接近
        - item: 檢查 mood_tags/supplementary_genres 是否與片名常識相符、不是幻覺
        若穩定率或內容合理性不理想，回頭調整 prompts.py 的 few-shot 範例後重跑。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ollama_client import query_with_retry
from prompts import (
    ITEM_ATTRIBUTE_ENUM_CHECKS,
    ITEM_ATTRIBUTE_REQUIRED_KEYS,
    ITEM_ATTRIBUTE_SYSTEM,
    USER_PROFILE_ENUM_CHECKS,
    USER_PROFILE_REQUIRED_KEYS,
    USER_PROFILE_SYSTEM,
    build_item_attribute_prompt,
    build_user_profile_prompt,
)


def load_movies(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["item_raw", "title", "genres"],
        encoding="latin-1",
    )
    return df


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_raw", "item_raw", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df


def sample_users(train_df, ratings_df, movies_df, idx2user_raw, idx2item_raw, n, max_history=15, seed=42):
    rng = random.Random(seed)
    user_idxs = train_df["user"].unique().tolist()
    sampled = rng.sample(user_idxs, min(n, len(user_idxs)))

    movies_lookup = movies_df.set_index("item_raw")[["title", "genres"]].to_dict("index")

    out = []
    for u_idx in sampled:
        u_raw = idx2user_raw[u_idx]
        user_ratings = ratings_df[ratings_df["user_raw"] == u_raw].sort_values(
            "timestamp", ascending=False
        )
        history = []
        for _, row in user_ratings.head(max_history).iterrows():
            movie = movies_lookup.get(row["item_raw"])
            if movie is None:
                continue
            history.append(
                {"title": movie["title"], "genres": movie["genres"], "rating": int(row["rating"])}
            )
        if not history:
            continue
        out.append({"user_idx": int(u_idx), "user_raw": int(u_raw), "history": history})
    return out


def sample_items(train_df, movies_df, idx2item_raw, n, seed=42):
    rng = random.Random(seed)
    item_idxs = train_df["item"].unique().tolist()
    sampled = rng.sample(item_idxs, min(n, len(item_idxs)))

    movies_lookup = movies_df.set_index("item_raw")[["title", "genres"]].to_dict("index")

    out = []
    for i_idx in sampled:
        i_raw = idx2item_raw[i_idx]
        movie = movies_lookup.get(i_raw)
        if movie is None:
            continue
        out.append(
            {"item_idx": int(i_idx), "item_raw": int(i_raw), "title": movie["title"], "genres": movie["genres"]}
        )
    return out


def run_users(sampled_users: list[dict]) -> list[dict]:
    results = []
    for i, u in enumerate(sampled_users):
        prompt = build_user_profile_prompt(u["history"])
        result = query_with_retry(
            prompt, USER_PROFILE_SYSTEM, USER_PROFILE_REQUIRED_KEYS, USER_PROFILE_ENUM_CHECKS
        )
        results.append({**u, **result})
        print(f"[user {i+1}/{len(sampled_users)}] success={result['success']} attempts={result['attempts']}")
    return results


def run_items(sampled_items: list[dict]) -> list[dict]:
    results = []
    for i, it in enumerate(sampled_items):
        prompt = build_item_attribute_prompt(it["title"], it["genres"])
        result = query_with_retry(
            prompt, ITEM_ATTRIBUTE_SYSTEM, ITEM_ATTRIBUTE_REQUIRED_KEYS, ITEM_ATTRIBUTE_ENUM_CHECKS
        )
        results.append({**it, **result})
        print(f"[item {i+1}/{len(sampled_items)}] success={result['success']} attempts={result['attempts']}")
    return results


def summarize(users_results, items_results) -> dict:
    def stats(results):
        n = len(results)
        n_success_first_try = sum(1 for r in results if r["success"] and r["attempts"] == 1)
        n_success_total = sum(1 for r in results if r["success"])
        avg_elapsed = sum(r["elapsed_sec_total"] for r in results) / n if n else 0
        return {
            "n": n,
            "first_try_success_rate": round(n_success_first_try / n, 4) if n else None,
            "overall_success_rate_after_retry": round(n_success_total / n, 4) if n else None,
            "avg_elapsed_sec_per_sample": round(avg_elapsed, 2),
        }

    return {"user_profiling": stats(users_results), "item_attribute": stats(items_results)}


def main():
    parser = argparse.ArgumentParser(description="Week 3 小規模驗證")
    parser.add_argument("--raw_dir", type=str, default="data/raw/ml-1m")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--n_users", type=int, default=100)
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 讀入 processed train.csv 與 id_maps.json")
    train_df = pd.read_csv(processed_dir / "train.csv")
    with open(processed_dir / "id_maps.json", "r", encoding="utf-8") as f:
        id_maps = json.load(f)
    idx2user_raw = {v: int(k) for k, v in id_maps["user2idx"].items()}
    idx2item_raw = {v: int(k) for k, v in id_maps["item2idx"].items()}

    print("[2/5] 讀入原始 movies.dat / ratings.dat")
    movies_df = load_movies(raw_dir)
    ratings_df = load_ratings(raw_dir)

    print(f"[3/5] 抽樣 {args.n_users} users / {args.n_items} items")
    sampled_users = sample_users(
        train_df, ratings_df, movies_df, idx2user_raw, idx2item_raw, args.n_users, seed=args.seed
    )
    sampled_items = sample_items(train_df, movies_df, idx2item_raw, args.n_items, seed=args.seed)

    print("[4/5] 呼叫 Ollama 生成（user profiling）")
    users_results = run_users(sampled_users)
    print("[4/5] 呼叫 Ollama 生成（item attribute）")
    items_results = run_items(sampled_items)

    print("[5/5] 輸出結果")
    with open(results_dir / "week3_pilot_users.json", "w", encoding="utf-8") as f:
        json.dump(users_results, f, ensure_ascii=False, indent=2)
    with open(results_dir / "week3_pilot_items.json", "w", encoding="utf-8") as f:
        json.dump(items_results, f, ensure_ascii=False, indent=2)

    summary = summarize(users_results, items_results)
    with open(results_dir / "week3_pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        "\n完成。請打開 results/week3_pilot_users.json 與 week3_pilot_items.json 抽10-20筆人工檢查合理性，"
        "並依 checkpoint 標準（穩定率90%以上）決定是否需要回頭調整 prompts.py 的 few-shot 範例。"
    )


if __name__ == "__main__":
    main()
