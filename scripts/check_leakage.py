"""
檢查 LLM user profiling 的輸入歷史（history）有沒有用到「train 切分點之後」的互動。

為什麼要查：
- Week 3 pilot 的 sample_users 是從原始 ratings.dat 取「該 user 最近 15 筆評分」(依 timestamp 由新到舊)，
  並沒有限制在 train.csv 內。
- 資料是依「全域時間」切 70/10/20，所以 user 最近的評分很可能落在 valid/test 區間。
  若 Week 4 沿用同樣邏輯，LLM 看到的就是 test 期間的偏好 → 增強特徵可能洩漏 test 標籤。

做法：把每位 user 的 history 片名對回 item index，統計它們出現在 train / valid / test /
「不在 train」的比例。history 若只來自 train，「在 train 的比例」應為 100%。

用法（專案根目錄）：
    python scripts/check_leakage.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def user_item_sets(df):
    d = defaultdict(set)
    for u, i in zip(df["user"].values, df["item"].values):
        d[int(u)].add(int(i))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="results/week4")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--raw_dir", default="data/raw/ml-1m")
    args = ap.parse_args()

    gen_dir, proc, raw = Path(args.gen_dir), Path(args.processed_dir), Path(args.raw_dir)
    rows = read_jsonl(gen_dir / "users.jsonl")
    if not rows:
        print("users.jsonl 是空的")
        return
    if "history" not in rows[0]:
        print("users.jsonl 的紀錄沒有 history 欄位，無法直接檢查。現有欄位：", sorted(rows[0].keys()))
        print("請把 run_week4_generate.py 貼出來，檢查它如何組 user history。")
        return

    movies = pd.read_csv(raw / "movies.dat", sep="::", engine="python",
                         names=["item_raw", "title", "genres"], encoding="latin-1")
    title2raw = {}
    for r, t in zip(movies["item_raw"], movies["title"]):
        title2raw.setdefault(t, int(r))
    with open(proc / "id_maps.json", "r", encoding="utf-8") as f:
        item2idx = {int(k): v for k, v in json.load(f)["item2idx"].items()}

    tr = user_item_sets(pd.read_csv(proc / "train.csv"))
    va = user_item_sets(pd.read_csv(proc / "valid.csv"))
    te = user_item_sets(pd.read_csv(proc / "test.csv"))

    def blank():
        return {"n_users": 0, "hist": 0, "train": 0, "valid": 0, "test": 0, "users_with_nontrain": 0}

    agg = {"全部已生成 user": blank(), "有 test 互動的 user": blank()}
    for r in rows:
        u = r["user_idx"]
        idxs = []
        for h in r["history"]:
            raw_id = title2raw.get(h["title"])
            idx = item2idx.get(raw_id) if raw_id is not None else None
            if idx is not None:
                idxs.append(idx)
        if not idxs:
            continue
        n_tr = sum(i in tr[u] for i in idxs)
        n_va = sum(i in va[u] for i in idxs)
        n_te = sum(i in te[u] for i in idxs)
        groups = ["全部已生成 user"] + (["有 test 互動的 user"] if te.get(u) else [])
        for g in groups:
            a = agg[g]
            a["n_users"] += 1
            a["hist"] += len(idxs)
            a["train"] += n_tr
            a["valid"] += n_va
            a["test"] += n_te
            a["users_with_nontrain"] += int(n_tr < len(idxs))

    print("history 片名對回 item index 後，各去向的比例（以 history 筆數計）：")
    for g, a in agg.items():
        if a["n_users"] == 0:
            print(f"\n[{g}] 沒有可比對的 user")
            continue
        h = a["hist"]
        nt = h - a["train"]
        print(f"\n[{g}] user 數={a['n_users']}，history 總筆數={h}")
        print(f"  在 train：{a['train'] / h:.1%}")
        print(f"  在 valid：{a['valid'] / h:.1%}")
        print(f"  在 test ：{a['test'] / h:.1%}")
        print(f"  不在 train（切分點之後的評分，含被冷啟動過濾者）：{nt / h:.1%}")
        print(f"  至少有 1 筆不在 train 的 user：{a['users_with_nontrain'] / a['n_users']:.1%}")
    print("\n判讀：history 若只來自 train，『在 train』應為 100%；"
          "有 test 互動的 user 若有相當比例落在 valid/test，代表 user 增強特徵含未來資訊（資料洩漏）。")


if __name__ == "__main__":
    main()
