"""
Week 4 - 全量批次生成（user profiling + item attribute）

與 Week3 pilot 的差異：
1. user 歷史「只用 train.csv 內的互動」。pilot 是直接拿 ratings.dat 全部資料取最近15筆，
   會把 valid/test 時期的評分餵給 LLM，生成的 profile 帶有未來資訊（資料洩漏）。
2. 全部 user/item 先用固定 seed 洗牌並存成 order.json，20%/50%/100% 就是同一個順序的前綴，
   做消融時不用重跑生成；每筆的耗時都有記錄，Week6 效益曲線直接加總即可。
3. 結果寫成 JSONL（一筆一行、即時 flush），中斷後重跑會自動跳過已完成的筆數。
4. 輸出若含簡體字 / 亂碼字元，自動多重試一次；仍不乾淨就標記 dirty=True（保留，供 Week5 噪音分析）。

用法（建議分階段，每階段結束就能先拿去訓練）：
    cd src/llm_augmentation
    python run_week4_generate.py --max_ratio 0.2
    python run_week4_generate.py --max_ratio 0.5
    python run_week4_generate.py --max_ratio 1.0

可選：先設環境變數 OLLAMA_NUM_PARALLEL=2 再啟動 ollama，並加 --workers 2，嘗試提高吞吐量
（需自行用 nvidia-smi 確認 VRAM 沒爆，且比較 wall time 是否真的下降）。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

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
from run_week3_pilot import load_movies, load_ratings

# 常見「簡體專用字」（非完整清單，只用來抓明顯殘留）
SIMPLIFIED_ONLY = set(
    "类评悬爱时这为们个说对没里画电见观实现点还门问题开关东车长马鸟鱼风飞"
    "与专业严乱习书买乐云产亲亿仅从仓从众优传伤伦传势务动区医华单卖卫发变号叹吗"
    "团园围图图块场声处备复头夹奋妇学宁宝寻导将层岁帮应开张当录态总恶惊惯愿战执"
    "护择据损执档杂极构枪标样欢气汉泪济灭灵点爷牵独猎现电画盖离种积称纪终经结给"
    "绝统继维综绿网罗职联脑与舰艺节苏药获让训议记讲许论设访证评识诉译试话该详语误"
    "说谁调谈请读变谢败账货质购贯资赛较辅达迁运进远连选递边郑钱铁错键镇闭阅队际陆"
    "险隐难预领频题颜风饭饮馆骑验"
)


def has_bad_text(parsed: dict) -> bool:
    text = json.dumps(parsed, ensure_ascii=False)
    for ch in text:
        o = ord(ch)
        if ch == "\ufffd" or ch in SIMPLIFIED_ONLY:
            return True
        if 0x00C0 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF:  # 拉丁擴充（如越南語殘留字元）
            return True
    return False


def gen_with_clean(prompt, system, required, enums) -> dict:
    r = query_with_retry(prompt, system, required, enums)
    dirty = False
    if r["success"] and has_bad_text(r["parsed"]):
        r2 = query_with_retry(prompt, system, required, enums)
        r["attempts"] += r2["attempts"]
        r["elapsed_sec_total"] = round(r["elapsed_sec_total"] + r2["elapsed_sec_total"], 2)
        if r2["success"] and not has_bad_text(r2["parsed"]):
            r["parsed"] = r2["parsed"]
        else:
            dirty = True
    r["dirty"] = dirty
    if r["success"]:
        r.pop("raw_outputs", None)  # 成功的不存原始輸出，節省空間；失敗的保留供檢查
    return r


def build_user_records(train_df, ratings_df, movies_df, idx2user_raw, idx2item_raw, max_history):
    t = train_df.copy()
    t["user_raw"] = t["user"].map(idx2user_raw)
    t["item_raw"] = t["item"].map(idx2item_raw)
    m = t.merge(ratings_df[["user_raw", "item_raw", "rating"]], on=["user_raw", "item_raw"], how="left")
    m = m.merge(movies_df, on="item_raw", how="left").dropna(subset=["title", "rating"])
    m = m.sort_values(["user", "timestamp"], ascending=[True, False])
    out = {}
    for u, g in m.groupby("user"):
        g = g.head(max_history)
        history = [
            {"title": r.title, "genres": r.genres, "rating": int(r.rating)} for r in g.itertuples()
        ]
        if history:
            out[int(u)] = {"user_idx": int(u), "history": history}
    return out


def build_item_records(train_df, movies_df, idx2item_raw):
    lookup = movies_df.set_index("item_raw")[["title", "genres"]].to_dict("index")
    out = {}
    for i in sorted(train_df["item"].unique().tolist()):
        mv = lookup.get(idx2item_raw[i])
        if mv is not None:
            out[int(i)] = {"item_idx": int(i), "title": mv["title"], "genres": mv["genres"]}
    return out


def load_done(path: Path, key: str) -> set[int]:
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)[key])
    return done


def run_stage(kind, ordered_recs, out_path, workers, worker_fn, key):
    done = load_done(out_path, key)
    todo = [r for r in ordered_recs if r[key] not in done]
    print(f"[{kind}] 目標 {len(ordered_recs)} 筆，已完成 {len(done)}，本次待生成 {len(todo)}")
    if not todo:
        return 0, 0.0

    lock = Lock()
    start = time.time()
    n_done = 0
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [ex.submit(worker_fn, r) for r in todo]
        with open(out_path, "a", encoding="utf-8") as f:
            for fut in as_completed(futures):
                rec = fut.result()
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                n_done += 1
                if n_done % 50 == 0 or n_done == len(todo):
                    el = time.time() - start
                    eta = el / n_done * (len(todo) - n_done)
                    print(f"  [{kind}] {n_done}/{len(todo)}  已用 {el/60:.1f} 分，預估剩餘 {eta/60:.1f} 分")
    except KeyboardInterrupt:
        print("收到中斷，取消尚未開始的工作（已寫入的結果會保留，重跑即可續傳）")
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return n_done, time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/ml-1m")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out_dir", default="results/week4")
    ap.add_argument("--max_ratio", type=float, default=0.2, help="本次生成到全體的多少比例（前綴）")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max_history", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir, proc_dir, out_dir = Path(args.raw_dir), Path(args.processed_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(proc_dir / "train.csv")
    with open(proc_dir / "id_maps.json", "r", encoding="utf-8") as f:
        id_maps = json.load(f)
    idx2user_raw = {v: int(k) for k, v in id_maps["user2idx"].items()}
    idx2item_raw = {v: int(k) for k, v in id_maps["item2idx"].items()}

    movies_df = load_movies(raw_dir)
    ratings_df = load_ratings(raw_dir)

    user_recs = build_user_records(train_df, ratings_df, movies_df, idx2user_raw, idx2item_raw, args.max_history)
    item_recs = build_item_records(train_df, movies_df, idx2item_raw)
    print(f"有 train 互動的 user：{len(user_recs)}（總 user {len(idx2user_raw)}）；item：{len(item_recs)}")

    order_path = out_dir / "order.json"
    if order_path.exists():
        with open(order_path, "r", encoding="utf-8") as f:
            order = json.load(f)
    else:
        rng = random.Random(args.seed)
        u_order, i_order = sorted(user_recs), sorted(item_recs)
        rng.shuffle(u_order)
        rng.shuffle(i_order)
        order = {"seed": args.seed, "user_order": u_order, "item_order": i_order}
        with open(order_path, "w", encoding="utf-8") as f:
            json.dump(order, f)

    n_u = math.ceil(args.max_ratio * len(order["user_order"]))
    n_i = math.ceil(args.max_ratio * len(order["item_order"]))
    ordered_users = [user_recs[i] for i in order["user_order"][:n_u] if i in user_recs]
    ordered_items = [item_recs[i] for i in order["item_order"][:n_i] if i in item_recs]

    def user_worker(rec):
        res = gen_with_clean(
            build_user_profile_prompt(rec["history"]),
            USER_PROFILE_SYSTEM, USER_PROFILE_REQUIRED_KEYS, USER_PROFILE_ENUM_CHECKS,
        )
        return {"user_idx": rec["user_idx"], **res}

    def item_worker(rec):
        res = gen_with_clean(
            build_item_attribute_prompt(rec["title"], rec["genres"]),
            ITEM_ATTRIBUTE_SYSTEM, ITEM_ATTRIBUTE_REQUIRED_KEYS, ITEM_ATTRIBUTE_ENUM_CHECKS,
        )
        return {"item_idx": rec["item_idx"], "title": rec["title"], "genres": rec["genres"], **res}

    timing = []
    for kind, recs, path, fn, key in (
        ("users", ordered_users, out_dir / "users.jsonl", user_worker, "user_idx"),
        ("items", ordered_items, out_dir / "items.jsonl", item_worker, "item_idx"),
    ):
        n, wall = run_stage(kind, recs, path, args.workers, fn, key)
        timing.append(
            {"kind": kind, "max_ratio": args.max_ratio, "workers": args.workers,
             "n_generated": n, "wall_sec": round(wall, 1)}
        )

    with open(out_dir / "timing.jsonl", "a", encoding="utf-8") as f:
        for t in timing:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print("完成。timing 已追加到 timing.jsonl（Week6 效益分析用）。")


if __name__ == "__main__":
    main()
