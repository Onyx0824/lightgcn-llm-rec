"""
Week 4 - 把 LLM 生成的結構化 JSON 轉成文字，再用 sentence-transformers 轉成 embedding。

輸出（results/week4/）：
    user_emb.npy  (n_users, d)  float32，已 L2 normalize；沒生成的列為 0
    user_has.npy  (n_users,)    bool，該 user 是否有可用的 LLM 特徵
    item_emb.npy / item_has.npy 同上

注意：
- 模型預設用 BAAI/bge-small-zh-v1.5（512維），因為生成文字是繁體中文。
  all-MiniLM-L6-v2 是英文模型，拿來編中文語意效果會差很多，會污染「LLM增強有沒有用」的結論。
- 文字只用 LLM 產出的欄位（不含 guessed_age_group：pilot 中 98/100 是 unknown，沒有資訊量）。
- 20%/50%/100% 的篩選不在這裡做，在訓練腳本用 order.json 的前綴決定，所以這支只需跑一次。
- 預設保留 dirty（簡體/亂碼殘留）的筆數，加 --drop_dirty 可排除，Week5 可拿來對照。

用法：
    python src/llm_augmentation/build_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def user_text(p: dict) -> str:
    pref = "、".join(p.get("preferred_genres", [])) or "無"
    dis = "、".join(p.get("disliked_genres", [])) or "無"
    return f"偏好類型：{pref}。不喜歡類型：{dis}。評分傾向：{p.get('rating_tendency', '')}。{p.get('profile_summary', '')}"


def item_text(p: dict) -> str:
    mood = "、".join(p.get("mood_tags", [])) or "無"
    supp = "、".join(p.get("supplementary_genres", [])) or "無"
    return f"氛圍：{mood}。適合觀眾：{p.get('target_audience', '')}。補充類型：{supp}。{p.get('profile_summary', '')}"


def encode_rows(model, rows, key, n_total, text_fn, drop_dirty):
    ids, texts = [], []
    for r in rows:
        if not r["success"] or r["parsed"] is None:
            continue
        if drop_dirty and r.get("dirty"):
            continue
        ids.append(r[key])
        texts.append(text_fn(r["parsed"]))
    emb = np.zeros((n_total, model.get_sentence_embedding_dimension()), dtype=np.float32)
    has = np.zeros(n_total, dtype=bool)
    if texts:
        vecs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        emb[ids] = vecs
        has[ids] = True
    return emb, has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="results/week4")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--drop_dirty", action="store_true")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    with open(Path(args.processed_dir) / "id_maps.json", "r", encoding="utf-8") as f:
        id_maps = json.load(f)
    n_users, n_items = len(id_maps["user2idx"]), len(id_maps["item2idx"])

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    start = time.time()

    u_rows = read_jsonl(gen_dir / "users.jsonl")
    i_rows = read_jsonl(gen_dir / "items.jsonl")
    u_emb, u_has = encode_rows(model, u_rows, "user_idx", n_users, user_text, args.drop_dirty)
    i_emb, i_has = encode_rows(model, i_rows, "item_idx", n_items, item_text, args.drop_dirty)

    np.save(gen_dir / "user_emb.npy", u_emb)
    np.save(gen_dir / "user_has.npy", u_has)
    np.save(gen_dir / "item_emb.npy", i_emb)
    np.save(gen_dir / "item_has.npy", i_has)

    stats = {
        "model": args.model,
        "dim": int(u_emb.shape[1]),
        "users_generated": len(u_rows),
        "users_usable": int(u_has.sum()),
        "items_generated": len(i_rows),
        "items_usable": int(i_has.sum()),
        "dirty_users": sum(1 for r in u_rows if r.get("dirty")),
        "dirty_items": sum(1 for r in i_rows if r.get("dirty")),
        "failed_users": sum(1 for r in u_rows if not r["success"]),
        "failed_items": sum(1 for r in i_rows if not r["success"]),
        "embed_sec": round(time.time() - start, 1),
    }
    with open(gen_dir / "embedding_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
