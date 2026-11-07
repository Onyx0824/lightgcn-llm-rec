"""
Step A - 建立「對照用」item 嵌入，用來檢驗 LLM 增強的效益是否真的來自 LLM 生成的語意。

兩種對照（都只動 item 端，輸出格式與 build_embeddings.py 完全相同，可直接給 --aug_dir 使用）：

  meta     metadata-only：只用 MovieLens 既有的「片名＋類型」組成文字，經同一個 bge 模型編碼。
           與 LLM 版本唯一的差別是「沒有 LLM 生成的氛圍／觀眾／摘要」。
           若 meta 追得上 LLM 增強，代表增益來自 metadata 本身，LLM 生成沒有額外價值。
  shuffle  打亂對照：把 LLM 的 item 嵌入在「有生成的 item」之間隨機置換（嵌入分布完全不變，
           但與 item 的對應被打散）。若 shuffle 仍有增益，代表增益來自額外參數／隨機特徵，而非語意內容。

用法（專案根目錄）：
    python scripts/build_control_embeddings.py --mode meta    --out_dir results/stepA/meta
    python scripts/build_control_embeddings.py --mode shuffle --out_dir results/stepA/shuffle --perm_seed 0

訓練時：
    python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed S --aug_side item \
        --fixed_epochs 25 --aug_dir results/stepA/meta --tag itemmeta
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

GENRE_ZH = {
    "Action": "動作", "Adventure": "冒險", "Animation": "動畫", "Children's": "兒童",
    "Comedy": "喜劇", "Crime": "犯罪", "Documentary": "紀錄片", "Drama": "劇情",
    "Fantasy": "奇幻", "Film-Noir": "黑色電影", "Horror": "恐怖", "Musical": "音樂劇",
    "Mystery": "懸疑", "Romance": "愛情", "Sci-Fi": "科幻", "Thriller": "驚悚",
    "War": "戰爭", "Western": "西部",
}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def meta_text(title: str, genres: str) -> str:
    gs = [GENRE_ZH.get(g, g) for g in genres.split("|") if g]
    return f"類型：{'、'.join(gs) or '無'}。片名：{title}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["meta", "shuffle"], required=True)
    ap.add_argument("--gen_dir", default="results/week4", help="主實驗的生成／嵌入資料夾")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="須與主實驗相同")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--perm_seed", type=int, default=0, help="shuffle 的置換種子")
    args = ap.parse_args()

    gen_dir, out_dir = Path(args.gen_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # user 端與 order.json 原封不動複製（對照只改 item 端；訓練時用 --aug_side item）
    for name in ("user_emb.npy", "user_has.npy", "order.json"):
        shutil.copy(gen_dir / name, out_dir / name)

    i_has = np.load(gen_dir / "item_has.npy")
    i_emb = np.load(gen_dir / "item_emb.npy")

    if args.mode == "meta":
        from sentence_transformers import SentenceTransformer

        with open(Path(args.processed_dir) / "id_maps.json", "r", encoding="utf-8") as f:
            n_items = len(json.load(f)["item2idx"])
        rows = [r for r in read_jsonl(gen_dir / "items.jsonl") if i_has[r["item_idx"]]]
        ids = [r["item_idx"] for r in rows]  # 與 LLM 版本使用完全相同的 item 集合
        texts = [meta_text(r["title"], r["genres"]) for r in rows]
        model = SentenceTransformer(args.model)
        vecs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        new_emb = np.zeros((n_items, vecs.shape[1]), dtype=np.float32)
        new_emb[ids] = vecs
        print("範例文字：", texts[:3])
    else:
        rng = np.random.default_rng(args.perm_seed)
        ids = np.flatnonzero(i_has)
        perm = rng.permutation(ids)
        new_emb = np.zeros_like(i_emb)
        new_emb[ids] = i_emb[perm]  # 第 ids[k] 個 item 拿到 perm[k] 的 LLM 嵌入
        print(f"shuffle：{len(ids)} 個 item 被置換，固定點 {(perm == ids).sum()} 個")

    np.save(out_dir / "item_emb.npy", new_emb)
    np.save(out_dir / "item_has.npy", i_has)  # 覆蓋範圍與 LLM 版相同
    with open(out_dir / "control_stats.json", "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "perm_seed": args.perm_seed, "model": args.model,
                   "items_usable": int(i_has.sum()), "dim": int(new_emb.shape[1])}, f, indent=2)
    print(f"已輸出到 {out_dir}")


if __name__ == "__main__":
    main()
