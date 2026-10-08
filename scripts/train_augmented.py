"""
Week 4 - 訓練 LLM 增強版 LightGCN，並與 baseline 比較。

用法：
    python scripts/train_augmented.py --ratio 0.2
    python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed 0
    python scripts/train_augmented.py --ratio 1.0 --alpha 0      # sanity check：應與 baseline 接近
    python scripts/train_augmented.py --ratio 0.2 --alpha 1.0 --seed 0 --fixed_epochs 25

評估邏輯與 train_baseline.py 完全一致（valid 排除 train；test 排除 train∪valid）。
結果存到 results/week4_aug_r{ratio}_a{alpha}_s{seed}.json
--fixed_epochs N > 0：不做 early stopping，固定訓練 N epoch 後直接評估 test，
結果改存 results/week4fe_aug_r{ratio}_a{alpha}_s{seed}.json（避免和 early-stopping 版本混在一起）。
結果 JSON 另含 test_aug_users / test_nonaug_users：test 指標依「該 user 是否有 LLM 增強」分群。
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_baseline import build_user_pos_items, sample_bpr_batch  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.model.lightgcn import bpr_loss, build_norm_adj  # noqa: E402
from src.model.lightgcn_aug import LightGCNAug  # noqa: E402


def ratio_mask(order: list[int], has: np.ndarray, ratio: float) -> np.ndarray:
    k = math.ceil(ratio * len(order))
    mask = np.zeros(len(has), dtype=bool)
    mask[order[:k]] = True
    return mask & has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--aug_dir", default="results/week4")
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed_epochs", type=int, default=0,
                    help=">0：固定訓練這麼多 epoch、不做 early stopping（降低停止點造成的變異）")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu")

    data_dir, aug_dir = Path(cfg["data_dir"]), Path(args.aug_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    valid_df = pd.read_csv(data_dir / "valid.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "id_maps.json", "r", encoding="utf-8") as f:
        id_maps = json.load(f)
    n_users, n_items = len(id_maps["user2idx"]), len(id_maps["item2idx"])

    # --- LLM 特徵 + 比例遮罩 ---
    u_emb, i_emb = np.load(aug_dir / "user_emb.npy"), np.load(aug_dir / "item_emb.npy")
    u_has, i_has = np.load(aug_dir / "user_has.npy"), np.load(aug_dir / "item_has.npy")
    with open(aug_dir / "order.json", "r", encoding="utf-8") as f:
        order = json.load(f)
    u_mask = ratio_mask(order["user_order"], u_has, args.ratio)
    i_mask = ratio_mask(order["item_order"], i_has, args.ratio)
    print(f"ratio={args.ratio} alpha={args.alpha} seed={args.seed} | 增強 user {u_mask.sum()}/{n_users}, item {i_mask.sum()}/{n_items}")

    train_pos = build_user_pos_items(train_df, n_users)
    valid_pos = {u: set(v) for u, v in build_user_pos_items(valid_df, n_users).items()}
    test_pos = {u: set(v) for u, v in build_user_pos_items(test_df, n_users).items()}
    train_valid_pos = {u: train_pos[u] + list(valid_pos.get(u, [])) for u in range(n_users)}
    # 分群評估用：test 中「有增強的 user」vs「沒有增強的 user」（只算有 test 正樣本的 user）
    test_aug = {u: s for u, s in test_pos.items() if s and u_mask[u]}
    test_nonaug = {u: s for u, s in test_pos.items() if s and not u_mask[u]}

    norm_adj = build_norm_adj(n_users, n_items, train_df["user"].values, train_df["item"].values).to(device)
    model = LightGCNAug(
        n_users, n_items, u_emb, i_emb, u_mask, i_mask,
        embed_dim=cfg.get("embed_dim", 64), n_layers=cfg.get("n_layers", 3),
        norm_adj=norm_adj, alpha=args.alpha,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    interactions = train_df[["user", "item"]].values
    batch_size = cfg.get("batch_size", 2048)
    n_batches = max(1, len(interactions) // batch_size)
    k, patience, eval_every = cfg.get("k", 20), cfg.get("patience", 5), cfg.get("eval_every", 5)
    fixed = args.fixed_epochs > 0
    n_epochs = args.fixed_epochs if fixed else cfg.get("n_epochs", 100)

    best_recall, best_epoch, no_improve, best_state = -1.0, -1, 0, None
    train_start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        for _ in range(n_batches):
            u, p, n = sample_bpr_batch(train_pos, interactions, n_items, batch_size)
            u, p, n = u.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            ps, ns, l2 = model(u, p, n)
            loss = bpr_loss(ps, ns, l2, cfg.get("weight_decay", 1e-4))
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"[epoch {epoch}] loss={total / n_batches:.4f}")

        if fixed:
            continue
        if epoch % eval_every == 0:
            model.eval()
            m = evaluate(model, train_pos, valid_pos, k=k)
            print(f"  valid: {m}")
            if m[f"recall@{k}"] > best_recall:
                best_recall, best_epoch, no_improve = m[f"recall@{k}"], epoch, 0
                best_state = {a: b.clone() for a, b in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping")
                    break
    train_sec = time.time() - train_start

    if fixed:
        # 固定 epoch 模式：用最終權重，valid 只在最後算一次供記錄
        model.eval()
        best_epoch = args.fixed_epochs
        best_recall = evaluate(model, train_pos, valid_pos, k=k)[f"recall@{k}"]
    elif best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_metrics = evaluate(model, train_valid_pos, test_pos, k=k)
    print(f"Test: {test_metrics}")
    sub_aug = {"n_users": len(test_aug), **evaluate(model, train_valid_pos, test_aug, k=k)}
    sub_non = {"n_users": len(test_nonaug), **evaluate(model, train_valid_pos, test_nonaug, k=k)}
    print(f"  有增強 user: {sub_aug}")
    print(f"  無增強 user: {sub_non}")

    base_path = Path(cfg.get("results_dir", "results")) / "baseline_results.json"
    delta = None
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            b = json.load(f)["test_metrics"]
        delta = {key: round(test_metrics[key] - b[key], 6) for key in test_metrics}
        print(f"相對 baseline 差值: {delta}")

    out = {
        "model": "LightGCN + LLM aug",
        "ratio": args.ratio, "alpha": args.alpha, "seed": args.seed,
        "n_aug_users": int(u_mask.sum()), "n_aug_items": int(i_mask.sum()),
        "best_epoch": best_epoch, "best_valid_recall": best_recall,
        "train_sec": round(train_sec, 1),
        "fixed_epochs": args.fixed_epochs,
        "test_metrics": test_metrics, "delta_vs_baseline": delta,
        "test_aug_users": sub_aug, "test_nonaug_users": sub_non,
    }
    prefix = "week4fe_aug" if fixed else "week4_aug"
    out_path = Path(cfg.get("results_dir", "results")) / f"{prefix}_r{args.ratio}_a{args.alpha}_s{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"結果已存到 {out_path}")


if __name__ == "__main__":
    main()
