"""
Week 2 - 訓練 LightGCN baseline
讀入 data/processed/{train,valid,test}.csv，訓練並用 early stopping，
最後在 test set 上算 Recall@20 / NDCG@20，把結果存到 results/baseline_results.json。

用法：
    python scripts/train_baseline.py --config configs/baseline.yaml

注意：這支要在你本機（有GPU、torch裝好）跑，這個sandbox裝不下完整的CUDA版torch
（磁碟空間不夠），所以邏輯已經用numpy/pandas單獨測過，但完整訓練迴圈沒有在這裡
實際跑過一次，第一次跑請先用小資料集或少量epoch跑通再放心跑完整訓練。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.lightgcn import LightGCN, build_norm_adj, bpr_loss  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402


def build_user_pos_items(df: pd.DataFrame, n_users: int) -> dict[int, list[int]]:
    d = {u: [] for u in range(n_users)}
    for u, it in zip(df["user"].values, df["item"].values):
        d[int(u)].append(int(it))
    return d


def sample_bpr_batch(
    user_pos_items: dict[int, list[int]],
    interactions: np.ndarray,
    n_items: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """對 train interactions 隨機抽 batch_size 筆 (user, pos_item)，
    每筆再抽一個該user沒互動過的 neg_item（BPR標準做法）。"""
    idx = np.random.randint(0, len(interactions), size=batch_size)
    batch = interactions[idx]
    users = batch[:, 0]
    pos_items = batch[:, 1]

    neg_items = np.empty(batch_size, dtype=np.int64)
    for i, u in enumerate(users):
        pos_set = user_pos_items[int(u)]
        while True:
            neg = np.random.randint(0, n_items)
            if neg not in pos_set:
                neg_items[i] = neg
                break

    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(pos_items, dtype=torch.long),
        torch.tensor(neg_items, dtype=torch.long),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu")
    print(f"使用裝置: {device}")

    data_dir = Path(cfg["data_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    valid_df = pd.read_csv(data_dir / "valid.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    with open(data_dir / "id_maps.json", "r", encoding="utf-8") as f:
        id_maps = json.load(f)
    n_users = len(id_maps["user2idx"])
    n_items = len(id_maps["item2idx"])
    print(f"n_users={n_users}, n_items={n_items}, train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    train_user_pos_items = build_user_pos_items(train_df, n_users)
    valid_user_pos_items = {
        u: set(items) for u, items in build_user_pos_items(valid_df, n_users).items()
    }
    test_user_pos_items = {
        u: set(items) for u, items in build_user_pos_items(test_df, n_users).items()
    }
    # 評估 test 時，除了 train 也要把 valid 的正樣本從候選排名中排除，
    # 否則 valid 裡的正確答案會佔掉 test top-K 的名額，變相低估 test 指標。
    train_and_valid_pos_items = {
        u: train_user_pos_items[u] + list(valid_user_pos_items.get(u, []))
        for u in range(n_users)
    }

    norm_adj = build_norm_adj(
        n_users, n_items, train_df["user"].values, train_df["item"].values
    ).to(device)

    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embed_dim=cfg.get("embed_dim", 64),
        n_layers=cfg.get("n_layers", 3),
        norm_adj=norm_adj,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    train_interactions = train_df[["user", "item"]].values

    n_epochs = cfg.get("n_epochs", 100)
    batch_size = cfg.get("batch_size", 2048)
    n_batches_per_epoch = max(1, len(train_interactions) // batch_size)
    eval_every = cfg.get("eval_every", 5)
    patience = cfg.get("patience", 5)
    k = cfg.get("k", 20)
    weight_decay = cfg.get("weight_decay", 1e-4)

    best_recall = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for _ in range(n_batches_per_epoch):
            users, pos_items, neg_items = sample_bpr_batch(
                train_user_pos_items, train_interactions, n_items, batch_size
            )
            users, pos_items, neg_items = (
                users.to(device),
                pos_items.to(device),
                neg_items.to(device),
            )

            optimizer.zero_grad()
            pos_scores, neg_scores, l2_reg = model(users, pos_items, neg_items)
            loss = bpr_loss(pos_scores, neg_scores, l2_reg, weight_decay)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches_per_epoch
        epoch_time = time.time() - epoch_start
        print(f"[epoch {epoch}] loss={avg_loss:.4f}  ({epoch_time:.1f}s)")

        if epoch % eval_every == 0 or epoch == n_epochs:
            model.eval()
            metrics = evaluate(model, train_user_pos_items, valid_user_pos_items, k=k)
            print(f"  valid: {metrics}")

            recall_key = f"recall@{k}"
            if metrics[recall_key] > best_recall:
                best_recall = metrics[recall_key]
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping：valid recall@{k} 連續 {patience} 次評估沒有進步")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"載入最佳checkpoint（epoch {best_epoch}, valid recall@{k}={best_recall:.4f}）")

    model.eval()
    test_metrics = evaluate(model, train_and_valid_pos_items, test_user_pos_items, k=k)
    print(f"Test set 結果: {test_metrics}")

    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    result_record = {
        "model": "LightGCN (baseline)",
        "config": cfg,
        "best_epoch": best_epoch,
        "best_valid_recall": best_recall,
        "test_metrics": test_metrics,
    }
    out_path = results_dir / "baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_record, f, ensure_ascii=False, indent=2)
    print(f"結果已存到 {out_path}")

    if cfg.get("save_checkpoint", True):
        ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "lightgcn_baseline.pt")
        print(f"模型權重已存到 {ckpt_dir / 'lightgcn_baseline.pt'}")


if __name__ == "__main__":
    main()
