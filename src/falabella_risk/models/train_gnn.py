from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


LEAKY_FEATURES = {
    # Post-outcome repayment behavior cannot be known at credit decision time.
    "repayment_latency_days",
    "on_time_repayment_share",
    # Label-derived graph aggregates leak target information.
    "neighborhood_default_rate_1hop",
    "neighborhood_default_rate_2hop",
    "peer_default_contagion_score",
}


def build_graph_tensors(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    labels: pd.DataFrame,
    train_idx: np.ndarray,
    drop_leaky_features: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, list[str]]:
    borrower_ids = features["borrower_id"].to_numpy(dtype=np.int64)
    id_to_idx = pd.Series(np.arange(len(borrower_ids), dtype=np.int64), index=borrower_ids)

    src_idx = id_to_idx.loc[edges["src_id"].to_numpy()].to_numpy(dtype=np.int64)
    dst_idx = id_to_idx.loc[edges["dst_id"].to_numpy()].to_numpy(dtype=np.int64)

    edge_index = np.vstack(
        [
            np.concatenate([src_idx, dst_idx]),
            np.concatenate([dst_idx, src_idx]),
        ]
    )

    y_map = labels.set_index("borrower_id")["default_flag"]
    y = y_map.loc[borrower_ids].to_numpy(dtype=np.int64)

    candidate_cols = [
        col for col in features.columns if col not in {"borrower_id", "default_flag"}
    ]
    excluded_cols: list[str] = []
    if drop_leaky_features:
        for col in list(candidate_cols):
            if col in LEAKY_FEATURES:
                candidate_cols.remove(col)
                excluded_cols.append(col)

    x_raw = features[candidate_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    x = np.empty_like(x_raw, dtype=np.float32)
    x[train_idx] = scaler.fit_transform(x_raw[train_idx]).astype(np.float32)
    non_train_idx = np.setdiff1d(np.arange(len(x_raw), dtype=np.int64), train_idx)
    x[non_train_idx] = scaler.transform(x_raw[non_train_idx]).astype(np.float32)

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
        borrower_ids,
        candidate_cols,
    )


class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.2):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 2)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        logits = self.out(h)
        return logits, h


def split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    rng = np.random.default_rng(seed)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def _split_class(idx_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_cls = len(idx_arr)
        n_train_cls = int(0.70 * n_cls)
        n_val_cls = int(0.15 * n_cls)
        return (
            idx_arr[:n_train_cls],
            idx_arr[n_train_cls : n_train_cls + n_val_cls],
            idx_arr[n_train_cls + n_val_cls :],
        )

    pos_train, pos_val, pos_test = _split_class(pos_idx)
    neg_train, neg_val, neg_test = _split_class(neg_idx)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])
    test_idx = np.concatenate([pos_test, neg_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def compute_metrics(logits: torch.Tensor, y: torch.Tensor, idx: np.ndarray) -> dict[str, float]:
    probs = torch.softmax(logits[idx], dim=1)[:, 1].detach().cpu().numpy()
    y_true = y[idx].detach().cpu().numpy()
    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }


def train_graphsage(
    feature_path: Path,
    edges_path: Path,
    labels_path: Path,
    model_out: Path,
    embeddings_out: Path,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_channels: int,
    dropout: float,
    weight_decay: float,
    early_stopping_patience: int,
    drop_leaky_features: bool,
) -> None:
    try:
        import torch_geometric  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "torch_geometric is required for train_gnn.py. Install dependencies first."
        ) from exc

    features = pd.read_parquet(feature_path)
    edges = pd.read_parquet(edges_path)
    labels = pd.read_parquet(labels_path)

    y_np = labels.set_index("borrower_id").loc[features["borrower_id"].to_numpy()][
        "default_flag"
    ].to_numpy(dtype=np.int64)
    train_idx, val_idx, test_idx = split_indices(y_np, seed)

    x, edge_index, y, borrower_ids, used_feature_cols = build_graph_tensors(
        features=features,
        edges=edges,
        labels=labels,
        train_idx=train_idx,
        drop_leaky_features=drop_leaky_features,
    )

    excluded = sorted([c for c in LEAKY_FEATURES if c in features.columns and c not in used_feature_cols])
    print(f"Using {len(used_feature_cols)} node features for GraphSAGE input")
    if excluded:
        print("Excluded potential leakage features:", ", ".join(excluded))

    model = GraphSAGEClassifier(
        in_channels=x.shape[1], hidden_channels=hidden_channels, dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    class_counts = torch.bincount(y)
    class_weights = (class_counts.sum() / torch.clamp(class_counts.float(), min=1.0)).float()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_auc = -1.0
    best_epoch = -1
    best_state = None
    no_improve_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, edge_index)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x, edge_index)
            val_metrics = compute_metrics(val_logits, y, val_idx)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.item()),
                "val_auc": float(val_metrics["auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
            }
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        print(
            f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | val_pr_auc={val_metrics['pr_auc']:.4f}"
        )

        if early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} after {no_improve_epochs} epochs without val AUC improvement."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits, embeddings = model(x, edge_index)
        test_metrics = compute_metrics(test_logits, y, test_idx)

    print("\nGraphSAGE test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nTraining curve:")
    print("epoch\ttrain_loss\tval_auc\tval_pr_auc")
    for row in history:
        print(
            f"{int(row['epoch'])}\t{row['train_loss']:.4f}\t{row['val_auc']:.4f}\t{row['val_pr_auc']:.4f}"
        )
    print(f"\nBest checkpoint epoch: {best_epoch} (val_auc={best_val_auc:.4f})")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)

    meta_out = model_out.with_suffix(".meta.json")
    meta_out.write_text(
        json.dumps(
            {
                "hidden_channels": hidden_channels,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "learning_rate": learning_rate,
                "epochs_requested": epochs,
                "early_stopping_patience": early_stopping_patience,
                "best_epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "test_auc": test_metrics["auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "feature_count": len(used_feature_cols),
                "excluded_features": excluded,
            },
            indent=2,
        )
    )

    emb_df = pd.DataFrame(embeddings.detach().cpu().numpy())
    emb_df.insert(0, "borrower_id", borrower_ids)
    emb_df.columns = ["borrower_id"] + [f"emb_{i:03d}" for i in range(emb_df.shape[1] - 1)]

    embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(emb_df, preserve_index=False), embeddings_out, compression="snappy"
    )

    print(f"Saved GraphSAGE model to: {model_out}")
    print(f"Saved GraphSAGE metadata to: {meta_out}")
    print(f"Saved embeddings to: {embeddings_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE and export borrower embeddings.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Input features parquet path.",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/raw/edges.parquet"),
        help="Input edges parquet path.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/raw/labels.parquet"),
        help="Input labels parquet path.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/graphsage.pt"),
        help="Output model path.",
    )
    parser.add_argument(
        "--embeddings-out",
        type=Path,
        default=Path("data/processed/gnn_embeddings.parquet"),
        help="Output embeddings parquet path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Hidden size for GraphSAGE")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop if val AUC does not improve for this many epochs (0 disables).",
    )
    parser.add_argument(
        "--allow-leaky-features",
        action="store_true",
        help="Include known leakage-prone features (not recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_graphsage(
        feature_path=args.features,
        edges_path=args.edges,
        labels_path=args.labels,
        model_out=args.model_out,
        embeddings_out=args.embeddings_out,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        drop_leaky_features=not args.allow_leaky_features,
    )


if __name__ == "__main__":
    main()
