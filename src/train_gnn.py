from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def build_graph_tensors(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
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

    x_raw = features.drop(columns=["borrower_id", "default_flag"], errors="ignore").to_numpy(
        dtype=np.float32
    )
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw).astype(np.float32)

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
        borrower_ids,
    )


class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 2)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        logits = self.out(h)
        return logits, h


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
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

    x, edge_index, y, borrower_ids = build_graph_tensors(features, edges, labels)

    train_idx, val_idx, test_idx = split_indices(len(x), seed)

    model = GraphSAGEClassifier(in_channels=x.shape[1], hidden_channels=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    class_counts = torch.bincount(y)
    class_weights = (class_counts.sum() / torch.clamp(class_counts.float(), min=1.0)).float()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_auc = -1.0
    best_state = None

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

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | val_pr_auc={val_metrics['pr_auc']:.4f}"
            )

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

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)

    emb_df = pd.DataFrame(embeddings.detach().cpu().numpy())
    emb_df.insert(0, "borrower_id", borrower_ids)
    emb_df.columns = ["borrower_id"] + [f"emb_{i:03d}" for i in range(emb_df.shape[1] - 1)]

    embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(emb_df, preserve_index=False), embeddings_out, compression="snappy")

    print(f"Saved GraphSAGE model to: {model_out}")
    print(f"Saved embeddings to: {embeddings_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE and export borrower embeddings.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features.parquet"),
        help="Input features parquet path.",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/edges.parquet"),
        help="Input edges parquet path.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/labels.parquet"),
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
        default=Path("data/gnn_embeddings.parquet"),
        help="Output embeddings parquet path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="Learning rate")
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
    )


if __name__ == "__main__":
    main()
