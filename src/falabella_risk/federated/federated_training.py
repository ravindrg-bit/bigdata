from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CITY_TO_CLIENT = {
    "Ciudad de México": "CDMX",
    "Mérida": "Mérida",
    "Monterrey": "Monterrey",
    "Guadalajara": "Guadalajara",
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def assign_regions_from_city(city_series: pd.Series) -> np.ndarray:
    mapped = city_series.map(CITY_TO_CLIENT)
    if mapped.isna().any():
        missing = city_series[mapped.isna()].astype(str).unique().tolist()
        raise ValueError(f"Unsupported city values for federated partitioning: {missing}")
    return mapped.to_numpy()


class BinaryLinear(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    regions_train: np.ndarray
    feature_columns: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    region_counts_full: dict[str, int]
    region_counts_train: dict[str, int]


def prepare_dataset(features_path: Path, seed: int, borrowers_path: Path) -> DatasetBundle:
    df = pd.read_parquet(features_path)
    if "borrower_id" not in df.columns or "default_flag" not in df.columns:
        raise ValueError("features.parquet must include borrower_id and default_flag")

    borrowers = pd.read_parquet(borrowers_path)
    if "borrower_id" not in borrowers.columns or "city" not in borrowers.columns:
        raise ValueError("borrowers.parquet must include borrower_id and city")

    borrower_city = borrowers[["borrower_id", "city"]].copy()
    borrower_city = borrower_city[borrower_city["city"].isin(CITY_TO_CLIENT.keys())]

    df = df.merge(borrower_city, on="borrower_id", how="inner")
    if df.empty:
        raise RuntimeError("No rows available after filtering to federated client cities.")

    y = df["default_flag"].astype(int).to_numpy()
    regions = assign_regions_from_city(df["city"])
    region_counts_full = {
        region: int((regions == region).sum()) for region in CITY_TO_CLIENT.values()
    }

    x = df.drop(columns=["default_flag", "borrower_id", "city"], errors="ignore").copy()
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    feature_columns = x.columns.tolist()

    idx = np.arange(len(df))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    idx_val, idx_test, _y_val, _y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x.iloc[idx_train].to_numpy(dtype=np.float32)).astype(np.float32)
    x_val = scaler.transform(x.iloc[idx_val].to_numpy(dtype=np.float32)).astype(np.float32)
    x_test = scaler.transform(x.iloc[idx_test].to_numpy(dtype=np.float32)).astype(np.float32)

    regions_train = regions[idx_train]
    region_counts_train = {
        region: int((regions_train == region).sum()) for region in CITY_TO_CLIENT.values()
    }

    return DatasetBundle(
        x_train=x_train,
        y_train=y[idx_train].astype(np.float32),
        x_val=x_val,
        y_val=y[idx_val].astype(np.float32),
        x_test=x_test,
        y_test=y[idx_test].astype(np.float32),
        regions_train=regions_train,
        feature_columns=feature_columns,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        region_counts_full=region_counts_full,
        region_counts_train=region_counts_train,
    )


def tensorize(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_local(
    global_state: dict[str, torch.Tensor],
    x_local: np.ndarray,
    y_local: np.ndarray,
    epochs: int,
    learning_rate: float,
) -> dict[str, torch.Tensor]:
    x_t, y_t = tensorize(x_local, y_local)

    model = BinaryLinear(in_features=x_t.shape[1])
    model.load_state_dict(global_state)

    pos_count = float((y_t == 1).sum())
    neg_count = float((y_t == 0).sum())
    pos_weight = torch.tensor(max(1.0, neg_count / max(1.0, pos_count)), dtype=torch.float32)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def fedavg(states: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No client states to aggregate")

    total_weight = float(sum(weights))
    agg: dict[str, torch.Tensor] = {}

    for key in states[0].keys():
        weighted = sum(state[key] * (w / total_weight) for state, w in zip(states, weights))
        agg[key] = weighted

    return agg


def evaluate_state(
    state: dict[str, torch.Tensor],
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    x_t, y_t = tensorize(x, y)

    model = BinaryLinear(in_features=x.shape[1])
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        start = perf_counter()
        logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        latency_ms = (perf_counter() - start) * 1000.0 / max(len(x), 1)

    y_true = y_t.cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "accuracy": float((y_pred == y_true).mean()),
        "latency_ms_per_row": float(latency_ms),
    }


def train_centralized_reference(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    x_t, y_t = tensorize(x_train, y_train)

    model = BinaryLinear(in_features=x_train.shape[1])

    pos_count = float((y_t == 1).sum())
    neg_count = float((y_t == 0).sum())
    pos_weight = torch.tensor(max(1.0, neg_count / max(1.0, pos_count)), dtype=torch.float32)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    return evaluate_state(model.state_dict(), x_test, y_test)


def train_federated(
    features_path: Path,
    borrowers_path: Path,
    model_out: Path,
    report_out: Path,
    seed: int,
    rounds: int,
    local_epochs: int,
    learning_rate: float,
) -> dict[str, object]:
    set_seed(seed)
    data = prepare_dataset(features_path, seed=seed, borrowers_path=borrowers_path)

    regions = [region for region in CITY_TO_CLIENT.values() if region in data.region_counts_train]
    region_masks = {region: (data.regions_train == region) for region in regions}

    print("Federated client counts (full selected dataset):", data.region_counts_full)
    print("Federated client counts (train split):", data.region_counts_train)

    global_model = BinaryLinear(in_features=data.x_train.shape[1])
    global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    best_state = None
    best_val_auc = -1.0
    round_history: list[dict[str, float | int]] = []

    for round_idx in range(1, rounds + 1):
        client_states: list[dict[str, torch.Tensor]] = []
        client_weights: list[int] = []

        for region in regions:
            mask = region_masks[region]
            x_local = data.x_train[mask]
            y_local = data.y_train[mask]

            local_state = train_local(
                global_state=global_state,
                x_local=x_local,
                y_local=y_local,
                epochs=local_epochs,
                learning_rate=learning_rate,
            )
            client_states.append(local_state)
            client_weights.append(int(mask.sum()))

        global_state = fedavg(client_states, client_weights)
        val_metrics = evaluate_state(global_state, data.x_val, data.y_val)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.detach().clone() for k, v in global_state.items()}

        round_history.append(
            {
                "round": int(round_idx),
                "val_auc": float(val_metrics["auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
                "val_brier": float(val_metrics["brier"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )

        print(
            f"Round {round_idx:02d} | val_auc={val_metrics['auc']:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Federated training failed to produce a best model.")

    fed_metrics = evaluate_state(best_state, data.x_test, data.y_test)
    centralized_epochs = max(8, int(rounds * local_epochs * 1.5))
    centralized_metrics = train_centralized_reference(
        x_train=data.x_train,
        y_train=data.y_train,
        x_test=data.x_test,
        y_test=data.y_test,
        epochs=centralized_epochs,
        learning_rate=learning_rate,
    )

    report = {
        "seed": seed,
        "clients": regions,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "centralized_epochs": centralized_epochs,
        "learning_rate": learning_rate,
        "dataset": {
            "in_features": int(data.x_train.shape[1]),
            "train_rows": int(len(data.x_train)),
            "val_rows": int(len(data.x_val)),
            "test_rows": int(len(data.x_test)),
            "client_rows_full": data.region_counts_full,
            "client_rows_train": data.region_counts_train,
        },
        "round_history": round_history,
        "federated_metrics": fed_metrics,
        "centralized_metrics": centralized_metrics,
        "auc_gap_vs_centralized": float(centralized_metrics["auc"] - fed_metrics["auc"]),
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_columns": data.feature_columns,
            "scaler_mean": data.scaler_mean,
            "scaler_scale": data.scaler_scale,
            "report": report,
        },
        model_out,
    )

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Federated test metrics:", fed_metrics)
    print("Centralized test metrics:", centralized_metrics)
    print(f"Saved federated model: {model_out}")
    print(f"Saved federated report: {report_out}")

    return report


@dataclass
class FederatedInferenceEngine:
    model_path: Path

    def __post_init__(self) -> None:
        payload = torch.load(self.model_path, map_location="cpu")
        self.feature_columns: list[str] = list(payload["feature_columns"])
        self.scaler_mean = np.array(payload["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.array(payload["scaler_scale"], dtype=np.float32)

        state_dict = payload["state_dict"]
        self.model = BinaryLinear(in_features=len(self.feature_columns))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare(self, row: pd.Series | pd.DataFrame) -> np.ndarray:
        if isinstance(row, pd.Series):
            x = row.to_frame().T
        else:
            x = row.copy()

        x = x.drop(columns=["default_flag", "borrower_id"], errors="ignore")
        for col in self.feature_columns:
            if col not in x.columns:
                x[col] = 0.0

        x = x[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        x_np = x.to_numpy(dtype=np.float32)

        scale = np.where(self.scaler_scale == 0.0, 1.0, self.scaler_scale)
        return (x_np - self.scaler_mean) / scale

    def predict_proba(self, row: pd.Series | pd.DataFrame) -> np.ndarray:
        x_np = self._prepare(row)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(torch.tensor(x_np, dtype=torch.float32))).cpu().numpy()
        return np.vstack([1.0 - probs, probs]).T

    def score_row(self, row: pd.Series) -> float:
        return float(self.predict_proba(row)[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated learning simulation with FedAvg.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Path to features parquet.",
    )
    parser.add_argument(
        "--borrowers",
        type=Path,
        default=Path("data/raw/borrowers.parquet"),
        help="Path to borrowers parquet for regional client partitioning.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/federated_model.pt"),
        help="Output path for federated model.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/federated_report.json"),
        help="Output path for federated metrics report.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--rounds", type=int, default=8, help="Federated rounds.")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per round.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_federated(
        features_path=args.features,
        borrowers_path=args.borrowers,
        model_out=args.model_out,
        report_out=args.report_out,
        seed=args.seed,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
