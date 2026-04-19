from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


LEAKY_FEATURES = {
    "repayment_latency_days",
    "on_time_repayment_share",
    "neighborhood_default_rate_1hop",
    "neighborhood_default_rate_2hop",
    "peer_default_contagion_score",
}


def evaluate(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    return evaluate_with_threshold(model, x_test, y_test, threshold=0.5)


def evaluate_with_threshold(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, float]:
    start = perf_counter()
    y_prob = model.predict_proba(x_test)[:, 1]
    latency_ms_per_row = (perf_counter() - start) * 1000.0 / max(1, len(x_test))
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "brier": brier_score_loss(y_test, y_prob),
        "latency_ms_per_row": latency_ms_per_row,
    }


def select_f1_threshold(y_true: pd.Series, y_proba: pd.Series) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(f1_scores.argmax())
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def build_hybrid_dataset(feature_path: Path, embedding_path: Path) -> pd.DataFrame:
    features = pd.read_parquet(feature_path)
    embeddings = pd.read_parquet(embedding_path)

    if "borrower_id" not in features.columns or "borrower_id" not in embeddings.columns:
        raise ValueError("Both features and embeddings must include borrower_id")

    merged = features.merge(embeddings, on="borrower_id", how="inner")
    if merged.empty:
        raise RuntimeError("No rows after joining features and embeddings.")

    if "default_flag" not in merged.columns:
        raise ValueError("Hybrid training data must include default_flag")

    return merged


def train_hybrid(
    feature_path: Path,
    embedding_path: Path,
    model_out: Path,
    seed: int,
    mlflow_uri: str | None,
    experiment_name: str,
    drop_leaky_features: bool,
) -> None:
    df = build_hybrid_dataset(feature_path, embedding_path)

    x = df.drop(columns=["borrower_id", "default_flag"], errors="ignore")
    excluded_cols: list[str] = []
    if drop_leaky_features:
        excluded_cols = [col for col in x.columns if col in LEAKY_FEATURES]
        if excluded_cols:
            x = x.drop(columns=excluded_cols)

    y = df["default_flag"].astype(int)

    if excluded_cols:
        print("Excluded potential leakage features:", ", ".join(sorted(excluded_cols)))

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    model = XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.5,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    default_metrics = evaluate_with_threshold(model, x_test, y_test, threshold=0.5)
    val_prob = model.predict_proba(x_val)[:, 1]
    best_threshold, val_best_f1 = select_f1_threshold(y_val, val_prob)
    calibrated_metrics = evaluate_with_threshold(model, x_test, y_test, threshold=best_threshold)

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="hybrid_ensemble_xgb"):
        for key, value in default_metrics.items():
            mlflow.log_metric(key, float(value))
        for key, value in calibrated_metrics.items():
            mlflow.log_metric(f"calibrated_{key}", float(value))
        mlflow.log_param("train_rows", int(len(x_train)))
        mlflow.log_param("val_rows", int(len(x_val)))
        mlflow.log_param("test_rows", int(len(x_test)))
        mlflow.log_param("feature_count", int(x.shape[1]))
        mlflow.log_param("calibration_threshold", float(best_threshold))
        mlflow.log_metric("val_best_f1", float(val_best_f1))
        mlflow.log_param("excluded_features", json.dumps(sorted(excluded_cols)))

        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)
        mlflow.log_artifact(str(model_out))

        threshold_out = model_out.with_suffix(".threshold.json")
        threshold_out.write_text(
            json.dumps(
                {
                    "decision_threshold": best_threshold,
                    "validation_best_f1": val_best_f1,
                    "excluded_features": sorted(excluded_cols),
                    "default_metrics": default_metrics,
                    "calibrated_metrics": calibrated_metrics,
                },
                indent=2,
            )
        )
        mlflow.log_artifact(str(threshold_out))

    print("Hybrid ensemble metrics (threshold=0.5):")
    for key, value in default_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(
        f"\nHybrid calibrated threshold (from val): {best_threshold:.4f} "
        f"(val_f1={val_best_f1:.4f})"
    )
    print("Hybrid ensemble calibrated test metrics:")
    for key, value in calibrated_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved hybrid model to: {model_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hybrid ensemble from tabular features and GNN embeddings."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Input feature parquet.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/gnn_embeddings.parquet"),
        help="Input GNN embedding parquet.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/hybrid_ensemble.pkl"),
        help="Output path for trained hybrid model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="",
        help="Optional MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="falabella_hybrid",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--allow-leaky-features",
        action="store_true",
        help="Include known leakage-prone features (not recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_hybrid(
        feature_path=args.features,
        embedding_path=args.embeddings,
        model_out=args.model_out,
        seed=args.seed,
        mlflow_uri=args.mlflow_uri if args.mlflow_uri else None,
        experiment_name=args.experiment,
        drop_leaky_features=not args.allow_leaky_features,
    )


if __name__ == "__main__":
    main()
