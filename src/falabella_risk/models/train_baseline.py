from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    start = perf_counter()
    y_proba = model.predict_proba(x_test)[:, 1]
    elapsed_ms = (perf_counter() - start) * 1000.0 / max(len(x_test), 1)

    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "brier": brier_score_loss(y_test, y_proba),
        "latency_ms_per_row": elapsed_ms,
    }


def run_training(
    feature_path: Path,
    model_out: Path,
    random_state: int,
    mlflow_uri: str | None,
    experiment_name: str,
) -> None:
    df = pd.read_parquet(feature_path)
    if "default_flag" not in df.columns:
        raise ValueError("features.parquet must include default_flag column")

    x = df.drop(columns=["default_flag", "borrower_id"], errors="ignore")
    y = df["default_flag"].astype(int)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="baseline_lr_xgb"):
        lr = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        n_jobs=None,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        lr.fit(x_train, y_train)
        lr_metrics = evaluate_model(lr, x_test, y_test)

        xgb = XGBClassifier(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=-1,
        )
        xgb.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        xgb_metrics = evaluate_model(xgb, x_test, y_test)

        for k, v in lr_metrics.items():
            mlflow.log_metric(f"lr_{k}", float(v))
        for k, v in xgb_metrics.items():
            mlflow.log_metric(f"xgb_{k}", float(v))

        mlflow.log_param("train_rows", int(len(x_train)))
        mlflow.log_param("val_rows", int(len(x_val)))
        mlflow.log_param("test_rows", int(len(x_test)))

        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(xgb, model_out)
        mlflow.log_artifact(str(model_out))

        print("Logistic Regression metrics:")
        for k, v in lr_metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nXGBoost metrics:")
        for k, v in xgb_metrics.items():
            print(f"  {k}: {v:.4f}")

        print(f"\nSaved best baseline artifact: {model_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline LR + XGBoost models.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Input features parquet path.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/baseline_xgb.pkl"),
        help="Path to save baseline XGBoost model artifact.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="",
        help="Optional MLflow tracking URI (e.g. sqlite:///mlruns.db).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="falabella_baseline",
        help="MLflow experiment name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        feature_path=args.features,
        model_out=args.model_out,
        random_state=args.seed,
        mlflow_uri=args.mlflow_uri if args.mlflow_uri else None,
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()
