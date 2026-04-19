from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "imbalanced-learn is required for SMOTE stage. Install with `pip install imbalanced-learn`."
    ) from exc


LEAKY_FEATURES = {
    "repayment_latency_days",
    "on_time_repayment_share",
    "neighborhood_default_rate_1hop",
    "neighborhood_default_rate_2hop",
    "peer_default_contagion_score",
}


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    start = perf_counter()
    y_proba = model.predict_proba(x_test)[:, 1]
    elapsed_ms = (perf_counter() - start) * 1000.0 / max(len(x_test), 1)

    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "brier": brier_score_loss(y_test, y_proba),
        "latency_ms_per_row": elapsed_ms,
    }


def select_f1_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    idx = int(np.argmax(f1_scores))
    return float(thresholds[idx]), float(f1_scores[idx])


def evaluate_stage(
    name: str,
    model,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    val_proba = model.predict_proba(x_val)[:, 1]
    threshold, val_best_f1 = select_f1_threshold(y_val, val_proba)

    metrics_default = evaluate_model(model, x_test, y_test, threshold=0.5)
    metrics_best = evaluate_model(model, x_test, y_test, threshold=threshold)

    print(f"\n{name}:")
    print(f"  AUC: {metrics_default['auc']:.4f}")
    print(f"  F1@0.5: {metrics_default['f1']:.4f}")
    print(f"  best_threshold(val): {threshold:.4f}")
    print(f"  Val F1@best_threshold: {val_best_f1:.4f}")
    print(f"  Test F1@best_threshold: {metrics_best['f1']:.4f}")

    out = {
        "auc": metrics_default["auc"],
        "pr_auc": metrics_default["pr_auc"],
        "brier": metrics_default["brier"],
        "latency_ms_per_row": metrics_default["latency_ms_per_row"],
        "f1_default": metrics_default["f1"],
        "f1_best": metrics_best["f1"],
        "val_f1_best": val_best_f1,
        "best_threshold": threshold,
    }
    return out


def run_training(
    feature_path: Path,
    model_out: Path,
    random_state: int,
    mlflow_uri: str | None,
    experiment_name: str,
    drop_leaky_features: bool,
) -> None:
    df = pd.read_parquet(feature_path)
    if "default_flag" not in df.columns:
        raise ValueError("features.parquet must include default_flag column")

    x = df.drop(columns=["default_flag", "borrower_id"], errors="ignore")
    excluded_cols: list[str] = []
    if drop_leaky_features:
        excluded_cols = [col for col in x.columns if col in LEAKY_FEATURES]
        if excluded_cols:
            x = x.drop(columns=excluded_cols)

    if excluded_cols:
        print("Excluded potential leakage features:", ", ".join(sorted(excluded_cols)))
    print(f"Using {x.shape[1]} tabular features for XGBoost")

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

    with mlflow.start_run(run_name="baseline_xgb_imbalance_stages"):
        base_params = {
            "n_estimators": 450,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": random_state,
            "n_jobs": -1,
        }

        # Step 1: Base model + threshold optimization on holdout predictions.
        xgb_base = XGBClassifier(**base_params)
        xgb_base.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        step1_metrics = evaluate_stage("Step 1 (base)", xgb_base, x_val, y_val, x_test, y_test)

        # Step 2: scale_pos_weight using train-set class ratio.
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg / max(pos, 1)

        xgb_spw = XGBClassifier(**base_params, scale_pos_weight=scale_pos_weight)
        xgb_spw.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        step2_metrics = evaluate_stage(
            "Step 2 (scale_pos_weight)", xgb_spw, x_val, y_val, x_test, y_test
        )

        # Step 3: RandomizedSearchCV on train set, scoring=f1.
        search_space = {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        }
        xgb_for_search = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
        random_search = RandomizedSearchCV(
            estimator=xgb_for_search,
            param_distributions=search_space,
            n_iter=30,
            scoring="f1",
            cv=3,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
        random_search.fit(x_train, y_train)
        xgb_search = random_search.best_estimator_
        step3_metrics = evaluate_stage(
            "Step 3 (randomized search)", xgb_search, x_val, y_val, x_test, y_test
        )

        # Step 4: SMOTE on train only, retrain with best-search params.
        smote = SMOTE(random_state=random_state)
        x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
        smote_params = random_search.best_params_.copy()
        xgb_smote = XGBClassifier(
            **smote_params,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=-1,
            scale_pos_weight=1.0,
        )
        xgb_smote.fit(x_train_smote, y_train_smote)
        step4_metrics = evaluate_stage(
            "Step 4 (SMOTE + retrain)", xgb_smote, x_val, y_val, x_test, y_test
        )

        stage_metrics = {
            "step1_base": step1_metrics,
            "step2_scale_pos_weight": step2_metrics,
            "step3_randomized_search": step3_metrics,
            "step4_smote": step4_metrics,
        }

        # Select stage based on validation F1 only; keep test set untouched for final reporting.
        best_stage_name, best_stage = max(stage_metrics.items(), key=lambda item: item[1]["val_f1_best"])

        model_by_stage = {
            "step1_base": xgb_base,
            "step2_scale_pos_weight": xgb_spw,
            "step3_randomized_search": xgb_search,
            "step4_smote": xgb_smote,
        }
        best_model = model_by_stage[best_stage_name]

        mlflow.log_params(
            {
                "train_rows": int(len(x_train)),
                "val_rows": int(len(x_val)),
                "test_rows": int(len(x_test)),
                "base_n_estimators": base_params["n_estimators"],
                "base_max_depth": base_params["max_depth"],
                "base_learning_rate": base_params["learning_rate"],
                "base_subsample": base_params["subsample"],
                "base_colsample_bytree": base_params["colsample_bytree"],
                "base_reg_lambda": base_params["reg_lambda"],
                "scale_pos_weight": float(scale_pos_weight),
                "random_search_best_params": json.dumps(random_search.best_params_),
                "selected_stage": best_stage_name,
                "selected_threshold": float(best_stage["best_threshold"]),
                "drop_leaky_features": bool(drop_leaky_features),
            }
        )

        for stage_name, metrics in stage_metrics.items():
            mlflow.log_metrics(
                {
                    f"{stage_name}_auc": float(metrics["auc"]),
                    f"{stage_name}_pr_auc": float(metrics["pr_auc"]),
                    f"{stage_name}_brier": float(metrics["brier"]),
                    f"{stage_name}_latency_ms_per_row": float(metrics["latency_ms_per_row"]),
                    f"{stage_name}_f1_default": float(metrics["f1_default"]),
                    f"{stage_name}_f1_best": float(metrics["f1_best"]),
                    f"{stage_name}_val_f1_best": float(metrics["val_f1_best"]),
                }
            )

        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_out)
        mlflow.log_artifact(str(model_out))

        threshold_out = model_out.with_suffix(".threshold.json")
        threshold_out.write_text(
            json.dumps(
                {
                    "selected_stage": best_stage_name,
                    "decision_threshold": best_stage["best_threshold"],
                    "f1_at_best_threshold": best_stage["f1_best"],
                    "val_f1_at_best_threshold": best_stage["val_f1_best"],
                    "excluded_features": sorted(excluded_cols),
                    "stage_metrics": stage_metrics,
                },
                indent=2,
            )
        )
        mlflow.log_artifact(str(threshold_out))

        print("\nSummary:")
        for stage_name, metrics in stage_metrics.items():
            print(
                f"  {stage_name}: AUC={metrics['auc']:.4f}, "
                f"F1@0.5={metrics['f1_default']:.4f}, "
                f"F1@best={metrics['f1_best']:.4f}"
            )

        print(
            f"\nSelected stage: {best_stage_name} "
            f"(best_threshold={best_stage['best_threshold']:.4f}, "
            f"F1={best_stage['f1_best']:.4f})"
        )
        print(f"Saved baseline artifact: {model_out}")


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
    parser.add_argument(
        "--allow-leaky-features",
        action="store_true",
        help="Include known leakage-prone features (not recommended).",
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
        drop_leaky_features=not args.allow_leaky_features,
    )


if __name__ == "__main__":
    main()
