from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

SENSITIVE_COLUMNS = ["gender_female_flag", "rural_flag", "indigenous_proxy"]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    auc = safe_auc(y_true, y_score)
    pr_auc = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "brier": brier,
        "positive_rate": float(y_pred.mean()),
    }


def group_fairness_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    group_values: np.ndarray,
    group_name: str,
) -> dict[str, object]:
    rows: list[dict[str, float | int | None]] = []

    unique_groups = sorted(np.unique(group_values).tolist())
    for value in unique_groups:
        mask = group_values == value
        y_t = y_true[mask]
        y_s = y_score[mask]
        y_p = y_pred[mask]

        positives = int((y_t == 1).sum())
        true_positive = int(((y_t == 1) & (y_p == 1)).sum())
        tpr = float(true_positive / positives) if positives > 0 else 0.0

        rows.append(
            {
                "group": int(value),
                "count": int(mask.sum()),
                "positive_rate": float(y_p.mean()),
                "true_positive_rate": tpr,
                "auc": safe_auc(y_t, y_s),
            }
        )

    positive_rates = [float(r["positive_rate"]) for r in rows]
    tprs = [float(r["true_positive_rate"]) for r in rows]

    return {
        "attribute": group_name,
        "demographic_parity_gap": float(max(positive_rates) - min(positive_rates)),
        "equal_opportunity_gap": float(max(tprs) - min(tprs)),
        "groups": rows,
    }


def compute_calibration_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_values: np.ndarray,
    bins: int = 10,
) -> dict[str, list[dict[str, float | int]]]:
    out: dict[str, list[dict[str, float | int]]] = {}
    edges = np.linspace(0.0, 1.0, bins + 1)

    for value in sorted(np.unique(group_values).tolist()):
        mask = group_values == value
        y_t = y_true[mask]
        y_s = y_score[mask]

        group_bins: list[dict[str, float | int]] = []
        for i in range(bins):
            left = float(edges[i])
            right = float(edges[i + 1])
            in_bin = (y_s >= left) & (y_s < right if i < bins - 1 else y_s <= right)

            if not np.any(in_bin):
                continue

            group_bins.append(
                {
                    "bin_start": left,
                    "bin_end": right,
                    "count": int(in_bin.sum()),
                    "mean_pred": float(y_s[in_bin].mean()),
                    "obs_default_rate": float(y_t[in_bin].mean()),
                }
            )

        out[str(int(value))] = group_bins

    return out


def fit_threshold_maps(
    y_score: np.ndarray,
    sensitive: dict[str, np.ndarray],
    target_positive_rate: float,
) -> dict[str, dict[str, float]]:
    target_positive_rate = float(np.clip(target_positive_rate, 0.01, 0.99))
    threshold_maps: dict[str, dict[str, float]] = {}

    for attr, values in sensitive.items():
        threshold_maps[attr] = {}
        for value in sorted(np.unique(values).tolist()):
            mask = values == value
            scores = y_score[mask]
            if len(scores) == 0:
                threshold = 0.5
            else:
                threshold = float(np.quantile(scores, 1.0 - target_positive_rate))
            threshold_maps[attr][str(int(value))] = threshold

    return threshold_maps


def predict_with_threshold_maps(
    y_score: np.ndarray,
    sensitive: dict[str, np.ndarray],
    threshold_maps: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    threshold_row = np.zeros_like(y_score, dtype=float)

    for attr, values in sensitive.items():
        mapping = threshold_maps[attr]
        threshold_row += np.array([mapping[str(int(v))] for v in values], dtype=float)

    threshold_row /= max(1, len(sensitive))
    y_pred = (y_score >= threshold_row).astype(int)
    return y_pred, threshold_row


def prepare_model_matrix(model, df: pd.DataFrame) -> pd.DataFrame:
    model_features = list(getattr(model, "feature_names_in_", []))
    x = df.drop(columns=["default_flag"], errors="ignore").copy()

    if model_features:
        for col in model_features:
            if col not in x.columns:
                x[col] = 0.0
        x = x[model_features]

    return x.astype(float)


def run_fairness_audit(
    features_path: Path,
    embeddings_path: Path,
    model_path: Path,
    output_json: Path,
    output_csv: Path,
    seed: int,
) -> dict[str, object]:
    features = pd.read_parquet(features_path)
    embeddings = pd.read_parquet(embeddings_path)

    merged = features.merge(embeddings, on="borrower_id", how="inner")
    if merged.empty:
        raise RuntimeError("No rows available after merging features and embeddings.")

    for col in SENSITIVE_COLUMNS + ["default_flag"]:
        if col not in merged.columns:
            raise ValueError(f"Required column missing for fairness audit: {col}")

    y = merged["default_flag"].astype(int).to_numpy()
    indices = np.arange(len(merged))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices,
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

    test_df = merged.iloc[idx_test].reset_index(drop=True)
    y_true = test_df["default_flag"].astype(int).to_numpy()

    model = joblib.load(model_path)
    x_test = prepare_model_matrix(model, test_df)
    y_score = model.predict_proba(x_test)[:, 1]

    y_pred_base = (y_score >= 0.5).astype(int)
    baseline_metrics = classification_metrics(y_true, y_score, y_pred_base)

    sensitive_test = {
        col: test_df[col].astype(int).to_numpy() for col in SENSITIVE_COLUMNS
    }

    baseline_fairness = {
        col: group_fairness_summary(y_true, y_score, y_pred_base, values, col)
        for col, values in sensitive_test.items()
    }

    target_positive_rate = float(y_pred_base.mean())
    threshold_maps = fit_threshold_maps(y_score, sensitive_test, target_positive_rate=target_positive_rate)
    y_pred_mitigated, threshold_row = predict_with_threshold_maps(y_score, sensitive_test, threshold_maps)

    mitigated_metrics = classification_metrics(y_true, y_score, y_pred_mitigated)
    mitigated_fairness = {
        col: group_fairness_summary(y_true, y_score, y_pred_mitigated, values, col)
        for col, values in sensitive_test.items()
    }

    calibration = {
        col: compute_calibration_by_group(y_true, y_score, values)
        for col, values in sensitive_test.items()
    }

    gaps = [
        float(mitigated_fairness[col]["demographic_parity_gap"]) for col in SENSITIVE_COLUMNS
    ]

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset": {
            "total_rows": int(len(merged)),
            "train_rows": int(len(idx_train)),
            "val_rows": int(len(idx_val)),
            "test_rows": int(len(idx_test)),
            "default_rate_test": float(y_true.mean()),
        },
        "baseline": {
            "classification_metrics": baseline_metrics,
            "fairness": baseline_fairness,
        },
        "mitigation": {
            "strategy": "intersectional_threshold_calibration",
            "threshold_maps": threshold_maps,
            "threshold_min": float(threshold_row.min()),
            "threshold_max": float(threshold_row.max()),
            "classification_metrics": mitigated_metrics,
            "fairness": mitigated_fairness,
        },
        "calibration_curves": calibration,
        "pass_criteria": {
            "parity_gap_threshold": 0.05,
            "all_parity_gaps_below_threshold": bool(all(g < 0.05 for g in gaps)),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    for stage_name, fairness in [
        ("baseline", baseline_fairness),
        ("mitigated", mitigated_fairness),
    ]:
        for attr, values in fairness.items():
            rows.append(
                {
                    "stage": stage_name,
                    "attribute": attr,
                    "demographic_parity_gap": values["demographic_parity_gap"],
                    "equal_opportunity_gap": values["equal_opportunity_gap"],
                }
            )

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fairness audit for hybrid risk model.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features.parquet"),
        help="Path to engineered features table.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/gnn_embeddings.parquet"),
        help="Path to GraphSAGE embeddings.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hybrid_ensemble.pkl"),
        help="Path to trained hybrid model artifact.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/fairness_report.json"),
        help="Path for fairness report JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/fairness_summary.csv"),
        help="Path for flat fairness summary CSV.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_fairness_audit(
        features_path=args.features,
        embeddings_path=args.embeddings,
        model_path=args.model,
        output_json=args.output_json,
        output_csv=args.output_csv,
        seed=args.seed,
    )

    print("Fairness audit completed.")
    print(f"Output JSON: {args.output_json}")
    print(f"Output CSV: {args.output_csv}")
    print(
        "Mitigated parity gaps:",
        {k: round(v["demographic_parity_gap"], 4) for k, v in report["mitigation"]["fairness"].items()},
    )


if __name__ == "__main__":
    main()
