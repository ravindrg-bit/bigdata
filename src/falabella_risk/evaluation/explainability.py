from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


DEFAULT_LABELS = {
    "neighborhood_default_rate_1hop": "Neighborhood default pressure",
    "neighborhood_default_rate_2hop": "Extended network risk",
    "repayment_latency_days": "Historical repayment delay",
    "routine_score": "Behavioral routine consistency",
    "call_volume_stability": "Call volume volatility",
    "CoDi_transaction_regularity": "CoDi usage regularity",
    "weighted_tie_strength": "Peer tie strength",
    "cohesion_score": "Group cohesion",
}


@dataclass
class ExplainabilityEngine:
    model_path: Path
    embeddings_path: Path | None = None
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.model = joblib.load(self.model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.model_features = list(getattr(self.model, "feature_names_in_", []))
        self.embeddings = None
        if self.embeddings_path and self.embeddings_path.exists():
            emb = pd.read_parquet(self.embeddings_path)
            if "borrower_id" in emb.columns:
                self.embeddings = emb.set_index("borrower_id")

    def _plain_label(self, feature_name: str) -> str:
        return DEFAULT_LABELS.get(feature_name, feature_name.replace("_", " ").title())

    def _prepare_input(self, x: pd.DataFrame) -> pd.DataFrame:
        data = x.copy()
        borrower_id = None
        if "borrower_id" in data.columns:
            borrower_id = int(float(data.iloc[0]["borrower_id"]))
            data = data.drop(columns=["borrower_id"], errors="ignore")

        if (
            self.embeddings is not None
            and borrower_id is not None
            and borrower_id in self.embeddings.index
        ):
            emb_row = self.embeddings.loc[borrower_id]
            if isinstance(emb_row, pd.DataFrame):
                emb_row = emb_row.iloc[0]
            emb_df = emb_row.to_frame().T.reset_index(drop=True)
            data = pd.concat([data.reset_index(drop=True), emb_df], axis=1)

        if self.model_features:
            for col in self.model_features:
                if col not in data.columns:
                    data[col] = 0.0
            data = data[self.model_features]

        return data.astype(float)

    def predict_explain(self, row: pd.Series | pd.DataFrame) -> dict[str, Any]:
        if isinstance(row, pd.Series):
            x = row.to_frame().T
        else:
            x = row.copy()

        x = self._prepare_input(x)

        start = perf_counter()
        prob = float(self.model.predict_proba(x)[0, 1])
        pred = int(prob >= self.threshold)

        shap_values = self.explainer.shap_values(x)
        if isinstance(shap_values, list):
            values = np.array(shap_values[1][0], dtype=float)
        else:
            values = np.array(shap_values[0], dtype=float)

        contributions = pd.DataFrame(
            {
                "feature": x.columns,
                "value": x.iloc[0].to_numpy(dtype=float),
                "shap_value": values,
            }
        )
        contributions["abs_shap"] = contributions["shap_value"].abs()
        top = contributions.sort_values("abs_shap", ascending=False).head(5)

        top_drivers = []
        for _, rec in top.iterrows():
            sign = "increased" if float(rec["shap_value"]) >= 0 else "reduced"
            top_drivers.append(
                {
                    "feature": str(rec["feature"]),
                    "label": self._plain_label(str(rec["feature"])),
                    "input_value": float(rec["value"]),
                    "shap_contribution": float(rec["shap_value"]),
                    "description": f"{self._plain_label(str(rec['feature']))} {sign} borrower risk.",
                }
            )

        latency_ms = (perf_counter() - start) * 1000.0
        result = {
            "risk_score": prob,
            "decision": "decline" if pred == 1 else "approve",
            "threshold": self.threshold,
            "top_drivers": top_drivers,
            "latency_ms": latency_ms,
        }

        json.dumps(result)
        return result

    def global_feature_importance(
        self,
        rows: pd.DataFrame,
        max_rows: int = 400,
    ) -> pd.DataFrame:
        if rows.empty:
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])

        data = rows.copy().head(max_rows)
        data = data.drop(columns=["default_flag"], errors="ignore")

        x = data.copy()
        if self.model_features:
            for col in self.model_features:
                if col not in x.columns:
                    x[col] = 0.0
            x = x[self.model_features]

        x = x.astype(float)

        shap_values = self.explainer.shap_values(x)
        if isinstance(shap_values, list):
            vals = np.array(shap_values[1], dtype=float)
        else:
            vals = np.array(shap_values, dtype=float)

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        mean_abs = np.abs(vals).mean(axis=0)
        out = pd.DataFrame({"feature": x.columns, "mean_abs_shap": mean_abs})
        out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return out
