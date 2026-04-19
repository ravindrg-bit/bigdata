from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ColdStartScorer:
    month2_model_path: Path | None = None
    month3_model_path: Path | None = None
    month2_threshold_path: Path | None = None
    month3_threshold_path: Path | None = None
    embeddings_path: Path | None = None

    def __post_init__(self) -> None:
        self.month2_model = joblib.load(self.month2_model_path) if self.month2_model_path else None
        self.month3_model = joblib.load(self.month3_model_path) if self.month3_model_path else None
        self.month2_threshold = self._load_threshold(self.month2_threshold_path)
        self.month3_threshold = self._load_threshold(self.month3_threshold_path)
        self.embeddings = None
        if self.embeddings_path and self.embeddings_path.exists():
            emb = pd.read_parquet(self.embeddings_path)
            if "borrower_id" in emb.columns:
                self.embeddings = emb.set_index("borrower_id")

    @staticmethod
    def _load_threshold(path: Path | None) -> float:
        if path is None or not path.exists():
            return 0.5
        try:
            payload = json.loads(path.read_text())
            return float(payload.get("decision_threshold", 0.5))
        except Exception:
            return 0.5

    @staticmethod
    def determine_phase(row: pd.Series) -> str:
        store_ok = float(row.get("store_visit_count", 0)) >= 3
        codi_ok = float(row.get("CoDi_wallet_flag", 0)) >= 1
        social_ok = float(row.get("degree_centrality", 0)) > 0

        if store_ok and not codi_ok:
            return "month_1"
        if store_ok and codi_ok and not social_ok:
            return "months_2_3"
        if social_ok:
            return "month_3_plus"
        return "month_1"

    @staticmethod
    def _credit_line(phase: str, risk_score: float) -> tuple[int, int]:
        if phase == "month_1":
            base_low, base_high = 500, 2000
        elif phase == "months_2_3":
            base_low, base_high = 2000, 8000
        else:
            base_low, base_high = 8000, 25000

        spread = base_high - base_low
        adjusted = int(base_high - spread * risk_score)
        return base_low, max(base_low, adjusted)

    @staticmethod
    def _score_month_1(row: pd.Series) -> float:
        z = (
            -1.2
            + 0.07 * float(row.get("store_visit_count", 0))
            - 0.65 * float(row.get("INE_verified_flag", 0))
            - 0.35 * float(row.get("CoDi_wallet_flag", 0))
            + 0.42 * float(row.get("rural_flag", 0))
            + 0.55 * float(row.get("indigenous_proxy", 0))
        )
        return float(1.0 / (1.0 + np.exp(-z)))

    def _score_model(self, model: Any, row: pd.Series) -> float:
        x = row.to_frame().T.copy()

        borrower_id = None
        if "borrower_id" in x.columns:
            try:
                borrower_id = int(float(x.iloc[0]["borrower_id"]))
            except Exception:
                borrower_id = None

        if self.embeddings is not None and borrower_id is not None and borrower_id in self.embeddings.index:
            emb_row = self.embeddings.loc[borrower_id]
            if isinstance(emb_row, pd.DataFrame):
                emb_row = emb_row.iloc[0]
            emb_df = emb_row.to_frame().T.reset_index(drop=True)
            x = pd.concat([x.reset_index(drop=True), emb_df], axis=1)

        model_features = list(getattr(model, "feature_names_in_", []))
        if model_features:
            for col in model_features:
                if col not in x.columns:
                    x[col] = 0.0
            x = x[model_features]

        x = x.drop(columns=["default_flag"], errors="ignore").astype(float)
        return float(model.predict_proba(x)[0, 1])

    def score(self, row: pd.Series) -> dict[str, Any]:
        phase = self.determine_phase(row)
        threshold = 0.5

        if phase == "month_1":
            risk = self._score_month_1(row)
            source = "rule_based"
        elif phase == "months_2_3":
            if self.month2_model is None:
                risk = self._score_month_1(row)
                source = "rule_based_fallback"
            else:
                risk = self._score_model(self.month2_model, row)
                source = "hybrid_lite"
                threshold = self.month2_threshold
        else:
            if self.month3_model is None:
                risk = self._score_month_1(row)
                source = "rule_based_fallback"
            else:
                risk = self._score_model(self.month3_model, row)
                source = "full_hybrid"
                threshold = self.month3_threshold

        low, high = self._credit_line(phase, risk)
        decision = "approve" if risk < threshold else "decline"

        return {
            "phase": phase,
            "risk_score": risk,
            "decision": decision,
            "decision_threshold": float(threshold),
            "credit_line_range_mxn": [low, high],
            "model_source": source,
        }
