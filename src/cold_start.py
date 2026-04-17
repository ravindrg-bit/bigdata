from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ColdStartScorer:
    month2_model_path: Path | None = None
    month3_model_path: Path | None = None

    def __post_init__(self) -> None:
        self.month2_model = joblib.load(self.month2_model_path) if self.month2_model_path else None
        self.month3_model = joblib.load(self.month3_model_path) if self.month3_model_path else None

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
        x = row.to_frame().T
        return float(model.predict_proba(x)[0, 1])

    def score(self, row: pd.Series) -> dict[str, Any]:
        phase = self.determine_phase(row)

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
        else:
            if self.month3_model is None:
                risk = self._score_month_1(row)
                source = "rule_based_fallback"
            else:
                risk = self._score_model(self.month3_model, row)
                source = "full_hybrid"

        low, high = self._credit_line(phase, risk)
        decision = "approve" if risk < 0.5 else "decline"

        return {
            "phase": phase,
            "risk_score": risk,
            "decision": decision,
            "credit_line_range_mxn": [low, high],
            "model_source": source,
        }
