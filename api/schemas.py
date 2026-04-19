from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BorrowerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Identity and demographics
    age: int = Field(..., ge=18, le=72, description="Borrower age")
    gender: Literal["Female", "Male"] = Field(..., description="Borrower gender")
    rural_flag: Literal[0, 1] = Field(..., description="1 if rural, else 0")
    indigenous_proxy: Literal[0, 1] = Field(..., description="1 if indigenous proxy, else 0")
    INE_verified_flag: Literal[0, 1] = Field(..., description="1 if INE is verified, else 0")
    married_flag: Literal[0, 1] = Field(..., description="1 if married/in partnership, else 0")
    num_children: int = Field(..., ge=0, le=6, description="Number of children")

    # Financial profile
    monthly_income_MXN: float = Field(..., ge=1200, le=150000, description="Monthly income in MXN")
    rent_MXN: float = Field(..., ge=0, le=15000, description="Monthly rent in MXN")
    informal_loans_flag: Literal[0, 1] = Field(..., description="1 if informal loans active, else 0")
    formal_debt_flag: Literal[0, 1] = Field(..., description="1 if formal debt active, else 0")
    electricity_water_MXN: float = Field(..., ge=0, le=2000, description="Utilities spend in MXN")

    # Banco Falabella relationship
    store_visit_count: int = Field(..., ge=0, le=30, description="Store visits")
    prior_CMR_usage: int = Field(..., ge=0, le=14, description="Prior CMR usage count")
    CoDi_wallet_flag: Literal[0, 1] = Field(..., description="1 if CoDi wallet active, else 0")

    # Behavioral signals
    call_routine_score: float = Field(..., ge=0.0, le=1.0, description="Call routine score")
    messaging_frequency: float = Field(..., ge=0, le=200, description="Messaging frequency")
    call_volume_stability: float = Field(..., ge=0.0, le=1.0, description="Call volume stability")
    app_opens: int = Field(..., ge=0, le=100, description="App opens")
    location_variance: float = Field(..., ge=0.0, le=1.0, description="Location variance")

    # Social network
    peer_connections: int = Field(..., ge=0, le=20, description="Peer connection count")
    group_cohesion_score: float = Field(..., ge=0.0, le=1.0, description="Group cohesion score")
    neighborhood_default_rate: float = Field(..., ge=0.0, le=1.0, description="Neighborhood default rate")


class TopDriver(BaseModel):
    feature: str
    value: float
    shap_contribution: float
    direction: Literal["increased risk", "reduced risk"]


class PredictionResponse(BaseModel):
    risk_score: float
    decision: Literal["approve", "decline"]
    cold_start_phase: Literal[1, 2, 3]
    credit_line_MXN: int
    top_drivers: list[TopDriver]
    explanation: str
