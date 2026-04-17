# Project Brief

## Objective
Build a hybrid credit risk model for thin-file lending using synthetic data, graph learning, and tabular ensemble modeling.

## Current Build Scope
- Synthetic data generation (8 parquet tables)
- Feature engineering (group, behavioral, graph-native)
- Baseline model training
- GraphSAGE + GCN comparison notebook flow
- Hybrid ensemble training with Optuna notebook path
- Explainability integration (top drivers + waterfall support)
- Fairness audit with mitigation and subgroup reports
- Federated simulation (FedAvg) with centralized benchmark
- Streamlit app (portfolio graph, borrower lookup, ethics panel)

## Key Generated Artifacts
1. Models: baseline_xgb.pkl, graphsage.pt, hybrid_ensemble.pkl, federated_model.pt
2. Reports: reports/fairness_report.json, reports/federated_report.json
3. Notebooks: roadmap 01-06 sequence available under notebooks/

## Remaining Milestones
1. Publish repository and deployment URLs.
2. Finalize executive collateral (PDFs + narrated demo video).
3. Add CI checks for notebook/script parity and artifact freshness.
