# Models README

## Purpose

This directory stores serialized model artifacts produced by training scripts.
Binary model artifacts are gitignored in normal workflows; this README documents how each file is produced.

## Artifact catalog

| Artifact | Produced by | Input data |
|---|---|---|
| models/baseline_xgb.pkl | src/falabella_risk/models/train_baseline.py | data/processed/features.parquet |
| models/graphsage.pt | src/falabella_risk/models/train_gnn.py | data/processed/features.parquet, data/raw/edges.parquet, data/raw/labels.parquet |
| models/hybrid_ensemble.pkl | src/falabella_risk/models/train_hybrid.py | data/processed/features.parquet, data/processed/gnn_embeddings.parquet |
| models/federated_model.pt | src/falabella_risk/federated/federated_training.py | data/processed/features.parquet |

## Reproduce all model binaries

Run from repository root:

```bash
make data
make train-baseline
make train-gnn
PYTHONPATH=src python -m falabella_risk.models.train_hybrid
PYTHONPATH=src python -m falabella_risk.federated.federated_training
```

## Naming convention

- baseline_xgb.pkl: tabular baseline XGBoost model
- graphsage.pt: GraphSAGE weights
- hybrid_ensemble.pkl: stacked hybrid XGBoost model over tabular plus graph embeddings
- federated_model.pt: federated linear model payload (weights plus feature/scaler metadata)

## Notes

- Output reports tied to these models are written to reports/.
- If model interfaces change, update this file and docs/model_card.md together.
