# Falabella Risk Engine

## Overview

Falabella Risk Engine is a credit-risk modeling project for thin-file lending using:

- Tabular baseline models
- Graph neural modeling (GraphSAGE)
- Hybrid ensemble modeling
- Federated training simulation
- Fairness auditing and explainability
- Streamlit-based product demo

The repository follows a src-layout package and a data-science oriented structure aligned with cookiecutter-style conventions.

## Project Structure

```text
falabella-risk-engine/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── external/
│   └── README.md
├── demo/
├── docs/
├── models/
│   └── README.md
├── notebooks/
│   ├── exploratory/
│   └── pipeline/
├── reports/
│   └── figures/
├── scripts/
├── src/
│   └── falabella_risk/
│       ├── app/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── federated/
│       ├── inference/
│       └── models/
└── tests/
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Optional notebook dependencies:

```bash
pip install -e ".[notebooks]"
```

You can also use Make targets:

```bash
make install-dev
```

## Data

Data layout and schema documentation are in data/README.md.

In short:

- data/raw contains generated source parquet tables
- data/processed contains feature and embedding artifacts used for training
- reports contains generated JSON/CSV outputs from fairness and federated workflows

## Running the pipeline

End-to-end training flow:

```bash
make data
make train-baseline
make train-gnn
PYTHONPATH=src python -m falabella_risk.models.train_hybrid
PYTHONPATH=src python -m falabella_risk.evaluation.fairness_audit
PYTHONPATH=src python -m falabella_risk.federated.federated_training
```

Run the Streamlit app:

```bash
make app
```

Or directly:

```bash
PYTHONPATH=src python -m streamlit run src/falabella_risk/app/main.py
```

## Notebooks guide

Notebooks are intentionally split into two tracks:

- notebooks/pipeline: sequential training workflow
	- 01_generate_data.ipynb
	- 02_feature_engineering.ipynb
	- 03_train_baseline.ipynb
	- 04_train_gnn.ipynb
	- 05_train_hybrid.ipynb
	- 06_federated_simulation.ipynb
- notebooks/exploratory: analysis, comparisons, fairness, and explainability
	- 01_EDA_graph_construction.ipynb
	- 02_baseline_model.ipynb
	- 03_gnn_training.ipynb
	- 04_hybrid_ensemble.ipynb
	- 05_fairness_audit.ipynb
	- 06_explainability_cold_start.ipynb

## Tests

Run tests with:

```bash
make test
```

Current tests include placeholder stubs for core modules and a test harness-ready structure under tests/.

## API scoring policy

The FastAPI service under api/ keeps endpoint contracts stable for frontend clients while adding
traceability and conservative manual cold-start behavior:

- `POST /predict/existing/{borrower_id}` uses precomputed tabular features + real graph embeddings.
- `POST /predict/manual` accepts the existing manual form payload and adds optional
	`graph_verification` metadata.
- Without successful graph verification, manual scoring uses:
	- training-set medians for unavailable historical fields,
	- conservative phase gating (no phase-3 promotion from self-reported peer fields alone),
	- zero embeddings with exact model embedding dimension.
- `POST /debug/compare/existing-vs-manual/{borrower_id}` provides internal diagnostics:
	38-feature side-by-side mapping, source counts, embedding stats, prediction comparison,
	and SHAP top-driver overlap/divergence.
- `GET /model-info` and `GET /schema` expose model/schema/policy versions and manual scoring
	observability notes for UI integration and audit trails.

## Reproducibility notes

- Use fixed seeds in CLI commands (default seed is 42 across modules).
- Use editable install plus pinned top-level dependencies.
- Canonical package metadata is in pyproject.toml.
- requirements.txt is retained as a transitional pinned top-level dependency file.
- MLflow run outputs are intentionally local and gitignored.

## License

This project is licensed under the MIT License. See LICENSE for details.
