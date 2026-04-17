# Phase 0 Audit

Date: 2026-04-17
Scope: No code or structure changes. Audit only.

## 1) Repository State

- `falabella-risk-engine` is currently not a Git repository.
- Per your rules, before any file moves in later phases we must:
  1. run `git init` inside `falabella-risk-engine`
  2. create an initial commit of the current state
  3. create and work on branch `refactor/repo-structure`

## 2) Current Tree (excluded: `.venv`, `__pycache__`, `mlruns`, `.git`)

```text
.
./demo
./demo/narration.aiff
./demo/narration_script.txt
./demo/demo_video.mp4
./.DS_Store
./LICENSE
./requirements.txt
./models
./models/baseline_xgb.pkl
./models/graphsage.pt
./models/federated_model.pt
./models/hybrid_ensemble.pkl
./docs
./docs/market_entry_playbook.pdf
./docs/project_brief.md
./docs/model_card.md
./docs/executive_report.pdf
./README.md
./.gitignore
./scripts
./scripts/generate_executive_assets.py
./scripts/write_roadmap_notebooks.py
./app.py
./.vscode
./.vscode/settings.json
./data
./data/borrowers.parquet
./data/fairness_summary.csv
./data/labels.parquet
./data/groups.parquet
./data/fairness_report.json
./data/federated_report.json
./data/repayments.parquet
./data/loans.parquet
./data/edges.parquet
./data/cdr.parquet
./data/gnn_embeddings.parquet
./data/mobile_events.parquet
./data/features.parquet
./notebooks
./notebooks/04_train_gnn.ipynb
./notebooks/03_gnn_training.ipynb
./notebooks/02_feature_engineering.ipynb
./notebooks/01_generate_data.ipynb
./notebooks/04_hybrid_ensemble.ipynb
./notebooks/05_fairness_audit.ipynb
./notebooks/01_EDA_graph_construction.ipynb
./notebooks/02_baseline_model.ipynb
./notebooks/06_explainability_cold_start.ipynb
./notebooks/05_train_hybrid.ipynb
./notebooks/06_federated_simulation.ipynb
./notebooks/03_train_baseline.ipynb
./src
./src/feature_engineering.py
./src/train_hybrid.py
./src/__init__.py
./src/train_gnn.py
./src/generate_data.py
./src/train_baseline.py
./src/federated_training.py
./src/cold_start.py
./src/fairness_audit.py
./src/explainability.py
./src/dashboard_components.py
```

## 3) Generated Artifacts / Ignore Candidates

These are present and should be gitignored and/or removed from tracking:

- OS cruft: `.DS_Store`
- Python cache: `__pycache__/`, `*.pyc`
- Virtual environments: `.venv/` (both parent and project-local)
- Experiment tracking outputs: `mlruns/`
- Model artifacts: `models/*.pt`, `models/*.pkl`
- Jupyter checkpoints: `.ipynb_checkpoints/` (pattern should be in `.gitignore`)

`.vscode/settings.json` currently uses `${workspaceFolder}/.venv/bin/python` and does not contain machine-specific absolute paths.

## 4) Duplicate Notebook Streams (Do Not Move Yet)

Two numbering tracks are currently mixed in one folder.

Stream A (pipeline track):
- `01_generate_data.ipynb`
- `02_feature_engineering.ipynb`
- `03_train_baseline.ipynb`
- `04_train_gnn.ipynb`
- `05_train_hybrid.ipynb`
- `06_federated_simulation.ipynb`

Stream B (exploratory/analysis/audit track):
- `01_EDA_graph_construction.ipynb`
- `02_baseline_model.ipynb`
- `03_gnn_training.ipynb`
- `04_hybrid_ensemble.ipynb`
- `05_fairness_audit.ipynb`
- `06_explainability_cold_start.ipynb`

Proposed split (pending your confirmation):
- `notebooks/pipeline/` <- Stream A
- `notebooks/exploratory/` <- Stream B

## 5) Parent-Folder Files Outside Project

Outside `falabella-risk-engine/` but in parent folder:
- `BigData.md`
- `read.me`

These need your direction: move into project, move to sibling docs folder, or keep in parent root.

## 6) Two `.venv` Directories

Detected:
- parent-level `.venv/`
- project-level `falabella-risk-engine/.venv/`

Neither should be tracked in Git.

## 7) Required Decisions Before Phase 1+

Please confirm all five items:

1. Notebook split confirmation: keep Stream A -> `pipeline/` and Stream B -> `exploratory/`?
2. `BigData.md` and `read.me`: where should they live?
3. `.venv` directories: only untrack/ignore, or also delete from disk later?
4. `data/*.parquet`: are these raw, processed, or mixed (for `data/raw` vs `data/processed` split)?
5. `.github/workflows/`: keep empty placeholder only, or scaffold CI now?
