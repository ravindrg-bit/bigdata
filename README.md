# Falabella Risk Engine

Hybrid credit risk modeling stack for thin-file lending scenarios.

## Current implementation status

- [x] Project scaffold
- [x] Synthetic data generation pipeline
- [x] Feature engineering pipeline
- [x] Baseline model training script
- [x] GraphSAGE training
- [x] Hybrid ensemble
- [x] Explainability module (predict + top drivers)
- [x] Cold-start routing module
- [x] Fairness audit module + report export
- [x] Federated simulation module (FedAvg) + benchmark report
- [x] Streamlit app with network, SHAP waterfall, fairness cards, override logging
- [x] Roadmap notebooks 01-06

## Quickstart

1. Use Python 3.11 and create a virtual environment:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate synthetic data:

```bash
python src/generate_data.py --seed 42 --output-dir data
```

4. Build feature table:

```bash
python src/feature_engineering.py --data-dir data --output data/features.parquet
```

5. Train tabular baseline:

```bash
python src/train_baseline.py --features data/features.parquet --model-out models/baseline_xgb.pkl
```

6. Train GraphSAGE and export embeddings:

```bash
python src/train_gnn.py --features data/features.parquet --edges data/edges.parquet --labels data/labels.parquet --model-out models/graphsage.pt --embeddings-out data/gnn_embeddings.parquet
```

7. Train hybrid ensemble:

```bash
python src/train_hybrid.py --features data/features.parquet --embeddings data/gnn_embeddings.parquet --model-out models/hybrid_ensemble.pkl
```

8. Run fairness audit:

```bash
python src/fairness_audit.py --features data/features.parquet --embeddings data/gnn_embeddings.parquet --model models/hybrid_ensemble.pkl --output-json data/fairness_report.json --output-csv data/fairness_summary.csv
```

9. Run federated simulation:

```bash
python src/federated_training.py --features data/features.parquet --model-out models/federated_model.pt --report-out data/federated_report.json --rounds 8 --local-epochs 2
```

10. Run dashboard:

```bash
streamlit run app.py
```

## Notebook deliverables

- notebooks/01_EDA_graph_construction.ipynb
- notebooks/02_baseline_model.ipynb
- notebooks/03_gnn_training.ipynb
- notebooks/04_hybrid_ensemble.ipynb
- notebooks/05_fairness_audit.ipynb
- notebooks/06_federated_simulation.ipynb

## Data artifacts generated

- borrowers.parquet
- loans.parquet
- repayments.parquet
- groups.parquet
- edges.parquet
- cdr.parquet
- mobile_events.parquet
- labels.parquet
- features.parquet
- gnn_embeddings.parquet

## Model artifacts generated

- models/baseline_xgb.pkl
- models/graphsage.pt
- models/hybrid_ensemble.pkl
- models/federated_model.pt

## Latest metrics snapshot

- Baseline XGBoost AUC: 0.9758
- Baseline Logistic Regression AUC: 0.9717
- Hybrid AUC: 0.9817
- Hybrid latency (ms/row): 0.0078
- Federated AUC: 0.9030
- Centralized reference AUC: 0.8984
- Federated-centralized AUC gap: -0.0046 (within 2-point target)
- Fairness parity gaps after mitigation:
	- gender_female_flag: 0.0016
	- rural_flag: 0.0456
	- indigenous_proxy: 0.0450

## Remaining external delivery steps

- Push repository to GitHub and attach public URL
- Deploy Streamlit app to Streamlit Community Cloud and add live link
- Record narrated demo walk-through
