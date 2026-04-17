from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def code_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": lines,
    }


def md_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"language": "markdown"},
        "source": lines,
    }


def build_nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / name
    path.write_text(json.dumps(build_nb(cells), indent=2), encoding="utf-8")


def main() -> None:
    write_notebook(
        "03_gnn_training.ipynb",
        [
            md_cell([
                "# 03 - GNN Training (GraphSAGE vs GCN)",
                "Trains GraphSAGE and compares it against a GCN baseline.",
            ]),
            code_cell([
                "from pathlib import Path",
                "import os, sys, mlflow, numpy as np, pandas as pd, torch",
                "from torch_geometric.nn import GCNConv",
                "PROJECT_ROOT = Path.cwd()",
                "if not (PROJECT_ROOT / 'src').exists():",
                "    PROJECT_ROOT = PROJECT_ROOT.parent",
                "os.chdir(PROJECT_ROOT)",
                "SRC_ROOT = PROJECT_ROOT / 'src'",
                "if str(SRC_ROOT) not in sys.path:",
                "    sys.path.insert(0, str(SRC_ROOT))",
            ]),
            code_cell([
                "from falabella_risk.models.train_gnn import train_graphsage, build_graph_tensors, split_indices, compute_metrics, GraphSAGEClassifier",
                "train_graphsage(",
                "    feature_path=Path('data/processed/features.parquet'),",
                "    edges_path=Path('data/raw/edges.parquet'),",
                "    labels_path=Path('data/raw/labels.parquet'),",
                "    model_out=Path('models/graphsage.pt'),",
                "    embeddings_out=Path('data/processed/gnn_embeddings.parquet'),",
                "    seed=42, epochs=35, learning_rate=0.003,",
                ")",
            ]),
            code_cell([
                "features = pd.read_parquet('data/processed/features.parquet')",
                "edges = pd.read_parquet('data/raw/edges.parquet')",
                "labels = pd.read_parquet('data/raw/labels.parquet')",
                "x, edge_index, y, _ = build_graph_tensors(features, edges, labels)",
                "train_idx, val_idx, test_idx = split_indices(len(x), 42)",
                "",
                "class GCNClassifier(torch.nn.Module):",
                "    def __init__(self, in_channels, hidden=128):",
                "        super().__init__()",
                "        self.c1 = GCNConv(in_channels, hidden)",
                "        self.c2 = GCNConv(hidden, hidden)",
                "        self.out = torch.nn.Linear(hidden, 2)",
                "    def forward(self, x_t, ei):",
                "        h = torch.relu(self.c1(x_t, ei))",
                "        h = torch.relu(self.c2(h, ei))",
                "        return self.out(h)",
                "",
                "gcn = GCNClassifier(x.shape[1])",
                "opt = torch.optim.Adam(gcn.parameters(), lr=0.003, weight_decay=1e-4)",
                "loss_fn = torch.nn.CrossEntropyLoss()",
                "best_state, best_auc = None, -1.0",
                "for _ in range(35):",
                "    gcn.train(); opt.zero_grad()",
                "    logits = gcn(x, edge_index)",
                "    loss = loss_fn(logits[train_idx], y[train_idx]); loss.backward(); opt.step()",
                "    with torch.no_grad():",
                "        val_logits = gcn(x, edge_index)",
                "        val_auc = compute_metrics(val_logits, y, val_idx)['auc']",
                "    if val_auc > best_auc:",
                "        best_auc = val_auc",
                "        best_state = {k: v.detach().clone() for k, v in gcn.state_dict().items()}",
                "gcn.load_state_dict(best_state)",
                "",
                "with torch.no_grad():",
                "    gcn_metrics = compute_metrics(gcn(x, edge_index), y, test_idx)",
                "",
                "sage = GraphSAGEClassifier(in_channels=x.shape[1], hidden_channels=128)",
                "sage.load_state_dict(torch.load('models/graphsage.pt', map_location='cpu'))",
                "sage.eval()",
                "with torch.no_grad():",
                "    sage_logits, _ = sage(x, edge_index)",
                "    sage_metrics = compute_metrics(sage_logits, y, test_idx)",
                "",
                "mlflow.set_experiment('falabella_gnn_training')",
                "with mlflow.start_run(run_name='graphsage_vs_gcn_compare'):",
                "    mlflow.log_metric('graphsage_auc', float(sage_metrics['auc']))",
                "    mlflow.log_metric('graphsage_pr_auc', float(sage_metrics['pr_auc']))",
                "    mlflow.log_metric('gcn_auc', float(gcn_metrics['auc']))",
                "    mlflow.log_metric('gcn_pr_auc', float(gcn_metrics['pr_auc']))",
                "",
                "pd.DataFrame([{'model': 'GraphSAGE', **sage_metrics}, {'model': 'GCN', **gcn_metrics}])",
            ]),
        ],
    )

    write_notebook(
        "04_hybrid_ensemble.ipynb",
        [
            md_cell([
                "# 04 - Hybrid Ensemble",
                "Runs Optuna tuning and trains final hybrid XGBoost model.",
            ]),
            code_cell([
                "from pathlib import Path",
                "import os, sys, joblib, mlflow, optuna, pandas as pd",
                "from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score",
                "from sklearn.model_selection import train_test_split",
                "from xgboost import XGBClassifier",
                "PROJECT_ROOT = Path.cwd()",
                "if not (PROJECT_ROOT / 'src').exists():",
                "    PROJECT_ROOT = PROJECT_ROOT.parent",
                "os.chdir(PROJECT_ROOT)",
                "SRC_ROOT = PROJECT_ROOT / 'src'",
                "if str(SRC_ROOT) not in sys.path:",
                "    sys.path.insert(0, str(SRC_ROOT))",
            ]),
            code_cell([
                "features = pd.read_parquet('data/processed/features.parquet')",
                "emb = pd.read_parquet('data/processed/gnn_embeddings.parquet')",
                "df = features.merge(emb, on='borrower_id', how='inner')",
                "x = df.drop(columns=['borrower_id', 'default_flag'], errors='ignore')",
                "y = df['default_flag'].astype(int)",
                "x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)",
                "x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)",
                "",
                "def objective(trial):",
                "    params = {",
                "        'n_estimators': trial.suggest_int('n_estimators', 200, 700),",
                "        'max_depth': trial.suggest_int('max_depth', 3, 8),",
                "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),",
                "        'subsample': trial.suggest_float('subsample', 0.7, 1.0),",
                "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),",
                "        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),",
                "        'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': 42, 'n_jobs': -1,",
                "    }",
                "    model = XGBClassifier(**params)",
                "    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)",
                "    return roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])",
                "",
                "study = optuna.create_study(direction='maximize')",
                "study.optimize(objective, n_trials=20)",
                "best = study.best_params | {'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': 42, 'n_jobs': -1}",
                "model = XGBClassifier(**best)",
                "model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)",
                "proba = model.predict_proba(x_test)[:, 1]",
                "pred = (proba >= 0.5).astype(int)",
                "metrics = {'auc': float(roc_auc_score(y_test, proba)), 'pr_auc': float(average_precision_score(y_test, proba)), 'f1': float(f1_score(y_test, pred)), 'brier': float(brier_score_loss(y_test, proba))}",
                "mlflow.set_experiment('falabella_hybrid_ensemble')",
                "with mlflow.start_run(run_name='hybrid_optuna_best'):",
                "    mlflow.log_params(best)",
                "    for k, v in metrics.items(): mlflow.log_metric(k, float(v))",
                "    Path('models').mkdir(parents=True, exist_ok=True)",
                "    joblib.dump(model, 'models/hybrid_ensemble.pkl')",
                "    mlflow.log_artifact('models/hybrid_ensemble.pkl')",
                "metrics",
            ]),
        ],
    )

    write_notebook(
        "05_fairness_audit.ipynb",
        [
            md_cell([
                "# 05 - Fairness Audit",
                "Evaluates parity/equal-opportunity and applies threshold mitigation.",
            ]),
            code_cell([
                "from pathlib import Path",
                "import json, os, sys",
                "import matplotlib.pyplot as plt",
                "import pandas as pd",
                "import seaborn as sns",
                "PROJECT_ROOT = Path.cwd()",
                "if not (PROJECT_ROOT / 'src').exists():",
                "    PROJECT_ROOT = PROJECT_ROOT.parent",
                "os.chdir(PROJECT_ROOT)",
                "SRC_ROOT = PROJECT_ROOT / 'src'",
                "if str(SRC_ROOT) not in sys.path:",
                "    sys.path.insert(0, str(SRC_ROOT))",
                "sns.set_theme(style='whitegrid')",
            ]),
            code_cell([
                "from falabella_risk.evaluation.fairness_audit import run_fairness_audit",
                "report = run_fairness_audit(",
                "    features_path=Path('data/processed/features.parquet'),",
                "    embeddings_path=Path('data/processed/gnn_embeddings.parquet'),",
                "    model_path=Path('models/hybrid_ensemble.pkl'),",
                "    output_json=Path('reports/fairness_report.json'),",
                "    output_csv=Path('reports/fairness_summary.csv'),",
                "    seed=42,",
                ")",
                "report['pass_criteria']",
            ]),
            code_cell([
                "summary = pd.read_csv('reports/fairness_summary.csv')",
                "summary",
            ]),
            code_cell([
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
                "sns.barplot(data=summary, x='attribute', y='demographic_parity_gap', hue='stage', ax=axes[0])",
                "axes[0].axhline(0.05, color='red', linestyle='--')",
                "axes[0].set_title('Demographic Parity Gap')",
                "sns.barplot(data=summary, x='attribute', y='equal_opportunity_gap', hue='stage', ax=axes[1])",
                "axes[1].set_title('Equal Opportunity Gap')",
                "plt.tight_layout()",
                "plt.show()",
            ]),
            code_cell([
                "loaded = json.loads(Path('reports/fairness_report.json').read_text(encoding='utf-8'))",
                "list(loaded['calibration_curves'].keys())",
            ]),
        ],
    )

    write_notebook(
        "06_federated_simulation.ipynb",
        [
            md_cell([
                "# 06 - Federated Simulation",
                "Runs FedAvg training and compares against centralized baseline.",
            ]),
            code_cell([
                "from pathlib import Path",
                "import json, os, sys",
                "import pandas as pd",
                "PROJECT_ROOT = Path.cwd()",
                "if not (PROJECT_ROOT / 'src').exists():",
                "    PROJECT_ROOT = PROJECT_ROOT.parent",
                "os.chdir(PROJECT_ROOT)",
                "SRC_ROOT = PROJECT_ROOT / 'src'",
                "if str(SRC_ROOT) not in sys.path:",
                "    sys.path.insert(0, str(SRC_ROOT))",
            ]),
            code_cell([
                "from falabella_risk.federated.federated_training import train_federated",
                "report = train_federated(",
                "    features_path=Path('data/processed/features.parquet'),",
                "    model_out=Path('models/federated_model.pt'),",
                "    report_out=Path('reports/federated_report.json'),",
                "    seed=42, rounds=8, local_epochs=2, learning_rate=0.01,",
                ")",
                "report",
            ]),
            code_cell([
                "loaded = json.loads(Path('reports/federated_report.json').read_text(encoding='utf-8'))",
                "pd.DataFrame([",
                "    {'mode': 'federated', **loaded['federated_metrics']},",
                "    {'mode': 'centralized', **loaded['centralized_metrics']},",
                "])",
            ]),
            code_cell([
                "print('AUC gap vs centralized:', round(loaded['auc_gap_vs_centralized'], 4))",
                "print('Within 2-point target:', abs(loaded['auc_gap_vs_centralized']) <= 0.02)",
            ]),
        ],
    )

    print("Roadmap notebooks written:")
    print("- notebooks/03_gnn_training.ipynb")
    print("- notebooks/04_hybrid_ensemble.ipynb")
    print("- notebooks/05_fairness_audit.ipynb")
    print("- notebooks/06_federated_simulation.ipynb")


if __name__ == "__main__":
    main()
