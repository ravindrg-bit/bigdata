# Cleanup Candidates

These items are likely safe to remove, but are intentionally not deleted without explicit approval.

## Candidate Paths

- `.venv/` (project-local virtual environment)
- `../.venv/` (parent-folder virtual environment)
- `mlruns/` (local MLflow runs and artifacts)
- `.vscode/settings.json` (local editor preferences; example is tracked as `.vscode/settings.json.example`)
- Any remaining `.DS_Store` files
- Any remaining `__pycache__/` directories and `*.pyc` files

## Notes

- Model binaries in `models/` are intentionally retained.
- Data parquet files are intentionally retained in `data/raw/` and `data/processed/`.
