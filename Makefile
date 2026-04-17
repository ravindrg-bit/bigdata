.PHONY: install install-dev format lint test clean data train-baseline train-gnn app notebook

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e '.[dev,notebooks]'

format:
	ruff check --fix src tests
	black src tests

lint:
	ruff check src tests
	black --check src tests

test:
	pytest

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

data:
	PYTHONPATH=src python -m falabella_risk.data.generate_data --seed 42 --output-dir data/raw
	PYTHONPATH=src python -m falabella_risk.features.feature_engineering --data-dir data/raw --output data/processed/features.parquet

train-baseline:
	PYTHONPATH=src python -m falabella_risk.models.train_baseline

train-gnn:
	PYTHONPATH=src python -m falabella_risk.models.train_gnn

app:
	PYTHONPATH=src python -m streamlit run src/falabella_risk/app/main.py

notebook:
	python -m jupyter lab
