.PHONY: install lint format test train train-real serve docker clean

DATA ?= academic_success_dataset.csv

install:        ## editable install with dev tools
	pip install -e ".[dev]"
	pre-commit install || true

lint:           ## ruff lint + format check
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:           ## unit + integration tests with coverage
	pytest --cov=academic_success --cov-report=term-missing

train:          ## train on synthetic data (no dataset needed)
	asp train --out artifacts

train-real:     ## train on the UCI file: make train-real DATA=path.csv
	asp train --data $(DATA) --out artifacts

serve:
	asp serve --model artifacts/model.joblib

docker:
	docker build -t academic-success .

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .coverage build *.egg-info src/*.egg-info
