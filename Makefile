# Tiered Cinnamon Forecast — critical commands
# Run all targets from the repo root. Data pipeline reads/writes under data/.
# uv manages the environment; `make setup` first if the venv is missing.

.DEFAULT_GOAL := help
.PHONY: help setup clean-data scope weekly features tiering pipeline verify test forecast report notebooks fmt clean all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Sync the uv environment from the lockfile
	uv sync

## --- Data pipeline (stage order matters; see comments) ---

clean-data: ## Stage 2: clean raw xlsx -> data/processed/cleaned.parquet
	uv run python -m src.cleaning

scope: clean-data ## Stage 3: merch/rubber exclusion -> data/processed/scoped.parquet
	uv run python -m src.scoping

weekly: scope ## Stage 4: weekly aggregation -> data/processed/weekly.parquet
	# cleaning.py also emits weekly.parquet once scoped.parquet exists
	uv run python -m src.cleaning

features: weekly ## Stage 5: feature engineering -> data/processed/featured.parquet
	uv run python -m src.features

tiering: features ## Stage 6: >=10-txn tier split -> data/processed/tiers.parquet
	uv run python -m src.tiering

pipeline: tiering ## Run the full data pipeline end-to-end (Stages 2-6)
	@echo "Data pipeline complete: cleaned -> scoped -> weekly -> featured -> tiers"

## --- Modeling / reporting ---

forecast: ## Train models, backtest, and assemble outputs/forecasts/forecast_12wk.csv (T005/T006/T007/T008/T010)
	uv run python -m src.models.high_volume
	uv run python -m src.models.sparse
	uv run python -m src.models.country_demo
	uv run python -m src.backtest
	uv run python -m src.assemble_forecast

report: ## Generate rubric figures under outputs/figures (T011)
	uv run python -m src.report_assets

notebooks: ## Execute all notebooks in place (T003/T012)
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

## --- Quality ---

verify: ## Re-run the pipeline's built-in assertions (leakage, partition, counts)
	$(MAKE) pipeline

test: ## Run the test suite if one exists
	@test -d tests && uv run pytest -q || echo "no tests/ directory yet"

all: pipeline test forecast report notebooks ## Full end-to-end run: data pipeline -> tests -> models/backtest/forecast -> figures -> notebooks
	@echo "End-to-end run complete: data pipeline, tests, forecast CSV, report figures, and notebooks are all up to date."

clean: ## Remove generated processed data and outputs (keeps raw)
	rm -f data/processed/*.parquet
	rm -rf outputs/figures/* outputs/forecasts/*
