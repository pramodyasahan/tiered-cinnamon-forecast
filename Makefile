# Tiered Cinnamon Forecast — critical commands
# Run all targets from the repo root. Data pipeline reads/writes under data/.
# uv manages the environment; `make setup` first if the venv is missing.

.DEFAULT_GOAL := help
.PHONY: help setup clean-data scope weekly features tiering pipeline verify test forecast report notebooks fmt clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Sync the uv environment from the lockfile
	uv sync

## --- Data pipeline (stage order matters; see comments) ---

clean-data: ## Stage 2: clean raw xlsx -> data/processed/cleaned.parquet
	uv run src/cleaning.py

scope: clean-data ## Stage 3: merch/rubber exclusion -> data/processed/scoped.parquet
	uv run src/scoping.py

weekly: scope ## Stage 4: weekly aggregation -> data/processed/weekly.parquet
	# cleaning.py also emits weekly.parquet once scoped.parquet exists
	uv run src/cleaning.py

features: weekly ## Stage 5: feature engineering -> data/processed/featured.parquet
	uv run src/features.py

tiering: features ## Stage 6: >=10-txn tier split -> data/processed/tiers.parquet
	uv run src/tiering.py

pipeline: tiering ## Run the full data pipeline end-to-end (Stages 2-6)
	@echo "Data pipeline complete: cleaned -> scoped -> weekly -> featured -> tiers"

## --- Modeling / reporting (pending T008-T014; targets ready once scripts exist) ---

forecast: ## Produce outputs/forecasts/forecast_12wk.csv (T008/T009/T013)
	@test -f src/models/high_volume.py || { echo "src/models/high_volume.py not built yet (T008)"; exit 1; }
	uv run src/models/high_volume.py
	uv run src/models/sparse.py
	uv run src/models/country_demo.py

report: ## Generate rubric figures under outputs/figures (T012)
	@test -f src/report_assets.py || { echo "src/report_assets.py not built yet (T012)"; exit 1; }
	uv run src/report_assets.py

notebooks: ## Execute EDA/result notebooks in place (T007/T014)
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

## --- Quality ---

verify: ## Re-run the pipeline's built-in assertions (leakage, partition, counts)
	$(MAKE) pipeline

test: ## Run the test suite if one exists
	@test -d tests && uv run pytest -q || echo "no tests/ directory yet"

clean: ## Remove generated processed data and outputs (keeps raw)
	rm -f data/processed/*.parquet
	rm -rf outputs/figures/* outputs/forecasts/*
