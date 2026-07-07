# Tiered Cinnamon Forecast

Weekly sales-quantity forecasting for a Sri Lankan cinnamon exporter's product
catalog, built from ~3.5 years of transaction-level export data. Products are
split into a high-volume tier and a sparse/intermittent tier, each forecast
with a method suited to its demand pattern, plus a product-country
drill-down demonstration on a small subset of products.

## Requirements

- Python 3.12 (pinned via `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- The raw workbook `Cinnamon_export_sales.xlsx`, placed at
  `data/raw/Cinnamon_export_sales.xlsx` (not committed to version control)

## Setup

```bash
uv sync
```

## Pipeline

The pipeline runs as a sequence of independently re-runnable stages, each
reading and writing local Parquet files under `data/processed/`. Run the
whole thing with `make`, or invoke a single stage directly with `uv run`.

```bash
make pipeline      # clean -> scope -> weekly aggregate -> features -> tiering
make all           # full end-to-end: pipeline -> test -> forecast -> report -> notebooks
```

| Stage | Command | Output |
|---|---|---|
| Clean | `make clean-data` | `data/processed/cleaned.parquet` |
| Scope | `make scope` | `data/processed/scoped.parquet` |
| Weekly aggregate | `make weekly` | `data/processed/weekly.parquet` |
| Feature engineering | `make features` | `data/processed/featured.parquet` |
| Tiering | `make tiering` | `data/processed/tiers.parquet` |
| Forecast | `make forecast` | `outputs/forecasts/forecast_12wk.csv`, `outputs/metrics/model_comparison.csv` |
| Report figures | `make report` | `outputs/figures/*.png` |
| Notebooks | `make notebooks` | executed in place under `notebooks/` |

Run `make help` for the full list of targets.

For the full runbook, assignment summary, next implementation steps, report
outline, presentation outline, and speaking script draft, see
[`docs/run-and-submission-guide.md`](docs/run-and-submission-guide.md).

## Tests

```bash
make test
```

Runs the `pytest` regression suite under `tests/`, covering feature-engineering
correctness (sparsity features, panel construction, leakage guards) with both
synthetic fixtures and checks against the regenerated processed data.

## Project layout

```
src/            Pipeline modules (cleaning, scoping, feature engineering, tiering, models)
tests/          Pytest regression suite
data/raw/       Raw input workbook (gitignored)
data/processed/ Generated Parquet intermediates (gitignored)
Makefile        Pipeline and test entry points
```
