# Tiered Cinnamon Forecast

Weekly sales-quantity forecasting for a Sri Lankan cinnamon exporter's product
catalog, built from ~3.5 years of transaction-level export data (~60K
transactions, 9,052 in-scope products, 91 destination countries). Products
are split into a **high-volume** tier and a **sparse/intermittent** tier,
each forecast with a method suited to its demand pattern, plus a
product-country drill-down demonstration on a small subset of products.

The full pipeline — cleaning, feature engineering, tiering, modeling,
backtesting, and figure generation — is reproducible end-to-end with a
single `make all`, runs entirely on CPU, and takes about 2 minutes on a
normal laptop.

## Results at a glance

Fold-1 time-ordered validation (`outputs/metrics/model_comparison.csv`):

| Tier | Model | Metric | Score | vs. baseline |
|---|---|---|---:|---|
| High-volume (1,525 products) | **LightGBM** (primary) | RMSE | 150.43 | beats seasonal-naive (360.58) by 2.4x |
| High-volume | XGBoost | RMSE | 155.96 | beats seasonal-naive by 2.3x |
| Sparse (7,527 products) | TSB (selected) | WMAPE | 3.14% | **loses** to naive baseline (2.01%) |
| Sparse | CrostonClassic / CrostonSBA | WMAPE | 21.4% / 20.3% | loses to naive baseline |

The sparse-tier result is reported as-is: the naive baseline beats every
primary intermittent-demand method on this validation fold. That finding,
and the reasoning behind it, is documented in
[`docs/submission/report.md`](docs/submission/report.md) rather than hidden
— this project's policy is that a model which doesn't beat its baseline
must say so.

No deep-learning model (LSTM/GRU/TFT) was used — see
[`docs/submission/report.md`](docs/submission/report.md) §5.3 for the
rejection rationale (data sparsity per series, CPU-only efficiency and
transparency requirements, and the project's own engineering constitution).

## Requirements

- Python 3.12 (pinned via `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- The raw workbook `Cinnamon_export_sales.xlsx`, placed at
  `data/raw/Cinnamon_export_sales.xlsx` (not committed to version control)

Dataset access note: the code is reproducible, but a fresh clone cannot run
the data pipeline unless a compatible workbook is placed at that path. See
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) for the required columns and
the reason the transaction-level workbook is not bundled by default.

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

For a reviewer-oriented walkthrough of the methodology, outputs, and modeling
decisions, see [`docs/PROJECT_EXPLAINED.md`](docs/PROJECT_EXPLAINED.md).

## Modeling approach

| Tier | Products | Primary model | Also compared | Baseline |
|---|---:|---|---|---|
| High-volume | 1,525 | LightGBM | XGBoost | Seasonal-naive |
| Sparse / intermittent | 7,527 | TSB | CrostonClassic, CrostonSBA | Naive (last value) |

- **Validation**: single time-ordered rolling-origin fold — train through
  2025-06-23, validate on the next 12 weeks (matching the forecast horizon).
  No random splits anywhere in the pipeline.
- **Features**: lag features (1/2/4/12 weeks), rolling mean/std (4/12
  weeks), calendar and Sri Lanka holiday features, and sparse-specific
  zero-streak-length features — all built strictly from past values, with
  regression tests guarding against leakage.
- **Why not deep learning**: most sparse-tier products have very few
  nonzero weeks of history (some sell once in three years), which is too
  little signal per series for an LSTM/GRU to learn from without a much
  larger global-model rewrite. CPU-only tree models and classical
  intermittent-demand methods train in under 1.3 seconds each, are fully
  auditable (SHAP importances, analytic Croston/TSB smoothing), and satisfy
  the project's efficiency and transparency requirements without GPU
  infrastructure. Full rationale in
  [`docs/submission/report.md`](docs/submission/report.md) §5.3.

## Tests

```bash
make test
```

Runs the `pytest` regression suite under `tests/`, covering feature-engineering
correctness (sparsity features, panel construction, leakage guards) with both
synthetic fixtures and checks against the regenerated processed data.

## Project layout

```
src/                  Pipeline modules
  cleaning.py         Clean raw transactions + weekly aggregation
  scoping.py          Merchandise/rubber-latex exclusion
  features.py         Feature engineering (lags, rolling stats, calendar, sparsity)
  tiering.py          High-volume / sparse product split
  metrics.py          RMSE / MAE / WMAPE / MASE
  backtest.py         Model comparison table
  assemble_forecast.py  Final 12-week forecast CSV assembly
  report_assets.py    Correlation / SHAP / loss / YoY figures
  models/             LightGBM, XGBoost, Croston/TSB, country drill-down

tests/                Pytest regression suite (36 tests)
notebooks/            Executed EDA, feature, and model-results notebooks
data/raw/             Raw input workbook (gitignored)
data/processed/       Generated Parquet intermediates (gitignored)
outputs/              Forecasts, metrics, figures (generated, gitignored)
specs/                Spec-driven-development specs, plans, and task logs
docs/                 Assignment brief, run guide, submission report/slides
Makefile              Pipeline, test, and reporting entry points
```

## License

[MIT](LICENSE)
