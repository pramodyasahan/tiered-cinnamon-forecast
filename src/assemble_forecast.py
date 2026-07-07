"""Forecast CSV assembly (spec 003 FR-24..FR-28, AC-4, plan §4.6).

Combines the already-completed high-volume (`src/models/high_volume.py`,
T005) and sparse (`src/models/sparse.py`, T006) final forecasts into a
single 12-week product-level forecast file with the schema required by
spec 001's data model / spec 003 §6:

    ProductID, forecast_week, WeekStart, predicted_Sales_Qty, tier, model

Each tier's final-forecast artifact carries *multiple* candidate models
(high-volume: lightgbm/xgboost/seasonal_naive; sparse: CrostonClassic/
CrostonSBA/TSB/Naive) -- this module selects exactly ONE model per tier
for the assembled submission file, reusing the choice already documented
elsewhere rather than inventing a new one:

- High-volume -> ``lightgbm``. `src/models/high_volume.py` explicitly
  names this the primary model in its module docstring and in the
  "Write final forecast (primary model: lightgbm)" stage / `primary_model`
  metric it logs.
- Sparse -> ``TSB``. `src/models/sparse.py` itself does not label a
  primary model in code (all four candidates are written out identically),
  so the choice is made here from `outputs/metrics/model_comparison.csv`
  (T008 backtest): among the three non-baseline intermittent-demand
  candidates, TSB has the best Fold-1 validation WMAPE/MASE (WMAPE=3.14,
  MASE=3.11), clearly ahead of CrostonSBA (WMAPE=20.34, MASE=34.44) and
  CrostonClassic (WMAPE=21.37, MASE=36.11). Per FR-23/EC-6, the honest
  finding that the `Naive` baseline actually beats all three on Fold 1
  is reported in the backtest table and submission report (T008/T013),
  not hidden here -- but the *baseline* is not substituted as the
  assembled "chosen" forecast, since the task is to assemble each tier's
  primary/candidate model output, matching the high-volume tier's
  lightgbm-over-seasonal_naive choice for consistency.

`data/processed/tiers.parquet` defines the in-scope product universe and
tier assignment. Every ProductID in `tiers.parquet` must appear in the
assembled output exactly `FORECAST_HORIZON` (12) times, with only
nonnegative predictions -- these are hard assertions raised *before* the
CSV is written (plan §13 "forecast row count is short" mitigation,
generalized to negativity and per-product coverage too).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.logging_utils import get_logger, log_metrics, stage

logger = get_logger(__name__)

TIERS_PATH = Path("data/processed/tiers.parquet")
HIGH_VOLUME_FORECAST_PATH = Path("outputs/models/high_volume/final_forecast.parquet")
SPARSE_FORECAST_PATH = Path("outputs/models/sparse/final_forecast.parquet")
OUTPUT_PATH = Path("outputs/forecasts/forecast_12wk.csv")

FORECAST_HORIZON = 12

HIGH_VOLUME_MODEL_CHOICE = "lightgbm"
SPARSE_MODEL_CHOICE = "TSB"

OUTPUT_COLUMNS = [
    "ProductID",
    "forecast_week",
    "WeekStart",
    "predicted_Sales_Qty",
    "tier",
    "model",
]


def select_high_volume_forecast(
    hv_raw: pd.DataFrame, model: str = HIGH_VOLUME_MODEL_CHOICE
) -> pd.DataFrame:
    """Filter the high-volume final-forecast frame down to the chosen model.

    `hv_raw` already carries `ProductID, forecast_week, WeekStart,
    predicted_Sales_Qty, model` (see `src/models/high_volume.py`); only the
    model filter and the `tier` tag are added here.
    """
    df = hv_raw.loc[hv_raw["model"] == model].copy()
    df["tier"] = "high_volume"
    return df[["ProductID", "forecast_week", "WeekStart", "predicted_Sales_Qty", "tier", "model"]]


def _add_forecast_week(df: pd.DataFrame, id_col: str, date_col: str) -> pd.DataFrame:
    """Derive `forecast_week` (1..N) per `id_col`, ordered by `date_col`.

    The sparse final-forecast artifact has no `forecast_week` column (only
    `unique_id, ds, model, y_pred`) -- it is reconstructed here from the
    per-product chronological order of `ds`, matching the horizon convention
    (`forecast_week=1` is the earliest forecast week) used by the
    high-volume tier.
    """
    df = df.sort_values([id_col, date_col]).copy()
    df["forecast_week"] = df.groupby(id_col).cumcount() + 1
    return df


def select_sparse_forecast(
    sparse_raw: pd.DataFrame, model: str = SPARSE_MODEL_CHOICE
) -> pd.DataFrame:
    """Filter the sparse final-forecast frame to the chosen model and
    reshape it onto the shared output schema (`ProductID`/`WeekStart`/
    `predicted_Sales_Qty` instead of `unique_id`/`ds`/`y_pred`).
    """
    df = sparse_raw.loc[sparse_raw["model"] == model].copy()
    df = _add_forecast_week(df, id_col="unique_id", date_col="ds")
    df = df.rename(
        columns={"unique_id": "ProductID", "ds": "WeekStart", "y_pred": "predicted_Sales_Qty"}
    )
    df["tier"] = "sparse"
    return df[["ProductID", "forecast_week", "WeekStart", "predicted_Sales_Qty", "tier", "model"]]


def validate_forecast(forecast: pd.DataFrame, tiers: pd.DataFrame) -> None:
    """Hard assertions per spec FR-26/FR-27/FR-28 -- raised before any write.

    - row count == `n_products * FORECAST_HORIZON` (FR-26);
    - every in-scope ProductID appears exactly `FORECAST_HORIZON` times (FR-28);
    - no negative predictions (FR-27).
    """
    in_scope_ids = set(tiers["ProductID"])
    n_products = len(in_scope_ids)
    expected_rows = n_products * FORECAST_HORIZON

    assert len(forecast) == expected_rows, (
        f"forecast row count mismatch: expected {expected_rows} "
        f"({n_products} products x {FORECAST_HORIZON} weeks), got {len(forecast)}"
    )

    counts = forecast.groupby("ProductID").size()
    missing = in_scope_ids - set(counts.index)
    assert not missing, f"{len(missing)} in-scope ProductID(s) missing from forecast, e.g. {sorted(missing)[:5]}"

    wrong_count = counts[counts != FORECAST_HORIZON]
    assert wrong_count.empty, (
        f"{len(wrong_count)} ProductID(s) do not have exactly {FORECAST_HORIZON} rows, "
        f"e.g. {wrong_count.head().to_dict()}"
    )

    assert (forecast["predicted_Sales_Qty"] >= 0).all(), "negative prediction found in assembled forecast"


def assemble_forecast(
    hv_raw: pd.DataFrame, sparse_raw: pd.DataFrame, tiers: pd.DataFrame
) -> pd.DataFrame:
    """Combine the chosen high-volume and sparse forecasts into the final
    submission schema, validating hard invariants before returning.
    """
    hv = select_high_volume_forecast(hv_raw)
    sparse = select_sparse_forecast(sparse_raw)
    combined = pd.concat([hv, sparse], ignore_index=True)
    combined = combined[OUTPUT_COLUMNS].sort_values(["tier", "ProductID", "forecast_week"]).reset_index(drop=True)
    validate_forecast(combined, tiers)
    return combined


def main() -> None:
    with stage(logger, "Load tiers + tier final forecasts"):
        tiers = pd.read_parquet(TIERS_PATH)
        hv_raw = pd.read_parquet(HIGH_VOLUME_FORECAST_PATH)
        sparse_raw = pd.read_parquet(SPARSE_FORECAST_PATH)
        log_metrics(
            logger,
            {
                "in_scope_products": tiers["ProductID"].nunique(),
                "high_volume_raw_rows": len(hv_raw),
                "sparse_raw_rows": len(sparse_raw),
            },
        )

    with stage(logger, f"Assemble forecast (high_volume={HIGH_VOLUME_MODEL_CHOICE}, sparse={SPARSE_MODEL_CHOICE})"):
        forecast = assemble_forecast(hv_raw, sparse_raw, tiers)
        log_metrics(
            logger,
            {
                "rows": len(forecast),
                "unique_products": forecast["ProductID"].nunique(),
                "expected_rows": tiers["ProductID"].nunique() * FORECAST_HORIZON,
            },
        )

    with stage(logger, "Write forecast_12wk.csv"):
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(OUTPUT_PATH, index=False)
        log_metrics(logger, {"path": str(OUTPUT_PATH)})


if __name__ == "__main__":
    main()
