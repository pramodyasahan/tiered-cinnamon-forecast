"""Tests for spec 003 T010 forecast-CSV assembly — T-1.

T-1 (plan §12): on a small synthetic tiers/predictions fixture, the
assembled forecast must have row count == `n_products * 12`, every
`ProductID` must appear exactly 12 times, and all predictions must be
nonnegative (spec FR-24..FR-28, AC-4).

Written before `src/assemble_forecast.py` exists (TDD, T009) -- this file
must fail on collection until that module is created (T010).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.assemble_forecast import (
    FORECAST_HORIZON,
    OUTPUT_COLUMNS,
    assemble_forecast,
)


def _synthetic_tiers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ProductID": ["HV1", "HV2", "SP1", "SP2"],
            "transaction_count": [50, 40, 2, 1],
            "total_sales_qty": [500.0, 400.0, 3.0, 1.0],
            "tier": ["high_volume", "high_volume", "sparse", "sparse"],
        }
    )


def _synthetic_hv_raw() -> pd.DataFrame:
    """Mimics `outputs/models/high_volume/final_forecast.parquet`: multiple
    candidate models (lightgbm/xgboost/seasonal_naive), one row per
    (ProductID, forecast_week, model).
    """
    rows = []
    for pid in ["HV1", "HV2"]:
        for model in ["lightgbm", "xgboost", "seasonal_naive"]:
            for week in range(1, FORECAST_HORIZON + 1):
                rows.append(
                    {
                        "ProductID": pid,
                        "forecast_week": week,
                        "WeekStart": pd.Timestamp("2025-09-22") + pd.Timedelta(weeks=week - 1),
                        "predicted_Sales_Qty": 10.0 + week,
                        "model": model,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_sparse_raw() -> pd.DataFrame:
    """Mimics `outputs/models/sparse/final_forecast.parquet`: columns
    `unique_id, ds, model, y_pred` (no `forecast_week` column -- must be
    derived), multiple candidate models per product.
    """
    rows = []
    for uid in ["SP1", "SP2"]:
        for model in ["CrostonClassic", "CrostonSBA", "TSB", "Naive"]:
            for week in range(1, FORECAST_HORIZON + 1):
                rows.append(
                    {
                        "unique_id": uid,
                        "ds": pd.Timestamp("2025-09-22") + pd.Timedelta(weeks=week - 1),
                        "model": model,
                        "y_pred": 1.0 + week,
                    }
                )
    return pd.DataFrame(rows)


def test_assembled_forecast_row_count_and_coverage():
    tiers = _synthetic_tiers()
    hv_raw = _synthetic_hv_raw()
    sparse_raw = _synthetic_sparse_raw()

    result = assemble_forecast(hv_raw, sparse_raw, tiers)

    n_products = tiers["ProductID"].nunique()
    assert len(result) == n_products * FORECAST_HORIZON

    counts = result.groupby("ProductID").size()
    assert (counts == FORECAST_HORIZON).all()
    assert set(counts.index) == set(tiers["ProductID"])


def test_assembled_forecast_predictions_nonnegative():
    tiers = _synthetic_tiers()
    hv_raw = _synthetic_hv_raw()
    sparse_raw = _synthetic_sparse_raw()

    result = assemble_forecast(hv_raw, sparse_raw, tiers)

    assert (result["predicted_Sales_Qty"] >= 0).all()


def test_assembled_forecast_has_expected_schema():
    tiers = _synthetic_tiers()
    hv_raw = _synthetic_hv_raw()
    sparse_raw = _synthetic_sparse_raw()

    result = assemble_forecast(hv_raw, sparse_raw, tiers)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert set(result["tier"]) == {"high_volume", "sparse"}


def test_assemble_forecast_raises_on_missing_product():
    """If tiers.parquet has a product with no matching forecast rows, the
    hard row-count/coverage assertion must fail loudly, not silently drop it.
    """
    tiers = _synthetic_tiers()
    hv_raw = _synthetic_hv_raw()
    # Drop all forecast rows for SP2 to simulate a missing product.
    sparse_raw = _synthetic_sparse_raw()
    sparse_raw = sparse_raw[sparse_raw["unique_id"] != "SP2"]

    with pytest.raises(AssertionError):
        assemble_forecast(hv_raw, sparse_raw, tiers)


def test_assemble_forecast_raises_on_negative_prediction():
    tiers = _synthetic_tiers()
    hv_raw = _synthetic_hv_raw()
    sparse_raw = _synthetic_sparse_raw()
    sparse_raw = sparse_raw.copy()
    tsb_index = sparse_raw.index[sparse_raw["model"] == "TSB"][0]
    sparse_raw.loc[tsb_index, "y_pred"] = -5.0

    with pytest.raises(AssertionError):
        assemble_forecast(hv_raw, sparse_raw, tiers)
