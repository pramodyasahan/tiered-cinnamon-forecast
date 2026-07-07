"""Tests for spec 003 T006 sparse-tier forecasting — T-4/T-5.

T-4: a single-observation `unique_id` must still produce exactly 12
forecast rows via the deterministic fallback path (spec EC-1, FR-13).

T-5: `assert_full_horizon` (the concrete guard against the plan §13
freq-mismatch failure mode) must actually raise when a forecast group is
short, proving the guard is not a no-op.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.models.sparse import (
    HORIZON,
    _naive_fallback_forecast,
    _run_statsforecast_with_fallback,
    assert_full_horizon,
    build_sparse_long_frame,
)


# --- T-4: single-observation product gets exactly 12 forecast rows ---------

def test_single_observation_product_gets_12_forecast_rows():
    """A synthetic long-format frame with one `unique_id` having only 1
    historical row must still receive exactly 12 rows per model from the
    fallback path (never dropped, per spec EC-1 / FR-13).
    """
    long_df = pd.DataFrame(
        {
            "unique_id": ["SOLO"],
            "ds": [pd.Timestamp("2025-09-01")],
            "y": [7.0],
        }
    )
    forecast_start = pd.Timestamp("2025-09-08")

    preds = _run_statsforecast_with_fallback(
        long_df, all_ids=["SOLO"], horizon=HORIZON, forecast_start=forecast_start
    )

    solo_rows = preds[preds["unique_id"] == "SOLO"]
    assert len(solo_rows) > 0
    for _, group in solo_rows.groupby("model"):
        assert len(group) == HORIZON

    # The fallback carries the single observed value forward.
    assert (solo_rows["y_pred"] == 7.0).all()


def test_naive_fallback_forecast_repeats_last_value():
    history = pd.DataFrame(
        {"unique_id": ["A", "A"], "ds": pd.to_datetime(["2025-01-06", "2025-01-13"]), "y": [3.0, 9.0]}
    )
    out = _naive_fallback_forecast(
        history, "A", pd.Timestamp("2025-01-20"), HORIZON, "y_pred"
    )
    assert len(out) == HORIZON
    assert (out["y_pred"] == 9.0).all()


def test_naive_fallback_forecast_empty_history_returns_zero():
    empty_history = pd.DataFrame(columns=["unique_id", "ds", "y"])
    out = _naive_fallback_forecast(
        empty_history, "B", pd.Timestamp("2025-01-20"), HORIZON, "y_pred"
    )
    assert len(out) == HORIZON
    assert (out["y_pred"] == 0.0).all()


# --- T-5: the length-assertion guard actually catches short forecasts ------

def test_assert_full_horizon_raises_on_short_forecast():
    """Directly unit-test the guard: a deliberately-short (8-row) forecast
    group for a single `unique_id` must trip the assertion (proves the
    guard catches the plan §13 freq-mismatch failure mode rather than
    being a no-op).
    """
    short_forecast = pd.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "ds": pd.date_range("2025-09-22", periods=8, freq="W-MON"),
            "y_pred": [1.0] * 8,
        }
    )
    with pytest.raises(AssertionError):
        assert_full_horizon(short_forecast, horizon=HORIZON)


def test_assert_full_horizon_passes_on_correct_length():
    correct_forecast = pd.DataFrame(
        {
            "unique_id": ["A"] * HORIZON,
            "ds": pd.date_range("2025-09-22", periods=HORIZON, freq="W-MON"),
            "y_pred": [1.0] * HORIZON,
        }
    )
    assert_full_horizon(correct_forecast, horizon=HORIZON)


def test_assert_full_horizon_raises_naming_offending_ids_across_multiple_ids():
    mixed = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": ["GOOD"] * HORIZON,
                    "ds": pd.date_range("2025-09-22", periods=HORIZON, freq="W-MON"),
                    "y_pred": [1.0] * HORIZON,
                }
            ),
            pd.DataFrame(
                {
                    "unique_id": ["BAD"] * 5,
                    "ds": pd.date_range("2025-09-22", periods=5, freq="W-MON"),
                    "y_pred": [1.0] * 5,
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(AssertionError, match="BAD"):
        assert_full_horizon(mixed, horizon=HORIZON)


# --- build_sparse_long_frame reshaping --------------------------------------

def test_build_sparse_long_frame_filters_and_renames():
    featured = pd.DataFrame(
        {
            "ProductID": ["P1", "P1", "P2"],
            "WeekStart": pd.to_datetime(["2025-01-06", "2025-01-13", "2025-01-06"]),
            "Sales_Qty": [1.0, 2.0, 3.0],
            "other_col": ["x", "y", "z"],
        }
    )
    long_df = build_sparse_long_frame(featured, sparse_ids=["P1"])
    assert list(long_df.columns) == ["unique_id", "ds", "y"]
    assert set(long_df["unique_id"]) == {"P1"}
    assert len(long_df) == 2
