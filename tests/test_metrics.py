"""Regression tests for spec 003 T-2/T-3 — shared metric-function correctness.

Written before `src/metrics.py` exists (TDD RED state). Hand-computed expected
values; no mocking, no data files.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import mase, mae, rmse, wmape


# --- T-2: RMSE/MAE on a known small array -----------------------------------

def test_rmse_known_array():
    """y_true=[1,2,3,4], y_pred=[1,2,4,4] -> errors [0,0,-1,0], squared [0,0,1,0].

    mean squared error = 1/4 = 0.25 -> rmse = sqrt(0.25) = 0.5
    """
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 4, 4]
    assert rmse(y_true, y_pred) == pytest.approx(0.5)


def test_mae_known_array():
    """Same array: abs errors [0,0,1,0] -> mean = 0.25."""
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 4, 4]
    assert mae(y_true, y_pred) == pytest.approx(0.25)


def test_rmse_mae_differ_on_outlier():
    """A single large error should move RMSE more than MAE (sanity check)."""
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 10]
    # MAE = 10/4 = 2.5; RMSE = sqrt(100/4) = 5.0
    assert mae(y_true, y_pred) == pytest.approx(2.5)
    assert rmse(y_true, y_pred) == pytest.approx(5.0)
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


# --- T-3: WMAPE/MASE with zero-demand weeks ---------------------------------

def test_wmape_zero_demand_weeks_no_crash():
    """y_true has zero-demand weeks but is not all-zero -> must not raise.

    y_true=[0,0,5,0,3], y_pred=[0,1,4,0,3]
    abs errors = [0,1,1,0,0] -> sum = 2
    sum(abs(y_true)) = 8
    wmape = 2/8 = 0.25
    """
    y_true = [0, 0, 5, 0, 3]
    y_pred = [0, 1, 4, 0, 3]
    assert wmape(y_true, y_pred) == pytest.approx(0.25)


def test_wmape_all_zero_actuals_returns_zero_not_raises():
    """All-zero y_true (fully zero-demand window) -> denominator is 0.

    Documented choice: return 0.0 instead of raising ZeroDivisionError or NaN/inf.
    """
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    result = wmape(y_true, y_pred)
    assert result == 0.0
    assert not math.isnan(result)
    assert not math.isinf(result)


def test_wmape_all_zero_actuals_with_nonzero_preds():
    """All-zero y_true but nonzero predictions: still must not raise.

    Documented behavior: with a zero denominator we return 0.0 regardless of the
    numerator (a defined, non-crashing sentinel), rather than inf.
    """
    y_true = [0, 0, 0]
    y_pred = [1, 2, 3]
    result = wmape(y_true, y_pred)
    assert not math.isnan(result)
    assert not math.isinf(result)


def test_mase_zero_demand_weeks_no_crash():
    """MASE on a series with zero-demand weeks, non-constant train history.

    y_true=[0,0,5,0,3], y_pred=[0,1,4,0,3] -> MAE(num) = (0+1+1+0+0)/5 = 0.4
    y_train_history = [1,0,2,0,3,0,4] (season_length=1)
    naive errors = |0-1|,|2-0|,|0-2|,|3-0|,|0-3|,|4-0| = [1,2,2,3,3,4] -> mean = 2.5
    mase = 0.4 / 2.5 = 0.16
    """
    y_true = [0, 0, 5, 0, 3]
    y_pred = [0, 1, 4, 0, 3]
    y_train_history = np.array([1, 0, 2, 0, 3, 0, 4])
    result = mase(y_true, y_pred, y_train_history, season_length=1)
    assert result == pytest.approx(0.16)
    assert not math.isnan(result)
    assert not math.isinf(result)


def test_mase_constant_train_history_zero_denominator_matching_zero_numerator():
    """Constant train history -> naive-error denominator is 0.

    If y_true == y_pred exactly (numerator also 0), documented choice: return 0.0
    (a perfect forecast against an undefined scale is treated as zero error).
    """
    y_true = [5, 5, 5]
    y_pred = [5, 5, 5]
    y_train_history = np.array([2, 2, 2, 2])  # constant -> naive error is always 0
    result = mase(y_true, y_pred, y_train_history, season_length=1)
    assert result == 0.0
    assert not math.isnan(result)
    assert not math.isinf(result)


def test_mase_constant_train_history_zero_denominator_nonzero_numerator():
    """Constant train history but forecast has nonzero error -> must not raise.

    Documented choice: return the numerator itself (MAE) as a graceful fallback
    scale of 1, rather than raising ZeroDivisionError or returning NaN/inf.
    """
    y_true = [5, 6, 5]
    y_pred = [5, 5, 5]
    y_train_history = np.array([2, 2, 2, 2])  # constant -> naive error is always 0
    result = mase(y_true, y_pred, y_train_history, season_length=1)
    # numerator MAE = (0+1+0)/3 = 1/3
    assert result == pytest.approx(1 / 3)
    assert not math.isnan(result)
    assert not math.isinf(result)
