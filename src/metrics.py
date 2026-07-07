"""Shared scoring functions reused by `src/backtest.py` (spec 003 FR-10/FR-15/FR-21).

Kept as small, independently-testable standalone functions rather than a single
monolithic scorer (review.md non-blocking suggestion, plan §4.5). All functions
accept array-likes (list/np.ndarray/pd.Series) and return a Python float.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def wmape(y_true, y_pred) -> float:
    """Weighted mean absolute percentage error: sum(|err|) / sum(|actual|).

    Sparse/intermittent demand windows can have `sum(|y_true|) == 0` (an
    all-zero-demand validation window). Rather than raising ZeroDivisionError
    or returning NaN/inf, we return 0.0 in that case -- a defined, non-crashing
    sentinel meaning "no demand to score against" (documented choice, spec FR-15).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mase(y_true, y_pred, y_train_history, season_length: int = 1) -> float:
    """Mean Absolute Scaled Error: MAE(y_true, y_pred) / MAE(seasonal-naive on train history).

    The scaling denominator is the mean absolute error of the seasonal-naive
    forecast computed on the training history: mean(|history[s:] - history[:-s]|)
    for season_length s. For a constant (or too-short) training history this
    denominator is 0. Rather than raising ZeroDivisionError or returning NaN/inf:
      - if the numerator (MAE of the actual forecast) is also 0, return 0.0
        (a perfect forecast against an undefined scale is treated as zero error);
      - otherwise return the numerator itself, i.e. fall back to an implicit
        scale of 1 (documented choice, spec FR-15) rather than crashing.
    """
    numerator = mae(y_true, y_pred)

    history = np.asarray(y_train_history, dtype=float)
    if len(history) <= season_length:
        denom = 0.0
    else:
        denom = float(np.mean(np.abs(history[season_length:] - history[:-season_length])))

    if denom == 0:
        return 0.0 if numerator == 0 else numerator
    return numerator / denom
