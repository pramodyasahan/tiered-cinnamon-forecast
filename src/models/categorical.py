"""Shared categorical-dtype helper for the high-volume models (LightGBM + XGBoost).

Both LightGBM's native categorical splits and XGBoost's
``tree_method="hist", enable_categorical=True`` path require every categorical
column to be a pandas ``category`` dtype, and — critically for XGBoost —
that dtype must be IDENTICAL (same category set, same signed/unsigned code
encoding) between the training and validation folds, or XGBoost raises an
error. A silent mismatch would also make LightGBM's categorical splits
inconsistent across folds (plan §2 Exa fact, §13 failure mode).

The fix used here: fit the ``category`` dtype **once**, across the FULL
high-volume product/categorical universe, *before* any rolling-origin train/
validation split. Every fold then reuses that same fitted dtype via
``apply_categorical_dtypes``, so train and validation always see an
identical category set and encoding — no per-fold refitting, ever.

Usage for T005 (``src/models/high_volume.py``):
    dtypes = fit_categorical_dtypes(full_df, CATEGORICAL_COLUMNS)  # once, pre-split
    train_df = apply_categorical_dtypes(train_fold, dtypes)
    valid_df = apply_categorical_dtypes(valid_fold, dtypes)
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def fit_categorical_dtypes(
    df: pd.DataFrame, columns: Iterable[str]
) -> dict[str, pd.CategoricalDtype]:
    """Fit a fixed ``pd.CategoricalDtype`` per column across the full universe.

    IMPORTANT: call this exactly once, on the complete high-volume dataset,
    *before* splitting into any rolling-origin train/validation fold. Do NOT
    call this per-fold — a fold-local fit can produce a category set (and
    signed/unsigned code width) that differs between train and validation,
    which XGBoost rejects outright and which silently changes LightGBM's
    categorical split boundaries (plan §13).

    Parameters
    ----------
    df:
        The full high-volume product/categorical universe (pre-split).
    columns:
        Column names to fit (e.g. ``ProductID``, ``Region``, ``Country``,
        ``"Sales Channel"``, ``"Brand Category"``, ``"Product Range"``).

    Returns
    -------
    dict mapping column name -> fitted ``pd.CategoricalDtype`` whose
    ``categories`` are the sorted unique non-null values observed in ``df``.
    """
    dtypes: dict[str, pd.CategoricalDtype] = {}
    for column in columns:
        categories = sorted(df[column].dropna().unique())
        dtypes[column] = pd.CategoricalDtype(categories=categories)
    return dtypes


def apply_categorical_dtypes(
    df: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype]
) -> pd.DataFrame:
    """Cast columns in ``df`` to the fitted dtypes from ``fit_categorical_dtypes``.

    Returns a copy. Any value not present in a column's fitted category set
    (an "unseen category" — e.g. a value that only appears in a validation
    fold and never in the full-universe fit, which should not happen if fit
    was run on the full universe, but is also the graceful path for any
    truly novel value) becomes ``NaN`` after the cast. It does not raise.

    Unseen values are masked to ``NaN`` with ``.where(...isin(categories))``
    before the ``.astype(dtype)`` call, rather than relying on
    ``.astype(CategoricalDtype)`` to silently drop out-of-category values
    itself: pandas has deprecated that implicit behavior (as of pandas 3.x)
    and will raise in a future version, so masking first keeps this
    helper's "unseen category -> NaN" contract stable across pandas
    versions.
    """
    out = df.copy()
    for column, dtype in dtypes.items():
        series = out[column]
        masked = series.where(series.isna() | series.isin(dtype.categories))
        out[column] = masked.astype(dtype)
    return out
