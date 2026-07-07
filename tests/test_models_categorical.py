"""Regression test for spec 003 T004 — shared categorical-dtype helper (T-8).

T-8: XGBoost/LightGBM category consistency — a fixture with a category
present only in a validation fold must not silently mis-encode; and a fold
that happens to omit some categories must not shrink the shared category set.
"""
from __future__ import annotations

import pandas as pd

from src.models.categorical import apply_categorical_dtypes, fit_categorical_dtypes


def test_fold_subset_keeps_full_category_set():
    """A fold containing only value A must still carry categories A, B, C."""
    full = pd.DataFrame({"Region": ["A", "B", "C", "A", "B"]})
    dtypes = fit_categorical_dtypes(full, ["Region"])

    fold_a_only = pd.DataFrame({"Region": ["A", "A"]})
    result = apply_categorical_dtypes(fold_a_only, dtypes)

    assert list(result["Region"].cat.categories) == ["A", "B", "C"]
    assert result["Region"].tolist() == ["A", "A"]


def test_unseen_category_becomes_nan_not_exception():
    """A value never seen in the full-universe fit becomes NaN, not an error."""
    full = pd.DataFrame({"Region": ["A", "B", "C", "A", "B"]})
    dtypes = fit_categorical_dtypes(full, ["Region"])

    unseen = pd.DataFrame({"Region": ["A", "D"]})
    result = apply_categorical_dtypes(unseen, dtypes)

    assert result["Region"].isna().tolist() == [False, True]
    assert list(result["Region"].cat.categories) == ["A", "B", "C"]


def test_train_and_validation_share_identical_dtype():
    """Train and validation subsets of the same fold must have identical category dtype."""
    full = pd.DataFrame({"ProductID": ["P1", "P2", "P3", "P1", "P2", "P3"]})
    dtypes = fit_categorical_dtypes(full, ["ProductID"])

    train = apply_categorical_dtypes(full.iloc[:4], dtypes)
    valid = apply_categorical_dtypes(full.iloc[4:], dtypes)

    assert train["ProductID"].dtype == valid["ProductID"].dtype
