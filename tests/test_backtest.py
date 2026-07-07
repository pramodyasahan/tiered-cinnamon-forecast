"""Tests for spec 003 T008 backtest/metrics table — T-6/T-9.

T-6: static scan proving no `shuffle=True` / `train_test_split` call exists
in `src/backtest.py` or any `src/models/*.py` module (no random split for
time-series validation, constitution "Disallowed Patterns").

T-9: the written model comparison table has at least one `is_baseline=True`
row per tier (review.md non-blocking suggestion, spec AC-6).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.backtest import score_high_volume, score_sparse

SCANNED_FILES = [Path("src/backtest.py"), *Path("src/models").glob("*.py")]


def test_no_random_split_static_scan():
    """Flags real calls only; skips backtick-quoted docstring mentions of the
    forbidden pattern (several model modules document the absence of
    `shuffle=True`/`train_test_split` by name)."""
    for path in SCANNED_FILES:
        code_lines = [line for line in path.read_text().splitlines() if "`" not in line]
        code = "\n".join(code_lines)
        assert "shuffle=True" not in code, f"{path} contains shuffle=True"
        assert "train_test_split" not in code, f"{path} contains train_test_split"


def test_high_volume_table_has_baseline_row():
    valid = pd.DataFrame(
        {
            "ProductID": ["p1", "p1", "p2", "p2"],
            "model": ["lightgbm", "seasonal_naive", "lightgbm", "seasonal_naive"],
            "y_true": [10.0, 10.0, 5.0, 5.0],
            "y_pred": [9.0, 8.0, 4.0, 3.0],
        }
    )
    timing = {
        "lightgbm_train_seconds": 1.0,
        "lightgbm_inference_seconds": 0.1,
        "seasonal_naive_inference_seconds": 0.01,
    }
    table = score_high_volume(valid, timing)
    assert table["is_baseline"].any()
    baseline_row = table.loc[table["model"] == "seasonal_naive"].iloc[0]
    assert baseline_row["is_baseline"] is True or baseline_row["is_baseline"] == True  # noqa: E712
    assert baseline_row["train_seconds"] == 0.0


def test_sparse_table_has_baseline_row_and_excludes_nan_actuals():
    valid = pd.DataFrame(
        {
            "unique_id": ["p1", "p1", "p1", "p2", "p2"],
            "model": ["Naive", "Naive", "Naive", "CrostonClassic", "CrostonClassic"],
            "y_pred": [1.0, 1.0, 1.0, 2.0, 2.0],
            "y": [1.0, float("nan"), 0.0, 2.0, 3.0],
        }
    )
    timing = pd.DataFrame(
        {
            "stage": ["fold1_validation", "final_forecast"],
            "train_seconds": [1.5, 0.5],
            "inference_seconds": [1.5, 0.5],
            "n_products": [2, 2],
        }
    )
    history_by_id = {"p1": [1.0, 1.0], "p2": [2.0, 2.0]}
    table = score_sparse(valid, history_by_id, timing)
    assert table["is_baseline"].any()
    assert not table["wmape"].isna().any()
    assert not table["mase"].isna().any()
    for _, row in table.iterrows():
        assert row["train_seconds"] == 1.5
        assert row["inference_seconds"] == 1.5
