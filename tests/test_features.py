"""Regression tests for spec 002 — feature-engineering correctness (B1 + B2).

T-1..T-6 run on synthetic fixtures (no data files).
T-7..T-8 assert against the regenerated real parquet and skip when absent.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features import (
    _add_lag_and_rolling_features,
    _complete_product_week_panel,
    build_feature_set,
)

WEEKLY_PARQUET = Path("data/processed/weekly.parquet")
FEATURED_PARQUET = Path("data/processed/featured.parquet")
TIERS_PARQUET = Path("data/processed/tiers.parquet")


def _featured(weekly: pd.DataFrame) -> pd.DataFrame:
    """Run the feature builder in-memory (no scoped day-of-week merge, no write)."""
    panel = _complete_product_week_panel(weekly)
    from src.features import _add_calendar_and_sparse_features

    featured = _add_lag_and_rolling_features(panel)
    featured = _add_calendar_and_sparse_features(featured)
    return featured.sort_values(["ProductID", "WeekStart"]).reset_index(drop=True)


# --- B1: zero_streak_length ------------------------------------------------

def test_zero_streak_pattern(single_product_pattern):
    """T-1: Sales_Qty [5,0,0,3,0] -> streak [0,1,2,0,1], cumulative [0,1,2,2,3]."""
    f = _featured(single_product_pattern)
    assert f["zero_streak_length"].tolist() == [0, 1, 2, 0, 1]
    assert f["cumulative_zero_weeks"].tolist() == [0, 1, 2, 2, 3]


def test_zero_streak_edges(all_zero_and_all_nonzero):
    """T-2: all-zero -> increasing streak; all-nonzero -> all zeros."""
    f = _featured(all_zero_and_all_nonzero)
    zero = f[f.ProductID == "ZERO0-001"].sort_values("WeekStart")
    nonz = f[f.ProductID == "NONZ0-001"].sort_values("WeekStart")
    assert zero["zero_streak_length"].tolist() == [1, 2, 3, 4]
    assert nonz["zero_streak_length"].tolist() == [0, 0, 0, 0]


# --- B2: panel anchor ------------------------------------------------------

def test_panel_anchor_no_prelaunch(two_products_diff_launch):
    """T-3: no product has rows before its own first observed week; both end at global max."""
    f = _featured(two_products_diff_launch)
    global_max = two_products_diff_launch["WeekStart"].max()
    for pid, first in two_products_diff_launch.groupby("ProductID")["WeekStart"].min().items():
        g = f[f.ProductID == pid]
        assert g["WeekStart"].min() == first, f"{pid} has pre-launch rows"
        assert g["WeekStart"].max() == global_max, f"{pid} does not extend to global max"
    # LATE product must have exactly 3 rows (weeks 2..4), not 5
    assert (f.ProductID == "LATE0-001").sum() == 3


def test_panel_gap_free(two_products_diff_launch):
    """T-4: each product's WeekStart sequence is contiguous weekly with no gaps."""
    f = _featured(two_products_diff_launch)
    for pid, g in f.groupby("ProductID"):
        weeks = g["WeekStart"].sort_values().reset_index(drop=True)
        deltas = weeks.diff().dropna()
        assert (deltas == pd.Timedelta(weeks=1)).all(), f"{pid} has a gap in its weekly grid"


# --- Regression guards -----------------------------------------------------

def test_rolling_no_current_row_leakage(single_product_pattern):
    """T-5: mutating row k's Sales_Qty must not change rolling features at row k."""
    base = _add_lag_and_rolling_features(_complete_product_week_panel(single_product_pattern))
    mutated_in = single_product_pattern.copy()
    k = 3
    mutated_in.loc[k, "Sales_Qty"] += 1_000_000
    mutated = _add_lag_and_rolling_features(_complete_product_week_panel(mutated_in))
    cols = ["Sales_Qty_rollmean_4", "Sales_Qty_rollstd_4", "Sales_Qty_rollmean_12", "Sales_Qty_rollstd_12"]
    before = base.iloc[k][cols].fillna(-1)
    after = mutated.iloc[k][cols].fillna(-1)
    assert before.equals(after), "rolling feature leaked the current row's Sales_Qty"


def test_lag1_equals_previous_week(single_product_pattern):
    """T-6: Sales_Qty_lag_1[k] == Sales_Qty[k-1] within a product."""
    f = _featured(single_product_pattern)
    qty = f["Sales_Qty"].tolist()
    lag1 = f["Sales_Qty_lag_1"].tolist()
    assert pd.isna(lag1[0])
    for i in range(1, len(qty)):
        assert lag1[i] == qty[i - 1]


# --- Scale checks against the regenerated real parquet ---------------------

@pytest.mark.skipif(
    not (FEATURED_PARQUET.exists() and WEEKLY_PARQUET.exists()),
    reason="regenerated featured/weekly parquet not present (run `make features`)",
)
def test_real_parquet_no_prelaunch():
    """T-7: no featured row precedes its product's first observed week; sparsity reduced."""
    weekly = pd.read_parquet(WEEKLY_PARQUET, columns=["ProductID", "WeekStart"])
    feat = pd.read_parquet(FEATURED_PARQUET, columns=["ProductID", "WeekStart", "Sales_Qty"])
    weekly["WeekStart"] = pd.to_datetime(weekly["WeekStart"])
    feat["WeekStart"] = pd.to_datetime(feat["WeekStart"])
    first_real = weekly.groupby("ProductID")["WeekStart"].min()
    first_feat = feat.groupby("ProductID")["WeekStart"].min()
    aligned = first_feat.reindex(first_real.index)
    assert (aligned == first_real).all(), "some product has fabricated pre-launch featured rows"
    assert (feat["Sales_Qty"] == 0).mean() < 0.972
    assert len(feat) < 1_683_672


@pytest.mark.skipif(
    not (FEATURED_PARQUET.exists() and TIERS_PARQUET.exists()),
    reason="regenerated featured/tiers parquet not present (run `make features` then `make tiering`)",
)
def test_tiering_partition_after_refit():
    """T-8: tiering remains a strict partition over all in-scope products."""
    from src.tiering import tier_products

    tiers = tier_products(featured_path=FEATURED_PARQUET, output_path=None)
    hv = set(tiers.loc[tiers.tier == "high_volume", "ProductID"])
    sp = set(tiers.loc[tiers.tier == "sparse", "ProductID"])
    allp = set(tiers["ProductID"])
    assert not (hv & sp)
    assert (hv | sp) == allp
    assert tiers["ProductID"].is_unique
