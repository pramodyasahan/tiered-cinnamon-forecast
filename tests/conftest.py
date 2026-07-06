"""Synthetic fixtures mimicking the weekly.parquet schema (spec 002 test plan)."""
from __future__ import annotations

import pandas as pd
import pytest

WEEK0 = pd.Timestamp("2022-02-28")  # a Monday; matches the real global min week


def _weeks(n: int, start: pd.Timestamp = WEEK0) -> list[pd.Timestamp]:
    return [start + pd.Timedelta(weeks=i) for i in range(n)]


def make_weekly(rows: list[dict]) -> pd.DataFrame:
    """Build a weekly-schema frame from partial rows, filling required columns."""
    df = pd.DataFrame(rows)
    df["WeekStart"] = pd.to_datetime(df["WeekStart"])
    df["Sales_USD"] = df.get("Sales_USD", df["Sales_Qty"])
    df["returns_exceeded_sales"] = df.get("returns_exceeded_sales", False)
    df["transaction_count"] = df.get("transaction_count", (df["Sales_Qty"] > 0).astype(int))
    for col, default in [
        ("Region", "Europe"),
        ("Country", "Germany"),
        ("Sales Channel", "Retail"),
        ("Brand Category", "Retail"),
        ("Product Range", "PREMIUM GRADE"),
    ]:
        if col not in df:
            df[col] = default
    return df


@pytest.fixture
def single_product_pattern() -> pd.DataFrame:
    """One product, contiguous weekly, Sales_Qty = [5,0,0,3,0] (T-1)."""
    weeks = _weeks(5)
    qty = [5, 0, 0, 3, 0]
    return make_weekly(
        [{"ProductID": "AAAAA-001", "WeekStart": w, "Sales_Qty": float(q)} for w, q in zip(weeks, qty)]
    )


@pytest.fixture
def all_zero_and_all_nonzero() -> pd.DataFrame:
    """Two products spanning the same 4 weeks: all-zero and all-nonzero (T-2).

    Both cover weeks 0..3 so the global max week adds no trailing-zero row to the
    all-nonzero product (spec 002 A-2 retains trailing zeros).
    """
    weeks = _weeks(4)
    rows = [{"ProductID": "ZERO0-001", "WeekStart": w, "Sales_Qty": 0.0} for w in weeks]
    rows += [
        {"ProductID": "NONZ0-001", "WeekStart": w, "Sales_Qty": q}
        for w, q in zip(weeks, [2.0, 4.0, 1.0, 5.0])
    ]
    return make_weekly(rows)


@pytest.fixture
def two_products_diff_launch() -> pd.DataFrame:
    """Product LATE first sells at week index 2; product EARLY at week 0 (T-3/T-4).

    Global max week is index 4, so the panel should span:
      EARLY: weeks 0..4 (5 rows)
      LATE : weeks 2..4 (3 rows)  -- no rows at weeks 0,1
    """
    weeks = _weeks(5)
    rows = [
        {"ProductID": "EARLY-001", "WeekStart": weeks[0], "Sales_Qty": 3.0},
        {"ProductID": "EARLY-001", "WeekStart": weeks[4], "Sales_Qty": 7.0},
        {"ProductID": "LATE0-001", "WeekStart": weeks[2], "Sales_Qty": 9.0},
    ]
    return make_weekly(rows)
