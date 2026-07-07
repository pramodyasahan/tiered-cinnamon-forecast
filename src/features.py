from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.logging_utils import get_logger, log_metrics, stage

LOGGER = get_logger(__name__)

WEEKLY_PATH = Path("data/processed/weekly.parquet")
SCOPED_PATH = Path("data/processed/scoped.parquet")
FEATURED_PATH = Path("data/processed/featured.parquet")
LAG_WEEKS = [1, 2, 4, 12]
ROLLING_WINDOWS = [4, 12]
CATEGORICAL_COLUMNS = [
    "Region",
    "Country",
    "Sales Channel",
    "Brand Category",
    "Product Range",
]


def _complete_product_week_panel(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["WeekStart"] = pd.to_datetime(weekly["WeekStart"])
    # Anchor each product's grid at its own first observed week (not the global
    # minimum) so pre-launch weeks are never fabricated; extend every product to
    # the global max week so all series share a common forecast origin. Trailing
    # zero weeks (product exists, no sales) are retained as real demand signal.
    global_max = weekly["WeekStart"].max()
    global_min = weekly["WeekStart"].min()
    all_weeks = pd.date_range(global_min, global_max, freq="W-MON", name="WeekStart")
    product_first = weekly.groupby("ProductID")["WeekStart"].min()

    products = pd.Index(sorted(weekly["ProductID"].unique()), name="ProductID")
    grid = pd.MultiIndex.from_product([products, all_weeks], names=["ProductID", "WeekStart"]).to_frame(index=False)
    first_by_row = grid["ProductID"].map(product_first)
    grid = grid.loc[grid["WeekStart"] >= first_by_row]
    panel_index = pd.MultiIndex.from_frame(grid[["ProductID", "WeekStart"]])

    panel = weekly.set_index(["ProductID", "WeekStart"]).reindex(panel_index).reset_index()
    panel["Sales_Qty"] = panel["Sales_Qty"].fillna(0.0)
    panel["Sales_USD"] = panel["Sales_USD"].fillna(0.0)
    panel["returns_exceeded_sales"] = panel["returns_exceeded_sales"].fillna(False).astype(bool)
    panel["transaction_count"] = panel["transaction_count"].fillna(0).astype(int)

    product_modes = weekly.groupby("ProductID")[CATEGORICAL_COLUMNS].agg(_mode_or_na)
    panel = panel.drop(columns=CATEGORICAL_COLUMNS).merge(
        product_modes.reset_index(), on="ProductID", how="left", validate="many_to_one"
    )
    return panel


def _mode_or_na(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.value_counts(sort=True).index[0]


def _weekly_day_of_week_features(scoped: pd.DataFrame) -> pd.DataFrame:
    tx = scoped[["ProductID", "Order Date"]].copy()
    tx["Order Date"] = pd.to_datetime(tx["Order Date"], errors="coerce")
    tx["WeekStart"] = tx["Order Date"] - pd.to_timedelta(tx["Order Date"].dt.weekday, unit="D")
    tx["WeekStart"] = tx["WeekStart"].dt.normalize()
    tx["order_dow"] = tx["Order Date"].dt.dayofweek
    tx["is_weekend_order"] = tx["order_dow"].isin([5, 6])

    grouped = tx.groupby(["ProductID", "WeekStart"], sort=True)
    dow = grouped.agg(
        dominant_dow=("order_dow", _mode_or_na),
        pct_weekend_orders=("is_weekend_order", "mean"),
    ).reset_index()
    return dow


def _add_lag_and_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    featured = panel.sort_values(["ProductID", "WeekStart"]).copy()
    grouped_sales = featured.groupby("ProductID", sort=False)["Sales_Qty"]

    for lag in LAG_WEEKS:
        featured[f"Sales_Qty_lag_{lag}"] = grouped_sales.shift(lag)

    shifted = grouped_sales.shift(1)
    for window in ROLLING_WINDOWS:
        featured[f"Sales_Qty_rollmean_{window}"] = (
            shifted.groupby(featured["ProductID"], sort=False).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        featured[f"Sales_Qty_rollstd_{window}"] = (
            shifted.groupby(featured["ProductID"], sort=False).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        )

    return featured


def _zero_streak(values: pd.Series) -> pd.Series:
    streaks: list[int] = []
    current = 0
    for value in values:
        if value == 0:
            current += 1
        else:
            current = 0
        streaks.append(current)
    return pd.Series(streaks, index=values.index)


def _add_calendar_and_sparse_features(featured: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(featured["WeekStart"])
    featured["month"] = dates.dt.month
    featured["week_of_month"] = ((dates.dt.day - 1) // 7) + 1
    featured["quarter"] = dates.dt.quarter
    featured["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    featured["is_year_end"] = dates.dt.month.isin([11, 12])

    years = range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1)
    lk_holidays = set(pd.to_datetime(list(holidays.country_holidays("LK", years=years).keys())))
    week_holiday = []
    for week_start in dates:
        week_days = pd.date_range(week_start, periods=7, freq="D")
        week_holiday.append(any(day in lk_holidays for day in week_days))
    featured["is_lk_holiday_week"] = week_holiday

    zero_mask = featured["Sales_Qty"].eq(0)
    # _zero_streak counts runs where value == 0, so it must see the numeric
    # Sales_Qty, not the boolean mask (a bool would invert the polarity).
    featured["zero_streak_length"] = (
        featured.groupby("ProductID", sort=False)["Sales_Qty"].transform(_zero_streak).astype(int)
    )
    featured["cumulative_zero_weeks"] = zero_mask.groupby(featured["ProductID"], sort=False).cumsum().astype(int)
    featured["unit_price"] = np.where(featured["Sales_Qty"] > 0, featured["Sales_USD"] / featured["Sales_Qty"], np.nan)

    return featured


def _assert_rolling_features_do_not_use_current_row(panel: pd.DataFrame) -> None:
    candidate_product = panel.groupby("ProductID").size().loc[lambda s: s >= 16].index[0]
    product_panel = panel.loc[panel["ProductID"] == candidate_product].copy()
    baseline = _add_lag_and_rolling_features(product_panel)

    row_pos = 12
    mutated_panel = product_panel.copy()
    mutated_panel.iloc[row_pos, mutated_panel.columns.get_loc("Sales_Qty")] += 999_999
    mutated = _add_lag_and_rolling_features(mutated_panel)

    rolling_cols = [f"Sales_Qty_rollmean_{w}" for w in ROLLING_WINDOWS] + [
        f"Sales_Qty_rollstd_{w}" for w in ROLLING_WINDOWS
    ]
    before = baseline.iloc[row_pos][rolling_cols]
    after = mutated.iloc[row_pos][rolling_cols]
    if not before.fillna(-1).equals(after.fillna(-1)):
        raise AssertionError("Rolling features changed when the current row's Sales_Qty was mutated")


def build_feature_set(
    weekly_path: Path = WEEKLY_PATH,
    scoped_path: Path = SCOPED_PATH,
    output_path: Path | None = FEATURED_PATH,
) -> pd.DataFrame:
    if not weekly_path.exists():
        raise FileNotFoundError(f"Weekly parquet not found: {weekly_path}")
    if not scoped_path.exists():
        raise FileNotFoundError(f"Scoped parquet not found: {scoped_path}")

    weekly = pd.read_parquet(weekly_path)
    scoped = pd.read_parquet(scoped_path)
    panel = _complete_product_week_panel(weekly)
    _assert_rolling_features_do_not_use_current_row(panel)

    featured = _add_lag_and_rolling_features(panel)
    dow_features = _weekly_day_of_week_features(scoped)
    featured = featured.merge(dow_features, on=["ProductID", "WeekStart"], how="left", validate="one_to_one")
    featured["dominant_dow"] = featured["dominant_dow"].astype("Float64")
    featured["pct_weekend_orders"] = featured["pct_weekend_orders"].fillna(0.0)
    featured = _add_calendar_and_sparse_features(featured)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        featured.to_parquet(output_path, index=False)

    featured.attrs["weekly_rows"] = len(weekly)
    featured.attrs["featured_rows"] = len(featured)
    featured.attrs["products"] = int(featured["ProductID"].nunique())
    featured.attrs["weeks"] = int(featured["WeekStart"].nunique())
    featured.attrs["zero_sales_share"] = float(featured["Sales_Qty"].eq(0).mean())
    return featured


def main() -> None:
    with stage(LOGGER, "Stage 5: build_feature_set()"):
        featured = build_feature_set()
        log_metrics(
            LOGGER,
            {
                "weekly_rows": featured.attrs["weekly_rows"],
                "featured_rows": featured.attrs["featured_rows"],
                "products": featured.attrs["products"],
                "weeks": featured.attrs["weeks"],
                "zero_sales_share": featured.attrs["zero_sales_share"],
                "output": FEATURED_PATH,
            },
        )


if __name__ == "__main__":
    main()
