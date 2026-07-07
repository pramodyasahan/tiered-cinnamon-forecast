"""Product-country drill-down demonstration (spec 003 FR-16-FR-19).

Written rationale (FR-18): this demo is deliberately limited to 3 approved
in-scope products (`16576-029`, `13766-020`, `89574-017`), each of which
sells to exactly 2 distinct destination countries. The scope limitation is
grounded in a verified, re-computed fact: among ALL in-scope (cinnamon-only)
products, zero products have 3 or more distinct destination countries -- the
maximum observed is 2. Because no product in the catalog offers a richer
per-country split than 2 destinations, and only a small number of products
have enough per-country transaction history to support even a naive
per-country forecast, country-level forecasting is demonstrated here rather
than attempted catalog-wide (spec 001 FR-17-FR-19, spec 003 NG3).

Data source note: this module reads `data/processed/scoped.parquet`
(transaction-level) rather than `data/processed/weekly.parquet`, because the
weekly file aggregates `Country` down to a single per-product-week mode
value, which silently collapses the per-country split whenever a product
sold to two different countries in different weeks -- exactly the case this
demo needs to preserve. Weekly aggregation is instead performed here per
(ProductID, Country, WeekStart), mirroring the ISO-week (Monday-start) logic
in `src.cleaning.aggregate_weekly()`.

Forecast method: for each (ProductID, Country) pair, the 12-week-ahead
forecast is a **rolling mean of the last up-to-4 observed weekly values**
carried forward flat across the whole horizon. This is intentionally simple
and not intended to be a production-accuracy method -- per spec plan §4.4 /
EC-5, the demo's purpose is to prove country-level drill-down is
*mechanically feasible* (i.e., that a forecast pipeline can be keyed on
(ProductID, Country) instead of just ProductID), not to compete with the
tiered high-volume/sparse models used for the main product-level forecast.

This module does not touch `src/tiering.py`, `data/processed/tiers.parquet`,
`src/models/high_volume.py`, `src/models/sparse.py`, or
`outputs/forecasts/forecast_12wk.csv` (FR-19, NG3).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.logging_utils import get_logger, log_metrics, stage

LOGGER = get_logger(__name__)

SCOPED_PATH = Path("data/processed/scoped.parquet")
OUTPUT_PATH = Path("outputs/models/country_demo/forecast.csv")

DEMO_PRODUCT_IDS = ["16576-029", "13766-020", "89574-017"]

FORECAST_ORIGIN = pd.Timestamp("2025-09-22")  # week after the pipeline's shared last observed WeekStart (2025-09-15)
HORIZON_WEEKS = 12
ROLLING_WINDOW = 4
METHOD_NAME = f"rolling_mean_last_{ROLLING_WINDOW}_weeks"


def _verify_max_two_countries(scoped: pd.DataFrame) -> tuple[int, int]:
    """Recompute FR-18's verified fact across ALL in-scope products.

    Returns (max_distinct_countries, num_products_with_3_or_more).
    """
    distinct_country_counts = scoped.groupby("ProductID")["Country"].nunique()
    max_distinct = int(distinct_country_counts.max())
    num_three_plus = int((distinct_country_counts >= 3).sum())

    assert max_distinct <= 2, (
        f"FR-18 violated: found a product with {max_distinct} distinct destination "
        "countries; expected the verified maximum of 2."
    )
    assert num_three_plus == 0, (
        f"FR-18 violated: {num_three_plus} in-scope product(s) have 3+ distinct "
        "destination countries; expected 0."
    )
    return max_distinct, num_three_plus


def _build_product_country_weekly(scoped: pd.DataFrame, product_ids: list[str]) -> pd.DataFrame:
    """Aggregate scoped transactions to (ProductID, Country, WeekStart, Sales_Qty).

    Mirrors src.cleaning.aggregate_weekly()'s ISO-week (Monday-start) derivation,
    but groups by Country as well so the per-country split survives.
    """
    subset = scoped.loc[scoped["ProductID"].isin(product_ids), ["ProductID", "Country", "Order Date", "Sales Qty"]].copy()
    subset["Order Date"] = pd.to_datetime(subset["Order Date"], errors="coerce")
    if subset["Order Date"].isna().any():
        raise AssertionError("Demo product subset contains null/unparsable Order Date values")

    subset["WeekStart"] = subset["Order Date"] - pd.to_timedelta(subset["Order Date"].dt.weekday, unit="D")
    subset["WeekStart"] = subset["WeekStart"].dt.normalize()

    weekly = (
        subset.groupby(["ProductID", "Country", "WeekStart"], sort=True, dropna=False)["Sales Qty"]
        .sum()
        .reset_index()
        .rename(columns={"Sales Qty": "Sales_Qty"})
    )
    weekly["Sales_Qty"] = weekly["Sales_Qty"].clip(lower=0)
    return weekly


def _forecast_pair(history: pd.DataFrame, product_id: str, country: str) -> pd.DataFrame:
    """Forecast 12 weeks ahead for one (ProductID, Country) pair.

    Method: rolling mean of the last up-to-ROLLING_WINDOW observed weekly
    values, carried forward flat across the whole horizon. Falls back to the
    single last observed value when fewer than ROLLING_WINDOW weeks exist
    (equivalent to last-observation-carried-forward for single-observation
    series).
    """
    ordered = history.sort_values("WeekStart")
    recent_values = ordered["Sales_Qty"].tail(ROLLING_WINDOW)
    predicted_value = float(recent_values.mean())
    predicted_value = max(predicted_value, 0.0)

    forecast_weeks = pd.date_range(start=FORECAST_ORIGIN, periods=HORIZON_WEEKS, freq="W-MON")
    return pd.DataFrame(
        {
            "ProductID": product_id,
            "Country": country,
            "forecast_week": range(1, HORIZON_WEEKS + 1),
            "WeekStart": forecast_weeks,
            "predicted_Sales_Qty": predicted_value,
            "method": METHOD_NAME,
        }
    )


def run_country_demo(scoped_path: Path = SCOPED_PATH, output_path: Path | None = OUTPUT_PATH) -> pd.DataFrame:
    if not scoped_path.exists():
        raise FileNotFoundError(f"Scoped parquet not found: {scoped_path}")

    scoped = pd.read_parquet(scoped_path)
    required = {"ProductID", "Country", "Order Date", "Sales Qty"}
    missing = required.difference(scoped.columns)
    if missing:
        raise ValueError(f"Scoped dataset is missing required columns: {sorted(missing)}")

    max_distinct, num_three_plus = _verify_max_two_countries(scoped)

    weekly = _build_product_country_weekly(scoped, DEMO_PRODUCT_IDS)

    pairs = weekly[["ProductID", "Country"]].drop_duplicates().sort_values(["ProductID", "Country"])
    if len(pairs) == 0:
        raise AssertionError("No (ProductID, Country) pairs found for the demo product subset")

    forecasts = []
    for _, row in pairs.iterrows():
        product_id, country = row["ProductID"], row["Country"]
        history = weekly[(weekly["ProductID"] == product_id) & (weekly["Country"] == country)]
        forecasts.append(_forecast_pair(history, product_id, country))

    result = pd.concat(forecasts, ignore_index=True)
    result = result[["ProductID", "Country", "forecast_week", "WeekStart", "predicted_Sales_Qty", "method"]]

    if (result["predicted_Sales_Qty"] < 0).any():
        raise AssertionError("Country-demo forecasts must be nonnegative")
    if len(result) != len(pairs) * HORIZON_WEEKS:
        raise AssertionError("Expected exactly HORIZON_WEEKS rows per (ProductID, Country) pair")

    result.attrs["max_distinct_countries"] = max_distinct
    result.attrs["num_products_with_3plus_countries"] = num_three_plus
    result.attrs["num_pairs"] = len(pairs)
    result.attrs["pairs"] = list(pairs.itertuples(index=False, name=None))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def main() -> None:
    with stage(LOGGER, "Country-Level Drill-Down Demo (FR-16-FR-19)"):
        result = run_country_demo()

        LOGGER.info(
            "Country-level forecasting is demonstrated on 3 products only because, "
            "among all in-scope cinnamon products, %d have 3+ distinct destination "
            "countries (verified max = %d).",
            result.attrs["num_products_with_3plus_countries"],
            result.attrs["max_distinct_countries"],
        )

        log_metrics(
            LOGGER,
            {
                "demo_products": ", ".join(DEMO_PRODUCT_IDS),
                "product_country_pairs": result.attrs["num_pairs"],
                "pairs": ", ".join(f"{pid}/{c}" for pid, c in result.attrs["pairs"]),
                "method": METHOD_NAME,
                "forecast_origin": str(FORECAST_ORIGIN.date()),
                "horizon_weeks": HORIZON_WEEKS,
                "output_rows": len(result),
                "output": OUTPUT_PATH,
            },
        )


if __name__ == "__main__":
    main()
