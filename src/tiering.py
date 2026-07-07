from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.logging_utils import get_logger, log_metrics, stage

LOGGER = get_logger(__name__)

FEATURED_PATH = Path("data/processed/featured.parquet")
TIERS_PATH = Path("data/processed/tiers.parquet")
HIGH_VOLUME_TRANSACTION_THRESHOLD = 10


def tier_products(featured_path: Path = FEATURED_PATH, output_path: Path | None = TIERS_PATH) -> pd.DataFrame:
    if not featured_path.exists():
        raise FileNotFoundError(f"Featured parquet not found: {featured_path}")

    featured = pd.read_parquet(featured_path, columns=["ProductID", "Sales_Qty", "transaction_count"])
    product_summary = (
        featured.groupby("ProductID", sort=True)
        .agg(transaction_count=("transaction_count", "sum"), total_sales_qty=("Sales_Qty", "sum"))
        .reset_index()
    )
    product_summary["tier"] = product_summary["transaction_count"].ge(HIGH_VOLUME_TRANSACTION_THRESHOLD).map(
        {True: "high_volume", False: "sparse"}
    )

    if product_summary["ProductID"].duplicated().any():
        raise AssertionError("Tier output must have exactly one row per ProductID")
    if product_summary["tier"].isna().any():
        raise AssertionError("Every ProductID must receive exactly one tier")

    high_volume = set(product_summary.loc[product_summary["tier"] == "high_volume", "ProductID"])
    sparse = set(product_summary.loc[product_summary["tier"] == "sparse", "ProductID"])
    all_products = set(product_summary["ProductID"])
    if high_volume & sparse:
        raise AssertionError("Tier partition overlap detected")
    if (high_volume | sparse) != all_products:
        raise AssertionError("Tier partition gap detected")

    total_volume = product_summary["total_sales_qty"].sum()
    high_volume_rows = product_summary["tier"].eq("high_volume")
    high_volume_volume = product_summary.loc[high_volume_rows, "total_sales_qty"].sum()
    product_summary.attrs["products"] = len(product_summary)
    product_summary.attrs["high_volume_products"] = int(high_volume_rows.sum())
    product_summary.attrs["sparse_products"] = int((~high_volume_rows).sum())
    product_summary.attrs["high_volume_volume_share"] = float(high_volume_volume / total_volume) if total_volume else 0.0

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        product_summary.to_parquet(output_path, index=False)

    return product_summary


def main() -> None:
    with stage(LOGGER, "Stage 6: tier_products()"):
        tiers = tier_products()
        log_metrics(
            LOGGER,
            {
                "threshold_transactions": HIGH_VOLUME_TRANSACTION_THRESHOLD,
                "products": tiers.attrs["products"],
                "high_volume_products": tiers.attrs["high_volume_products"],
                "sparse_products": tiers.attrs["sparse_products"],
                "high_volume_volume_share": tiers.attrs["high_volume_volume_share"],
                "output": TIERS_PATH,
            },
        )


if __name__ == "__main__":
    main()
