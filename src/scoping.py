from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.logging_utils import get_logger, log_metrics, stage

LOGGER = get_logger(__name__)

CLEANED_PATH = Path("data/processed/cleaned.parquet")
SCOPED_PATH = Path("data/processed/scoped.parquet")

MERCHANDISE_RANGES = {
    "APPAREL",
    "BROCHURES / MAGAZINE",
    "GLASS / CERAMIC/ PORCELAIN",
    "PACKAGING",
    "POINT OF SALE MATERIALS",
    "PROMOTIONAL MATERIAL",
    "PVC/PRESENTERS",
    "WOODEN BOXES / PRESENTERS",
}

RUBBER_LATEX_RANGES = {
    "BULK RUBBER",
    "CENTRIFUGED LATEX",
    "PALE CREPE",
    "PREMIUM CREPE GRADE",
    "PREMIUM LATEX GRADE",
    "PROCESSED SHEETS",
    "WATTE REGIONAL GRADE",
    "WATTE SINGLE ESTATE GRADE",
}

EXCLUDED_PRODUCT_RANGES = MERCHANDISE_RANGES | RUBBER_LATEX_RANGES
SPLIT_CATEGORY_PRODUCT_IDS = {"14998-097", "53704-013", "56500-075", "72065-040"}


def scope_products(cleaned_path: Path = CLEANED_PATH, output_path: Path | None = SCOPED_PATH) -> pd.DataFrame:
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned parquet not found: {cleaned_path}")

    cleaned = pd.read_parquet(cleaned_path)
    if "ProductID" not in cleaned.columns or "Product Range" not in cleaned.columns:
        raise ValueError("Cleaned dataset must contain ProductID and Product Range columns")

    product_range = cleaned["Product Range"].astype("string").str.strip().str.upper()
    excluded_mask = product_range.isin(EXCLUDED_PRODUCT_RANGES)
    scoped = cleaned.loc[~excluded_mask].copy()

    excluded = cleaned.loc[excluded_mask]
    retained_split = set(scoped.loc[scoped["ProductID"].isin(SPLIT_CATEGORY_PRODUCT_IDS), "ProductID"])
    excluded_split = set(excluded.loc[excluded["ProductID"].isin(SPLIT_CATEGORY_PRODUCT_IDS), "ProductID"])
    if retained_split != SPLIT_CATEGORY_PRODUCT_IDS or excluded_split != SPLIT_CATEGORY_PRODUCT_IDS:
        raise AssertionError(
            "Split-category ProductIDs must have both retained and excluded rows: "
            f"retained={sorted(retained_split)}, excluded={sorted(excluded_split)}"
        )

    remaining_excluded_ranges = (
        scoped["Product Range"].astype("string").str.strip().str.upper().isin(EXCLUDED_PRODUCT_RANGES)
    )
    if remaining_excluded_ranges.any():
        raise AssertionError("Scoped dataset still contains excluded merchandise/rubber product ranges")

    scoped.attrs["cleaned_rows"] = len(cleaned)
    scoped.attrs["excluded_rows"] = int(excluded_mask.sum())
    scoped.attrs["scoped_rows"] = len(scoped)
    scoped.attrs["cleaned_products"] = int(cleaned["ProductID"].nunique())
    scoped.attrs["scoped_products"] = int(scoped["ProductID"].nunique())

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scoped.to_parquet(output_path, index=False)

    return scoped


def main() -> None:
    with stage(LOGGER, "Stage 3: scope_products()"):
        scoped = scope_products()
        log_metrics(
            LOGGER,
            {
                "cleaned_rows": scoped.attrs["cleaned_rows"],
                "excluded_rows": scoped.attrs["excluded_rows"],
                "scoped_rows": scoped.attrs["scoped_rows"],
                "cleaned_products": scoped.attrs["cleaned_products"],
                "scoped_products": scoped.attrs["scoped_products"],
                "output": SCOPED_PATH,
            },
        )


if __name__ == "__main__":
    main()
