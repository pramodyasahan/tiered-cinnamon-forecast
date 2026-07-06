from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/Cinnamon_export_sales.xlsx")
CLEANED_PATH = Path("data/processed/cleaned.parquet")
SCOPED_PATH = Path("data/processed/scoped.parquet")
WEEKLY_PATH = Path("data/processed/weekly.parquet")
ARTIFACT_INDICES = [60667, 60668, 60669]
REQUIRED_COLUMNS = {
    "Region",
    "Country",
    "Customer Code",
    "Customer ID",
    "Brand Category",
    "Product Range",
    "Sales Channel",
    "Product Code",
    "Order Date",
    "Invoice Date",
    "Invoice No",
    "Sales USD",
    "Sales Qty",
    "Sales KG",
}
CATEGORICAL_MODE_COLUMNS = [
    "Region",
    "Country",
    "Sales Channel",
    "Brand Category",
    "Product Range",
]


def _read_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw workbook not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Raw workbook is missing required columns: {sorted(missing)}")
    return df


def _drop_export_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    artifact_rows = df.loc[df.index.intersection(ARTIFACT_INDICES)]
    if len(artifact_rows) != 3:
        raise AssertionError(
            f"Expected 3 export-artifact rows at indices {ARTIFACT_INDICES}, found {len(artifact_rows)}"
        )

    regions = artifact_rows["Region"].astype("string")
    if not (
        (regions.iloc[0] == "Total")
        and regions.iloc[1:].isna().iloc[0]
        and regions.iloc[2].startswith("Applied filters:")
    ):
        raise AssertionError("Export-artifact rows did not match the expected Total/spacer/filter pattern")

    return df.drop(index=ARTIFACT_INDICES)


def clean_transactions(raw_path: Path = RAW_PATH, output_path: Path | None = CLEANED_PATH) -> pd.DataFrame:
    """Clean raw transactions through the spec's Stage 2 contract."""
    raw = _read_raw(raw_path)
    post_artifact = _drop_export_artifacts(raw)
    duplicate_count = int(post_artifact.duplicated().sum())
    cleaned = post_artifact.drop_duplicates().copy()

    cleaned["ProductID"] = cleaned["Product Code"].astype("string").str.slice(0, 9)
    if cleaned["ProductID"].isna().any() or (cleaned["ProductID"].str.len() != 9).any():
        raise AssertionError("Every cleaned row must have a 9-character ProductID")

    cleaned["Order Date"] = pd.to_datetime(cleaned["Order Date"], errors="coerce")
    cleaned["Invoice Date"] = pd.to_datetime(cleaned["Invoice Date"], errors="coerce")

    complete_date_mask = cleaned["Order Date"].notna() & cleaned["Invoice Date"].notna()
    lead_days = (cleaned.loc[complete_date_mask, "Invoice Date"] - cleaned.loc[complete_date_mask, "Order Date"]).dt.days
    median_lead_days = int(lead_days.median())
    negative_lead_rate = float((lead_days < 0).mean())

    missing_order_mask = cleaned["Order Date"].isna()
    if cleaned.loc[missing_order_mask, "Invoice Date"].isna().any():
        raise AssertionError("Cannot impute Order Date for rows with missing Invoice Date")
    cleaned.loc[missing_order_mask, "Order Date"] = (
        cleaned.loc[missing_order_mask, "Invoice Date"] - pd.to_timedelta(median_lead_days, unit="D")
    )

    if cleaned["Order Date"].isna().any():
        raise AssertionError("Order Date imputation failed; null values remain")

    cleaned.attrs["raw_rows"] = len(raw)
    cleaned.attrs["artifact_rows_dropped"] = len(raw) - len(post_artifact)
    cleaned.attrs["duplicate_rows_dropped"] = duplicate_count
    cleaned.attrs["missing_order_dates_imputed"] = int(missing_order_mask.sum())
    cleaned.attrs["median_lead_days"] = median_lead_days
    cleaned.attrs["negative_lead_rate"] = negative_lead_rate

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(output_path, index=False)

    return cleaned


def _mode_or_na(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.value_counts(sort=True).index[0]


def aggregate_weekly(scoped_path: Path = SCOPED_PATH, output_path: Path | None = WEEKLY_PATH) -> pd.DataFrame:
    if not scoped_path.exists():
        raise FileNotFoundError(f"Scoped parquet not found: {scoped_path}")

    scoped = pd.read_parquet(scoped_path)
    required = {"ProductID", "Order Date", "Sales Qty", "Sales USD", *CATEGORICAL_MODE_COLUMNS}
    missing = required.difference(scoped.columns)
    if missing:
        raise ValueError(f"Scoped dataset is missing required columns: {sorted(missing)}")

    transactions = scoped.copy()
    transactions["Order Date"] = pd.to_datetime(transactions["Order Date"], errors="coerce")
    if transactions["Order Date"].isna().any():
        raise AssertionError("Scoped dataset contains null or unparsable Order Date values")

    transactions["WeekStart"] = transactions["Order Date"] - pd.to_timedelta(
        transactions["Order Date"].dt.weekday, unit="D"
    )
    transactions["WeekStart"] = transactions["WeekStart"].dt.normalize()

    grouped = transactions.groupby(["ProductID", "WeekStart"], sort=True, dropna=False)
    numeric = grouped.agg(
        Sales_Qty_raw=("Sales Qty", "sum"),
        Sales_USD=("Sales USD", "sum"),
        transaction_count=("ProductID", "size"),
    )
    modes = grouped[CATEGORICAL_MODE_COLUMNS].agg(_mode_or_na)
    weekly = numeric.join(modes).reset_index()
    weekly["returns_exceeded_sales"] = weekly["Sales_Qty_raw"] < 0
    weekly["Sales_Qty"] = weekly["Sales_Qty_raw"].clip(lower=0)
    weekly = weekly.drop(columns=["Sales_Qty_raw"])

    ordered_columns = [
        "ProductID",
        "WeekStart",
        "Sales_Qty",
        "Sales_USD",
        "returns_exceeded_sales",
        "transaction_count",
        *CATEGORICAL_MODE_COLUMNS,
    ]
    weekly = weekly[ordered_columns]

    if weekly.duplicated(["ProductID", "WeekStart"]).any():
        raise AssertionError("Weekly output must contain one row per ProductID and WeekStart")
    if (weekly["Sales_Qty"] < 0).any():
        raise AssertionError("Weekly Sales_Qty must be floored at zero")

    weekly.attrs["scoped_rows"] = len(scoped)
    weekly.attrs["weekly_rows"] = len(weekly)
    weekly.attrs["returns_exceeded_sales_weeks"] = int(weekly["returns_exceeded_sales"].sum())
    weekly.attrs["min_week"] = str(weekly["WeekStart"].min().date())
    weekly.attrs["max_week"] = str(weekly["WeekStart"].max().date())

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_parquet(output_path, index=False)

    return weekly


def main() -> None:
    cleaned = clean_transactions()
    print(f"raw_rows={cleaned.attrs['raw_rows']}")
    print(f"artifact_rows_dropped={cleaned.attrs['artifact_rows_dropped']}")
    print(f"duplicate_rows_dropped={cleaned.attrs['duplicate_rows_dropped']}")
    print(f"missing_order_dates_imputed={cleaned.attrs['missing_order_dates_imputed']}")
    print(f"median_lead_days={cleaned.attrs['median_lead_days']}")
    print(f"negative_lead_rate={cleaned.attrs['negative_lead_rate']:.4f}")
    print(f"cleaned_rows={len(cleaned)}")
    print(f"output={CLEANED_PATH}")

    if SCOPED_PATH.exists():
        weekly = aggregate_weekly()
        print(f"scoped_rows={weekly.attrs['scoped_rows']}")
        print(f"weekly_rows={weekly.attrs['weekly_rows']}")
        print(f"returns_exceeded_sales_weeks={weekly.attrs['returns_exceeded_sales_weeks']}")
        print(f"week_range={weekly.attrs['min_week']}..{weekly.attrs['max_week']}")
        print(f"output={WEEKLY_PATH}")


if __name__ == "__main__":
    main()
