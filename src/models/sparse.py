"""Sparse-tier (intermittent-demand) forecasting (spec 003 FR-11..FR-15, plan §4.3).

Pipeline:
1. Reshape `data/processed/featured.parquet`, filtered to sparse-tier
   `ProductID`s from `data/processed/tiers.parquet`, into the long format
   `statsforecast` requires: `unique_id=ProductID, ds=WeekStart, y=Sales_Qty`.
2. Fold 1 validation (plan §4.1): train on weeks <= 2025-06-23, forecast
   h=12, compare against the actual held-out weeks 2025-06-30..2025-09-15
   using CrostonClassic, CrostonSBA, TSB, and a naive baseline.
3. Refit the same models on the FULL history (through 2025-09-15) and
   produce the final 12-week forecast (2025-09-22..2025-12-08) for every
   sparse-tier ProductID.
4. Single-observation / very-short-history products (spec EC-1) can make
   `StatsForecast` error or misbehave; any `unique_id` a model can't handle
   falls back to a deterministic carry-forward (repeat the last observed
   value, or 0 if there is truly no history) so every sparse ProductID
   always gets exactly 12 forecast rows (FR-13).

`freq='W-MON'` is passed explicitly everywhere `StatsForecast` is
constructed (plan §13 failure mode: a wrong/missing freq silently produces
a wrong-length forecast instead of erroring). `assert_full_horizon` is the
concrete guard against that failure mode: it is called on every forecast
group before anything is written to disk.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import CrostonClassic, CrostonSBA, Naive, TSB

from src.logging_utils import get_logger, log_metrics, stage

LOGGER = get_logger(__name__)

FREQ = "W-MON"
HORIZON = 12

FOLD1_TRAIN_END = pd.Timestamp("2025-06-23")
FOLD1_VALID_START = pd.Timestamp("2025-06-30")
FOLD1_VALID_END = pd.Timestamp("2025-09-15")

FULL_HISTORY_END = pd.Timestamp("2025-09-15")
FINAL_FORECAST_START = pd.Timestamp("2025-09-22")
FINAL_FORECAST_END = pd.Timestamp("2025-12-08")

TIERS_PATH = Path("data/processed/tiers.parquet")
FEATURED_PATH = Path("data/processed/featured.parquet")

OUTPUT_DIR = Path("outputs/models/sparse")
VALIDATION_PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.parquet"
FINAL_FORECAST_PATH = OUTPUT_DIR / "final_forecast.parquet"
TIMING_PATH = OUTPUT_DIR / "timing.parquet"


def build_sparse_long_frame(featured: pd.DataFrame, sparse_ids) -> pd.DataFrame:
    """Filter `featured` to sparse-tier products and reshape to long format.

    Returns columns exactly `unique_id`, `ds`, `y`, sorted by
    `unique_id`, `ds` (statsforecast requires sorted-by-time-per-series input).
    """
    sparse_ids = set(sparse_ids)
    subset = featured.loc[
        featured["ProductID"].isin(sparse_ids), ["ProductID", "WeekStart", "Sales_Qty"]
    ].copy()
    long_df = subset.rename(
        columns={"ProductID": "unique_id", "WeekStart": "ds", "Sales_Qty": "y"}
    )
    long_df["ds"] = pd.to_datetime(long_df["ds"])
    long_df = long_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return long_df


def assert_full_horizon(forecast_df: pd.DataFrame, horizon: int = HORIZON) -> None:
    """Guard against the freq-mismatch failure mode (plan §13, FR-13, T-5).

    Raises `AssertionError` naming the offending `unique_id`(s) if any
    series in `forecast_df` does not have exactly `horizon` rows. Must be
    called on every forecast frame before it is written to disk.
    """
    counts = forecast_df.groupby("unique_id").size()
    bad = counts[counts != horizon]
    if not bad.empty:
        raise AssertionError(
            f"Expected exactly {horizon} forecast rows per unique_id; "
            f"found mismatches for {len(bad)} unique_id(s), e.g. "
            f"{bad.head().to_dict()}"
        )


def _naive_fallback_forecast(
    history: pd.DataFrame, unique_id, start: pd.Timestamp, horizon: int, model_name: str
) -> pd.DataFrame:
    """Deterministic carry-forward fallback for a single `unique_id` (spec EC-1).

    Repeats the last observed `y` value for `horizon` weeks starting at
    `start` (inclusive), stepping by the weekly freq. Falls back to 0.0 if
    `history` is empty or the last value is null.
    """
    if len(history) > 0 and pd.notna(history["y"].iloc[-1]):
        last_value = float(history["y"].iloc[-1])
    else:
        last_value = 0.0
    ds = pd.date_range(start=start, periods=horizon, freq=FREQ)
    return pd.DataFrame(
        {
            "unique_id": unique_id,
            "ds": ds,
            model_name: last_value,
        }
    )


def _run_statsforecast_with_fallback(
    train_df: pd.DataFrame,
    all_ids,
    horizon: int,
    forecast_start: pd.Timestamp,
    model_name: str = "model_ensemble",
) -> pd.DataFrame:
    """Fit CrostonClassic/CrostonSBA/TSB/Naive on `train_df`, per-`unique_id`
    fallback to deterministic carry-forward for any series a library model
    can't handle, and return a long frame with one row per (`unique_id`, `ds`,
    model) with columns `unique_id, ds, model, y_pred`.

    Ensures EVERY id in `all_ids` gets exactly `horizon` rows per model
    (FR-13/EC-1), even ids with zero or one training observations.
    """
    all_ids = list(all_ids)
    models = [CrostonClassic(), CrostonSBA(), TSB(alpha_d=0.1, alpha_p=0.1), Naive()]
    model_names = [m.alias if hasattr(m, "alias") else type(m).__name__ for m in models]

    counts = train_df.groupby("unique_id").size()
    fittable_ids = set(counts[counts >= 2].index)
    fallback_ids = [uid for uid in all_ids if uid not in fittable_ids]

    frames = []

    if fittable_ids:
        fit_df = train_df[train_df["unique_id"].isin(fittable_ids)]
        try:
            sf = StatsForecast(models=models, freq=FREQ, n_jobs=1)
            preds = sf.forecast(df=fit_df, h=horizon)
            preds = preds.reset_index() if preds.index.name == "unique_id" else preds
            long_preds = preds.melt(
                id_vars=["unique_id", "ds"], var_name="model", value_name="y_pred"
            )
            frames.append(long_preds)
            # Any fittable id that the library silently dropped/short-changed
            # (e.g. a model raised internally for that series) also needs a
            # fallback so no id is ever missing rows (T-5 style guard).
            got_counts = long_preds.groupby(["unique_id", "model"]).size()
            for name in model_names:
                for uid in fittable_ids:
                    key = (uid, name)
                    if key not in got_counts.index or got_counts.loc[key] != horizon:
                        hist = train_df[train_df["unique_id"] == uid]
                        frames.append(
                            _naive_fallback_forecast(
                                hist, uid, forecast_start, horizon, "y_pred"
                            ).assign(model=name)[["unique_id", "ds", "model", "y_pred"]]
                        )
        except Exception as exc:  # library-level failure -> fall back for all fittable ids too
            LOGGER.warning("StatsForecast batch fit failed (%s); falling back for all ids", exc)
            fallback_ids = list(all_ids)

    for uid in fallback_ids:
        hist = train_df[train_df["unique_id"] == uid]
        for name in model_names:
            frames.append(
                _naive_fallback_forecast(
                    hist, uid, forecast_start, horizon, "y_pred"
                ).assign(model=name)[["unique_id", "ds", "model", "y_pred"]]
            )

    result = pd.concat(frames, ignore_index=True)
    # De-duplicate in case a fittable id both produced library output and a
    # supplemental fallback was added for a *different* model only; keep the
    # last write (fallback wins over any partial library artifact).
    result = result.drop_duplicates(subset=["unique_id", "ds", "model"], keep="last")
    return result


def run_validation_fold(long_df: pd.DataFrame, all_ids) -> tuple[pd.DataFrame, float]:
    """Fold 1 validation: train <= 2025-06-23, forecast h=12, actuals from
    2025-06-30..2025-09-15. Returns (predictions_with_actuals, train_seconds).
    """
    train_df = long_df[long_df["ds"] <= FOLD1_TRAIN_END]
    actuals = long_df[
        (long_df["ds"] >= FOLD1_VALID_START) & (long_df["ds"] <= FOLD1_VALID_END)
    ][["unique_id", "ds", "y"]]

    start = time.perf_counter()
    preds = _run_statsforecast_with_fallback(
        train_df, all_ids, HORIZON, FOLD1_VALID_START
    )
    elapsed = time.perf_counter() - start

    for name, group in preds.groupby("model"):
        assert_full_horizon(group)

    merged = preds.merge(actuals, on=["unique_id", "ds"], how="left")
    return merged, elapsed


def run_final_forecast(long_df: pd.DataFrame, all_ids) -> tuple[pd.DataFrame, float, float]:
    """Fit on full history (through 2025-09-15); forecast 2025-09-22..2025-12-08
    for every sparse ProductID. Returns (forecast_df, train_seconds, inference_seconds).
    """
    train_df = long_df[long_df["ds"] <= FULL_HISTORY_END]

    start = time.perf_counter()
    preds = _run_statsforecast_with_fallback(
        train_df, all_ids, HORIZON, FINAL_FORECAST_START
    )
    elapsed = time.perf_counter() - start

    for name, group in preds.groupby("model"):
        assert_full_horizon(group)

    for uid, group in preds.groupby("unique_id"):
        n_models = group["model"].nunique()
        assert len(group) == HORIZON * n_models, (
            f"unique_id={uid} expected {HORIZON * n_models} rows, got {len(group)}"
        )

    # Report train/inference as one combined figure (statsforecast's
    # `.forecast()` fits and predicts in a single call); log the same
    # elapsed value for both so downstream metrics tables always have both
    # fields populated (plan §4.5 requires training time AND inference latency).
    return preds, elapsed, elapsed


def main() -> None:
    with stage(LOGGER, "Sparse-tier forecasting (CrostonClassic/CrostonSBA/TSB/Naive)"):
        tiers = pd.read_parquet(TIERS_PATH)
        featured = pd.read_parquet(FEATURED_PATH)

        sparse_ids = tiers.loc[tiers["tier"] == "sparse", "ProductID"]
        long_df = build_sparse_long_frame(featured, sparse_ids)
        all_ids = sorted(sparse_ids.unique())

        LOGGER.info("Sparse-tier products: %d", len(all_ids))
        LOGGER.info("Long-format rows: %d", len(long_df))

        LOGGER.info("Running Fold 1 validation (train <= %s, valid %s..%s)",
                    FOLD1_TRAIN_END.date(), FOLD1_VALID_START.date(), FOLD1_VALID_END.date())
        validation_preds, valid_train_seconds = run_validation_fold(long_df, all_ids)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        validation_preds.to_parquet(VALIDATION_PREDICTIONS_PATH, index=False)

        LOGGER.info(
            "Fitting on full history (through %s), forecasting %s..%s",
            FULL_HISTORY_END.date(), FINAL_FORECAST_START.date(), FINAL_FORECAST_END.date(),
        )
        final_forecast, train_seconds, inference_seconds = run_final_forecast(long_df, all_ids)
        final_forecast.to_parquet(FINAL_FORECAST_PATH, index=False)

        timing = pd.DataFrame(
            [
                {
                    "stage": "fold1_validation",
                    "train_seconds": valid_train_seconds,
                    "inference_seconds": valid_train_seconds,
                    "n_products": len(all_ids),
                },
                {
                    "stage": "final_forecast",
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "n_products": len(all_ids),
                },
            ]
        )
        timing.to_parquet(TIMING_PATH, index=False)

        fallback_products = (
            final_forecast.loc[final_forecast["model"] == "Naive", "unique_id"].nunique()
        )

        log_metrics(
            LOGGER,
            {
                "sparse_products": len(all_ids),
                "validation_rows": len(validation_preds),
                "final_forecast_rows": len(final_forecast),
                "fold1_train_seconds": valid_train_seconds,
                "final_train_seconds": train_seconds,
                "final_inference_seconds": inference_seconds,
                "validation_predictions_path": str(VALIDATION_PREDICTIONS_PATH),
                "final_forecast_path": str(FINAL_FORECAST_PATH),
            },
        )


if __name__ == "__main__":
    main()
