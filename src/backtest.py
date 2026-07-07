"""Model comparison / backtest metrics table (spec 003 FR-20..FR-23, plan §4.5).

Combines the Fold 1 validation predictions already written by
`src/models/high_volume.py` (T005) and `src/models/sparse.py` (T006) into one
table using `src/metrics.py` (T002) -- never reimplements RMSE/MAE/WMAPE/MASE
inline. High-volume rows report RMSE/MAE; sparse rows report WMAPE/MASE
(FR-10/FR-15/FR-21). No random split is used anywhere in this module (T-6).

Sparse validation predictions can have a NaN actual (`y`) for a product-week
where the product had not yet been observed at all in `featured.parquet`
(panel starts at first observation, not at the fold's train cutoff) -- those
rows are excluded from scoring since there is no real demand to compare
against, not because of a data-quality problem.

`statsforecast` fits all four sparse models (CrostonClassic/CrostonSBA/TSB/
Naive) in a single batched call, so per-model train/inference timing isn't
separately available for the sparse tier; the stage-level aggregate from
`sparse/timing.parquet` is reported for every sparse model row (documented
simplification).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.logging_utils import get_logger, stage
from src.metrics import mae, mase, rmse, wmape
from src.models.sparse import FOLD1_TRAIN_END, build_sparse_long_frame

LOGGER = get_logger(__name__)

HV_VALID_PATH = Path("outputs/models/high_volume/validation_predictions.parquet")
HV_TIMING_PATH = Path("outputs/models/high_volume/timing.json")
SPARSE_VALID_PATH = Path("outputs/models/sparse/validation_predictions.parquet")
SPARSE_TIMING_PATH = Path("outputs/models/sparse/timing.parquet")
TIERS_PATH = Path("data/processed/tiers.parquet")
FEATURED_PATH = Path("data/processed/featured.parquet")

OUTPUT_PATH = Path("outputs/metrics/model_comparison.csv")

HV_BASELINE_MODELS = {"seasonal_naive"}
SPARSE_BASELINE_MODELS = {"Naive"}

METRIC_COLUMNS = ["rmse", "mae", "wmape", "mase"]


def score_high_volume(valid: pd.DataFrame, timing: dict) -> pd.DataFrame:
    """One row per high-volume model: RMSE/MAE plus timing from `timing.json`.

    Baseline models (`seasonal_naive`) require no fit step; `train_seconds`
    is reported as 0.0 for them rather than left missing.
    """
    rows = []
    for model, group in valid.groupby("model"):
        rows.append(
            {
                "tier": "high_volume",
                "model": model,
                "is_baseline": model in HV_BASELINE_MODELS,
                "rmse": rmse(group["y_true"], group["y_pred"]),
                "mae": mae(group["y_true"], group["y_pred"]),
                "wmape": None,
                "mase": None,
                "train_seconds": timing.get(f"{model}_train_seconds", 0.0),
                "inference_seconds": timing.get(f"{model}_inference_seconds"),
            }
        )
    return pd.DataFrame(rows)


def score_sparse(
    valid: pd.DataFrame, history_by_id: dict[str, list[float]], timing: pd.DataFrame
) -> pd.DataFrame:
    """One row per sparse model: WMAPE (globally pooled) and MASE (mean of
    per-`unique_id` MASE, each scaled by that product's own training history).
    """
    fold1 = timing.loc[timing["stage"] == "fold1_validation"].iloc[0]
    scored = valid.dropna(subset=["y"])

    rows = []
    for model, group in scored.groupby("model"):
        per_series_mase = [
            mase(g["y"], g["y_pred"], history_by_id.get(uid, []), season_length=1)
            for uid, g in group.groupby("unique_id")
        ]
        rows.append(
            {
                "tier": "sparse",
                "model": model,
                "is_baseline": model in SPARSE_BASELINE_MODELS,
                "rmse": None,
                "mae": None,
                "wmape": wmape(group["y"], group["y_pred"]),
                "mase": float(pd.Series(per_series_mase).mean()) if per_series_mase else 0.0,
                "train_seconds": float(fold1["train_seconds"]),
                "inference_seconds": float(fold1["inference_seconds"]),
            }
        )
    return pd.DataFrame(rows)


def build_sparse_history(tiers: pd.DataFrame, featured: pd.DataFrame) -> dict[str, list[float]]:
    """Per-`unique_id` list of `y` values for weeks <= the Fold 1 train cutoff.

    Reuses `build_sparse_long_frame` (T006) rather than re-deriving the
    long-format reshape rules.
    """
    sparse_ids = tiers.loc[tiers["tier"] == "sparse", "ProductID"]
    long_df = build_sparse_long_frame(featured, sparse_ids)
    train_df = long_df[long_df["ds"] <= FOLD1_TRAIN_END]
    return {uid: g["y"].tolist() for uid, g in train_df.groupby("unique_id")}


def build_model_comparison() -> pd.DataFrame:
    hv_valid = pd.read_parquet(HV_VALID_PATH)
    hv_timing = json.loads(HV_TIMING_PATH.read_text())
    hv_table = score_high_volume(hv_valid, hv_timing)

    sparse_valid = pd.read_parquet(SPARSE_VALID_PATH)
    sparse_timing = pd.read_parquet(SPARSE_TIMING_PATH)
    tiers = pd.read_parquet(TIERS_PATH)
    featured = pd.read_parquet(FEATURED_PATH)
    history_by_id = build_sparse_history(tiers, featured)
    sparse_table = score_sparse(sparse_valid, history_by_id, sparse_timing)

    table = pd.concat([hv_table, sparse_table], ignore_index=True)
    return table[
        ["tier", "model", "is_baseline"] + METRIC_COLUMNS + ["train_seconds", "inference_seconds"]
    ]


def main() -> None:
    with stage(LOGGER, "Backtest: model comparison table"):
        table = build_model_comparison()
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUTPUT_PATH, index=False)
        LOGGER.info("Wrote %d rows to %s", len(table), OUTPUT_PATH)
        LOGGER.info("\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
