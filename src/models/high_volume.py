"""High-volume tier forecasting: LightGBM (primary) + XGBoost (comparison) + seasonal-naive baseline.

Implements spec 003 FR-6..FR-10 (plan.md §4.2) for the 1,525 ``tier=="high_volume"``
products in ``data/processed/tiers.parquet`` (79.4% of total volume).

Data split (plan.md §4.1, Fold 1 only -- required; folds 2-3 skipped, see note below):
    Train:      featured rows with WeekStart <= 2025-06-23
    Validate:   featured rows with WeekStart in [2025-06-30, 2025-09-15]
                (the last 12 observed weeks -- matches the 12-week forecast horizon)

Fold 2-3 decision: SKIPPED. Plan §4.1 marks the expanding-window folds 2-3 as
"if time allows" and explicitly allows "Fold 1 alone, clearly documented as a
single time-ordered holdout" as the minimum acceptable fallback. This module
implements exactly that minimum: one time-ordered holdout, not a K-fold average.
No `train_test_split` and no `shuffle=True` appear anywhere in this file (grep-able
guarantee for plan §13 / test-strategy T-6).

Categorical handling: `src/models/categorical.py` (T004) is used for ALL categorical
encoding. `fit_categorical_dtypes` is called exactly ONCE on the full high-volume
frame (every row, every week) before any split, and the same fitted dtypes dict is
reused for the Fold 1 train subset, the Fold 1 validation subset, the full-history
refit, and every recursive forecast step. Never refit per fold (plan §13 failure mode).

Recursive multi-step forecasting (step 9): this is a single *global* model over all
products, and its lag/rolling features (lag_1/2/4/12, rollmean_4/12, rollstd_4/12)
depend on prior weeks' Sales_Qty. A 12-week-ahead forecast therefore cannot be
produced in one shot -- it is generated recursively, one week at a time:
  1. Seed a (n_products x 12) matrix with each product's last 12 *actual* observed
     Sales_Qty values (left-padded with NaN if fewer than 12 weeks of history exist).
  2. For forecast step h = 1..12: derive lag_1/2/4/12 and rollmean/rollstd_4/12
     directly from that matrix (matching the exact pandas `.shift(1).rolling(w,
     min_periods=...)` semantics used to build `featured.parquet`), derive calendar
     features from the forecast week's actual date, and carry forward each
     product's static features (ProductID/Region/Country/Sales Channel/Brand
     Category/Product Range -- verified constant per product -- plus dominant_dow,
     pct_weekend_orders, zero_streak_length, cumulative_zero_weeks, unit_price,
     which are NOT re-derivable for unobserved future weeks and are documented here
     as frozen at their last-observed value; a documented simplification, not a bug).
  3. Predict week h, clip to >= 0, append the prediction to the matrix (drop the
     oldest column), and repeat for week h+1.
This is implemented once and shared by LightGBM and XGBoost (`_recursive_forecast`).
The seasonal-naive baseline forecast does NOT need this recursion: because the
horizon (12 weeks) exactly equals the seasonal lag (12), "the value 12 weeks before
forecast week h" always falls inside the already-observed history window for every
h in 1..12, so it is read directly off the seeded matrix.
"""
from __future__ import annotations

import time
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.logging_utils import get_logger, log_metrics, stage
from src.metrics import mae, rmse
from src.models.categorical import apply_categorical_dtypes, fit_categorical_dtypes

logger = get_logger(__name__)

FEATURED_PATH = Path("data/processed/featured.parquet")
TIERS_PATH = Path("data/processed/tiers.parquet")
OUTPUT_DIR = Path("outputs/models/high_volume")

CATEGORICAL_COLUMNS = [
    "ProductID",
    "Region",
    "Country",
    "Sales Channel",
    "Brand Category",
    "Product Range",
]
BOOL_COLUMNS = ["is_year_end", "is_lk_holiday_week", "returns_exceeded_sales"]
NON_FEATURE_COLUMNS = ["WeekStart", "Sales_Qty", "Sales_USD", "returns_exceeded_sales", "transaction_count"]

FEATURE_COLUMNS = [
    "ProductID",
    "Region",
    "Country",
    "Sales Channel",
    "Brand Category",
    "Product Range",
    "Sales_Qty_lag_1",
    "Sales_Qty_lag_2",
    "Sales_Qty_lag_4",
    "Sales_Qty_lag_12",
    "Sales_Qty_rollmean_4",
    "Sales_Qty_rollstd_4",
    "Sales_Qty_rollmean_12",
    "Sales_Qty_rollstd_12",
    "dominant_dow",
    "pct_weekend_orders",
    "month",
    "week_of_month",
    "quarter",
    "week_of_year",
    "is_year_end",
    "is_lk_holiday_week",
    "zero_streak_length",
    "cumulative_zero_weeks",
    "unit_price",
]

TRAIN_CUTOFF = pd.Timestamp("2025-06-23")
VALID_START = pd.Timestamp("2025-06-30")
VALID_END = pd.Timestamp("2025-09-15")
FULL_HISTORY_END = pd.Timestamp("2025-09-15")
FORECAST_HORIZON = 12
LAG_WINDOW = 12  # matches Sales_Qty_lag_12 / rollmean_12 / rollstd_12
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 2000
RANDOM_SEED = 42

_LK_HOLIDAY_CACHE: dict[int, set] = {}


def _lk_holiday_days(year: int) -> set:
    if year not in _LK_HOLIDAY_CACHE:
        _LK_HOLIDAY_CACHE[year] = set(pd.to_datetime(list(holidays.country_holidays("LK", years=[year]).keys())))
    return _LK_HOLIDAY_CACHE[year]


def _is_holiday_week(week_start: pd.Timestamp) -> bool:
    week_days = pd.date_range(week_start, periods=7, freq="D")
    years = {int(d.year) for d in week_days}
    holiday_days: set = set()
    for year in years:
        holiday_days |= _lk_holiday_days(year)
    return any(day in holiday_days for day in week_days)


def load_high_volume_frame() -> pd.DataFrame:
    """Load `featured.parquet` filtered to high-volume ProductIDs, sorted by product/week."""
    featured = pd.read_parquet(FEATURED_PATH)
    tiers = pd.read_parquet(TIERS_PATH)
    hv_ids = set(tiers.loc[tiers["tier"] == "high_volume", "ProductID"])
    df = featured[featured["ProductID"].isin(hv_ids)].copy()
    df = df.sort_values(["ProductID", "WeekStart"]).reset_index(drop=True)
    for col in BOOL_COLUMNS:
        df[col] = df[col].astype(int)
    df["dominant_dow"] = df["dominant_dow"].astype("float64")
    return df


def fold1_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered Fold 1 split. No shuffling, no `train_test_split` call."""
    train = df.loc[df["WeekStart"] <= TRAIN_CUTOFF].copy()
    valid = df.loc[(df["WeekStart"] >= VALID_START) & (df["WeekStart"] <= VALID_END)].copy()
    return train, valid


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df["Sales_Qty"].astype(float)
    return X, y


def seasonal_naive_predict(df: pd.DataFrame) -> pd.Series:
    """Real lag-12 seasonal naive, falling back to lag-1, falling back to 0."""
    pred = df["Sales_Qty_lag_12"].copy()
    pred = pred.fillna(df["Sales_Qty_lag_1"])
    pred = pred.fillna(0.0)
    return pred


def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
) -> tuple[lgb.Booster, dict, float]:
    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False
    )
    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=train_set,
        categorical_feature=CATEGORICAL_COLUMNS,
        free_raw_data=False,
    )
    eval_history: dict = {}
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "seed": RANDOM_SEED,
    }
    start = time.perf_counter()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.record_evaluation(eval_history),
            lgb.log_evaluation(period=100),
        ],
    )
    train_seconds = time.perf_counter() - start
    return booster, eval_history, train_seconds


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
) -> tuple[XGBRegressor, float]:
    model = XGBRegressor(
        tree_method="hist",
        enable_categorical=True,
        n_estimators=MAX_BOOST_ROUNDS,
        learning_rate=0.05,
        max_depth=8,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="rmse",
        random_state=RANDOM_SEED,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    train_seconds = time.perf_counter() - start
    return model, train_seconds


def _rolling_std_min_periods(window_vals: np.ndarray, min_periods: int = 2, ddof: int = 1) -> np.ndarray:
    """Mimic pandas `.rolling(window, min_periods=min_periods).std(ddof=ddof)` on a dense matrix."""
    count = np.sum(~np.isnan(window_vals), axis=1)
    with np.errstate(invalid="ignore"):
        std = np.nanstd(window_vals, axis=1, ddof=ddof)
    return np.where(count < min_periods, np.nan, std)


def build_recursive_state(
    df: pd.DataFrame, window: int = LAG_WINDOW
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Seed the last `window` actual Sales_Qty values per product plus a static-feature template row.

    Returns (product_ids, history matrix shape (n_products, window), static_df with one
    row per product carrying the last observed row's static/non-recomputable features).
    """
    df_sorted = df.sort_values(["ProductID", "WeekStart"])
    grouped = df_sorted.groupby("ProductID", sort=True)
    product_ids = np.array(list(grouped.groups.keys()))
    n = len(product_ids)
    history = np.full((n, window), np.nan)
    static_rows = []
    for i, pid in enumerate(product_ids):
        g = grouped.get_group(pid)
        qty = g["Sales_Qty"].to_numpy(dtype=float)
        tail = qty[-window:]
        history[i, window - len(tail):] = tail
        static_rows.append(g.iloc[-1])
    static_df = pd.DataFrame(static_rows).reset_index(drop=True)
    static_df["ProductID"] = product_ids
    return product_ids, history, static_df


def _build_step_features(
    static_df: pd.DataFrame, history: np.ndarray, week_start: pd.Timestamp
) -> pd.DataFrame:
    step_df = static_df.copy()
    step_df["Sales_Qty_lag_1"] = history[:, -1]
    step_df["Sales_Qty_lag_2"] = history[:, -2]
    step_df["Sales_Qty_lag_4"] = history[:, -4]
    step_df["Sales_Qty_lag_12"] = history[:, -12]
    step_df["Sales_Qty_rollmean_4"] = np.nanmean(history[:, -4:], axis=1)
    step_df["Sales_Qty_rollstd_4"] = _rolling_std_min_periods(history[:, -4:])
    step_df["Sales_Qty_rollmean_12"] = np.nanmean(history, axis=1)
    step_df["Sales_Qty_rollstd_12"] = _rolling_std_min_periods(history)
    step_df["month"] = week_start.month
    step_df["week_of_month"] = ((week_start.day - 1) // 7) + 1
    step_df["quarter"] = week_start.quarter
    step_df["week_of_year"] = int(week_start.isocalendar()[1])
    step_df["is_year_end"] = int(week_start.month in (11, 12))
    step_df["is_lk_holiday_week"] = int(_is_holiday_week(week_start))
    return step_df


def recursive_forecast(
    predict_fn, product_ids: np.ndarray, history: np.ndarray, static_df: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Single-step recursive 12-week-ahead forecast shared by LightGBM and XGBoost."""
    history = history.copy()
    rows = []
    for h in range(1, FORECAST_HORIZON + 1):
        week_start = FULL_HISTORY_END + pd.Timedelta(weeks=h)
        step_df = _build_step_features(static_df, history, week_start)
        X_step = step_df[FEATURE_COLUMNS].copy()
        pred = np.asarray(predict_fn(X_step), dtype=float)
        pred = np.clip(pred, 0.0, None)
        rows.append(
            pd.DataFrame(
                {
                    "ProductID": product_ids,
                    "forecast_week": h,
                    "WeekStart": week_start,
                    "predicted_Sales_Qty": pred,
                    "model": model_name,
                }
            )
        )
        history = np.column_stack([history[:, 1:], pred])
    return pd.concat(rows, ignore_index=True)


def seasonal_naive_forecast(product_ids: np.ndarray, history: np.ndarray) -> pd.DataFrame:
    """Seasonal-naive final forecast: horizon (12) == season length (12), so the value 12
    weeks before every forecast week is already inside the seeded history window -- no
    recursion needed. Falls back to the last observed value, then 0, if a lag is NaN.
    """
    rows = []
    last_observed = history[:, -1]
    for h in range(1, FORECAST_HORIZON + 1):
        week_start = FULL_HISTORY_END + pd.Timedelta(weeks=h)
        pred = history[:, h - 1].copy()
        nan_mask = np.isnan(pred)
        pred[nan_mask] = last_observed[nan_mask]
        pred = np.nan_to_num(pred, nan=0.0)
        pred = np.clip(pred, 0.0, None)
        rows.append(
            pd.DataFrame(
                {
                    "ProductID": product_ids,
                    "forecast_week": h,
                    "WeekStart": week_start,
                    "predicted_Sales_Qty": pred,
                    "model": "seasonal_naive",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict[str, object] = {}

    with stage(logger, "Load high-volume frame"):
        df = load_high_volume_frame()
        n_products = df["ProductID"].nunique()
        log_metrics(logger, {"rows": len(df), "products": n_products})

    with stage(logger, "Fit categorical dtypes (once, full universe)"):
        categorical_dtypes = fit_categorical_dtypes(df, CATEGORICAL_COLUMNS)
        df = apply_categorical_dtypes(df, categorical_dtypes)

    with stage(logger, "Fold 1 split (time-ordered, no shuffle)"):
        train_df, valid_df = fold1_split(df)
        log_metrics(
            logger,
            {
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "train_end": str(TRAIN_CUTOFF.date()),
                "valid_start": str(VALID_START.date()),
                "valid_end": str(VALID_END.date()),
            },
        )
        train_df = apply_categorical_dtypes(train_df, categorical_dtypes)
        valid_df = apply_categorical_dtypes(valid_df, categorical_dtypes)
        X_train, y_train = build_xy(train_df)
        X_valid, y_valid = build_xy(valid_df)

    with stage(logger, "Train LightGBM (Fold 1)"):
        lgb_booster, lgb_eval_history, lgb_train_seconds = train_lightgbm(X_train, y_train, X_valid, y_valid)
        lgb_pred_valid = lgb_booster.predict(X_valid, num_iteration=lgb_booster.best_iteration)
        log_metrics(
            logger,
            {
                "best_iteration": lgb_booster.best_iteration,
                "train_seconds": lgb_train_seconds,
                "valid_rmse": rmse(y_valid, lgb_pred_valid),
                "valid_mae": mae(y_valid, lgb_pred_valid),
            },
        )
        timings["lightgbm_train_seconds"] = lgb_train_seconds
        timings["lightgbm_best_iteration"] = int(lgb_booster.best_iteration)

    with stage(logger, "Train XGBoost (Fold 1)"):
        xgb_model, xgb_train_seconds = train_xgboost(X_train, y_train, X_valid, y_valid)
        xgb_pred_valid = xgb_model.predict(X_valid)
        xgb_best_iteration = xgb_model.best_iteration
        log_metrics(
            logger,
            {
                "best_iteration": xgb_best_iteration,
                "train_seconds": xgb_train_seconds,
                "valid_rmse": rmse(y_valid, xgb_pred_valid),
                "valid_mae": mae(y_valid, xgb_pred_valid),
            },
        )
        timings["xgboost_train_seconds"] = xgb_train_seconds
        timings["xgboost_best_iteration"] = int(xgb_best_iteration)

    with stage(logger, "Seasonal-naive baseline (Fold 1 validation)"):
        naive_pred_valid = seasonal_naive_predict(valid_df)
        log_metrics(
            logger,
            {
                "valid_rmse": rmse(y_valid, naive_pred_valid),
                "valid_mae": mae(y_valid, naive_pred_valid),
            },
        )

    with stage(logger, "Write validation predictions"):
        valid_key = valid_df[["ProductID", "WeekStart"]].reset_index(drop=True)
        val_out = pd.concat(
            [
                valid_key.assign(y_true=y_valid.to_numpy(), model="lightgbm", y_pred=lgb_pred_valid),
                valid_key.assign(y_true=y_valid.to_numpy(), model="xgboost", y_pred=xgb_pred_valid),
                valid_key.assign(
                    y_true=y_valid.to_numpy(), model="seasonal_naive", y_pred=naive_pred_valid.to_numpy()
                ),
            ],
            ignore_index=True,
        )
        val_path = OUTPUT_DIR / "validation_predictions.parquet"
        val_out.to_parquet(val_path, index=False)
        log_metrics(logger, {"rows": len(val_out), "path": str(val_path)})

    with stage(logger, "Refit on full history through 2025-09-15"):
        X_full, y_full = build_xy(df)
        start = time.perf_counter()
        full_train_set = lgb.Dataset(
            X_full, label=y_full, categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False
        )
        lgb_final = lgb.train(
            {
                "objective": "regression",
                "metric": "rmse",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "min_data_in_leaf": 20,
                "verbose": -1,
                "seed": RANDOM_SEED,
            },
            full_train_set,
            num_boost_round=max(lgb_booster.best_iteration, 1),
        )
        lgb_final_train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        xgb_final = XGBRegressor(
            tree_method="hist",
            enable_categorical=True,
            n_estimators=max(xgb_best_iteration + 1, 1),
            learning_rate=0.05,
            max_depth=8,
            random_state=RANDOM_SEED,
        )
        xgb_final.fit(X_full, y_full)
        xgb_final_train_seconds = time.perf_counter() - start

        log_metrics(
            logger,
            {
                "lightgbm_final_train_seconds": lgb_final_train_seconds,
                "lightgbm_num_boost_round": max(lgb_booster.best_iteration, 1),
                "xgboost_final_train_seconds": xgb_final_train_seconds,
                "xgboost_n_estimators": max(xgb_best_iteration + 1, 1),
            },
        )
        timings["lightgbm_final_train_seconds"] = lgb_final_train_seconds
        timings["xgboost_final_train_seconds"] = xgb_final_train_seconds

    with stage(logger, "Recursive 12-week forecast (all 3 methods)"):
        product_ids, history, static_df = build_recursive_state(df, window=LAG_WINDOW)
        static_df = apply_categorical_dtypes(static_df, categorical_dtypes)

        start = time.perf_counter()
        lgb_forecast = recursive_forecast(
            lambda X: lgb_final.predict(X), product_ids, history, static_df, "lightgbm"
        )
        lgb_inference_seconds = time.perf_counter() - start
        timings["lightgbm_inference_seconds"] = lgb_inference_seconds

        start = time.perf_counter()
        xgb_forecast = recursive_forecast(
            lambda X: xgb_final.predict(X), product_ids, history, static_df, "xgboost"
        )
        xgb_inference_seconds = time.perf_counter() - start
        timings["xgboost_inference_seconds"] = xgb_inference_seconds

        start = time.perf_counter()
        naive_forecast = seasonal_naive_forecast(product_ids, history)
        naive_inference_seconds = time.perf_counter() - start
        timings["seasonal_naive_inference_seconds"] = naive_inference_seconds

        log_metrics(
            logger,
            {
                "lightgbm_inference_seconds": lgb_inference_seconds,
                "xgboost_inference_seconds": xgb_inference_seconds,
                "seasonal_naive_inference_seconds": naive_inference_seconds,
            },
        )

    with stage(logger, "Write final forecast (primary model: lightgbm)"):
        forecast_out = pd.concat([lgb_forecast, xgb_forecast, naive_forecast], ignore_index=True)
        forecast_out = forecast_out[
            ["ProductID", "forecast_week", "WeekStart", "predicted_Sales_Qty", "model"]
        ]
        forecast_path = OUTPUT_DIR / "final_forecast.parquet"
        forecast_out.to_parquet(forecast_path, index=False)
        expected_rows = n_products * FORECAST_HORIZON * 3
        log_metrics(
            logger,
            {
                "rows": len(forecast_out),
                "expected_rows": expected_rows,
                "products": n_products,
                "primary_model": "lightgbm",
                "path": str(forecast_path),
            },
        )
        assert len(forecast_out) == expected_rows, "final forecast row count mismatch"
        assert (forecast_out["predicted_Sales_Qty"] >= 0).all(), "negative prediction found"

    with stage(logger, "Write timing summary"):
        timing_path = OUTPUT_DIR / "timing.json"
        pd.Series(timings).to_json(timing_path, indent=2)
        log_metrics(logger, {"path": str(timing_path)})


if __name__ == "__main__":
    main()
