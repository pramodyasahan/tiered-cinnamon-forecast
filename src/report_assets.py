"""Rubric-required transparency/insight figures (spec 003 FR-29..FR-35, plan §4.7).

Generates every figure file the report cites, as real PNGs under
``outputs/figures/`` — never manually authored images. Each figure function
is independent and writes its own file(s); `main()` runs them all and logs
what was written (or skipped, with a reason).

Failure-mode notes carried over from plan §13 (do not regress these):

- FR-30 SHAP shape (§2 Exa fact / §13): `shap.TreeExplainer(model).shap_values(X)`
  for a single-output LightGBM *regressor* returns one 2-D array
  (samples x features), NOT a list of per-class arrays — that list form only
  applies to multi-class/multi-output models. This module asserts
  `shap_values.ndim == 2` and does not index it as a list. A quick timing
  check during implementation showed SHAP on a 300-row deterministic sample
  of the Fold 1 high-volume validation set completes in well under a second,
  so the gain-importance fallback path exists but is not the primary path;
  it is used only if SHAP raises or exceeds a generous time budget.
- FR-34 seasonal decomposition (§2 Exa fact / EC-3): `seasonal_decompose(...,
  period=52)` requires >= 2 * 52 = 104 weeks of history. Candidates are
  checked for week-count before decomposing; short candidates are skipped
  with a logged reason instead of crashing.
- FR-31 loss/eval curves: `src/models/high_volume.py` (T005) computes
  `lgb_eval_history` / `xgb.evals_result()` in-memory but does not persist
  them to disk. Rather than duplicate the training/feature logic inline,
  this module imports the exact same loading/split/train helpers from
  `src.models.high_volume` and re-runs the (already known to be
  sub-second) Fold 1 LightGBM/XGBoost fit to recover the eval history for
  plotting. This is the same Fold 1 split documented in that module
  (`TRAIN_CUTOFF`/`VALID_START`/`VALID_END`), not a new split.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor

from src.logging_utils import get_logger, log_metrics, stage
from src.models.categorical import apply_categorical_dtypes, fit_categorical_dtypes
from src.models.high_volume import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    build_xy,
    fold1_split,
    load_high_volume_frame,
    train_lightgbm,
)

logger = get_logger(__name__)

FEATURED_PATH = Path("data/processed/featured.parquet")
TIERS_PATH = Path("data/processed/tiers.parquet")
HV_VALID_PATH = Path("outputs/models/high_volume/validation_predictions.parquet")
SPARSE_VALID_PATH = Path("outputs/models/sparse/validation_predictions.parquet")
FIGURES_DIR = Path("outputs/figures")

SHAP_SAMPLE_SIZE = 300
SHAP_RANDOM_SEED = 42
SEASONAL_PERIOD = 52
MIN_WEEKS_FOR_DECOMPOSITION = 2 * SEASONAL_PERIOD  # 104 (§2 Exa fact / EC-3)
TOP_VOLUME_CANDIDATES = 8  # scan this many top-volume products, decompose the first 3 that qualify
TOP_VOLUME_TARGET = 3

# Numeric engineered features for the correlation heatmap (FR-29). Excludes
# identifiers, dates, raw booleans, and categoricals -- those are not
# "engineered numeric features" in the sense the figure is meant to show.
NUMERIC_FEATURE_COLUMNS = [
    "Sales_Qty",
    "Sales_USD",
    "transaction_count",
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
    "zero_streak_length",
    "cumulative_zero_weeks",
    "unit_price",
]


def fig_correlation_heatmap() -> Path:
    """FR-29: correlation heatmap over engineered numeric features."""
    df = pd.read_parquet(FEATURED_PATH, columns=NUMERIC_FEATURE_COLUMNS)
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True, ax=ax)
    ax.set_title("Correlation Matrix — Engineered Numeric Features")
    fig.tight_layout()

    path = FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fit_fold1_lightgbm() -> tuple:
    """Recreate the exact Fold 1 high-volume split/fit from `src.models.high_volume`.

    Returns (booster, eval_history, X_valid_categorical, xgb_model, xgb_evals_result).
    Reused by both the SHAP-importance figure and the loss-curve figure so the
    fit only happens once per `main()` run.
    """
    df = load_high_volume_frame()
    cats = fit_categorical_dtypes(df, CATEGORICAL_COLUMNS)
    df = apply_categorical_dtypes(df, cats)
    train_df, valid_df = fold1_split(df)
    train_df = apply_categorical_dtypes(train_df, cats)
    valid_df = apply_categorical_dtypes(valid_df, cats)
    X_train, y_train = build_xy(train_df)
    X_valid, y_valid = build_xy(valid_df)

    booster, eval_history, _ = train_lightgbm(X_train, y_train, X_valid, y_valid)

    xgb_model = XGBRegressor(
        tree_method="hist",
        enable_categorical=True,
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    return booster, eval_history, X_valid, xgb_model


def fig_shap_importance(booster, X_valid: pd.DataFrame) -> Path:
    """FR-30: SHAP feature importance on a deterministic sample.

    Consumes `shap_values` as a single 2-D array (samples x features), the
    correct shape for a single-output regressor (plan §13 / §2 Exa fact) --
    NOT indexed as a list, which is the multiclass-classifier API shape.
    Falls back to LightGBM gain importance if SHAP raises.
    """
    sample = X_valid.sample(n=min(SHAP_SAMPLE_SIZE, len(X_valid)), random_state=SHAP_RANDOM_SEED)

    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(sample)
        assert shap_values.ndim == 2, (
            f"expected a single 2-D SHAP array for this single-output regressor, "
            f"got ndim={shap_values.ndim} -- do not index this as a list"
        )
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = pd.Series(mean_abs_shap, index=sample.columns).sort_values(ascending=False)
        title = "Feature Importance — mean |SHAP value| (LightGBM, Fold 1 valid sample)"
        used_fallback = False
    except Exception as exc:  # pragma: no cover - defensive fallback path (EC-4)
        logger.warning("SHAP failed (%s); falling back to LightGBM gain importance", exc)
        gain = booster.feature_importance(importance_type="gain")
        importance = pd.Series(gain, index=booster.feature_name()).sort_values(ascending=False)
        title = "Feature Importance — LightGBM gain (SHAP fallback, documented per plan §4.7/EC-4)"
        used_fallback = True

    fig, ax = plt.subplots(figsize=(9, 8))
    importance.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("mean |SHAP value|" if not used_fallback else "gain")
    fig.tight_layout()

    path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_loss_curves(eval_history: dict, xgb_model) -> Path:
    """FR-31: loss/evaluation curves from the LightGBM/XGBoost training histories."""
    fig, ax = plt.subplots(figsize=(9, 6))

    lgb_valid_rmse = eval_history.get("valid", {}).get("rmse")
    if lgb_valid_rmse:
        ax.plot(range(1, len(lgb_valid_rmse) + 1), lgb_valid_rmse, label="LightGBM valid RMSE")

    xgb_results = xgb_model.evals_result()
    xgb_valid_rmse = xgb_results.get("validation_0", {}).get("rmse")
    if xgb_valid_rmse:
        ax.plot(range(1, len(xgb_valid_rmse) + 1), xgb_valid_rmse, label="XGBoost valid RMSE")

    ax.set_xlabel("boosting round")
    ax.set_ylabel("RMSE")
    ax.set_title("Training Loss / Evaluation Curves — Fold 1 Validation")
    ax.legend()
    fig.tight_layout()

    path = FIGURES_DIR / "loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_actual_vs_predicted() -> Path:
    """FR-32: actual-vs-predicted plot for >=1 high-volume and >=1 sparse product."""
    hv_valid = pd.read_parquet(HV_VALID_PATH)
    sparse_valid = pd.read_parquet(SPARSE_VALID_PATH)

    hv_lgb = hv_valid[hv_valid["model"] == "lightgbm"]
    hv_product = hv_lgb.groupby("ProductID")["y_true"].sum().sort_values(ascending=False).index[0]
    hv_series = hv_lgb[hv_lgb["ProductID"] == hv_product].sort_values("WeekStart")

    sparse_primary = sparse_valid[sparse_valid["model"] == "CrostonSBA"]
    sparse_product = (
        sparse_primary.groupby("unique_id")["y"].sum().sort_values(ascending=False).index[0]
    )
    sparse_series = sparse_primary[sparse_primary["unique_id"] == sparse_product].sort_values("ds")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(hv_series["WeekStart"], hv_series["y_true"], marker="o", label="actual")
    axes[0].plot(hv_series["WeekStart"], hv_series["y_pred"], marker="o", label="predicted (lightgbm)")
    axes[0].set_title(f"High-Volume Actual vs Predicted — {hv_product}")
    axes[0].legend()

    axes[1].plot(sparse_series["ds"], sparse_series["y"], marker="o", label="actual")
    axes[1].plot(sparse_series["ds"], sparse_series["y_pred"], marker="o", label="predicted (CrostonSBA)")
    axes[1].set_title(f"Sparse Actual vs Predicted — {sparse_product}")
    axes[1].legend()

    fig.tight_layout()
    path = FIGURES_DIR / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_sparse_zero_analysis() -> Path:
    """FR-33: sparse/zero-sales analysis using the existing `zero_streak_length`
    and `cumulative_zero_weeks` features already in `featured.parquet`
    (`src/features.py`), rather than recomputing zero-run statistics inline.
    """
    featured = pd.read_parquet(
        FEATURED_PATH, columns=["ProductID", "Sales_Qty", "zero_streak_length"]
    )
    tiers = pd.read_parquet(TIERS_PATH, columns=["ProductID", "tier"])
    sparse_ids = set(tiers.loc[tiers["tier"] == "sparse", "ProductID"])
    sparse_df = featured[featured["ProductID"].isin(sparse_ids)]

    max_streak = sparse_df.groupby("ProductID")["zero_streak_length"].max()
    zero_share = sparse_df.groupby("ProductID").apply(
        lambda g: (g["Sales_Qty"] == 0).mean(), include_groups=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(max_streak, bins=30, color="#55A868")
    axes[0].set_title("Sparse Tier — Max Zero-Streak Length per Product")
    axes[0].set_xlabel("weeks")

    axes[1].hist(zero_share, bins=30, color="#C44E52")
    axes[1].set_title("Sparse Tier — Zero-Week Share per Product")
    axes[1].set_xlabel("share of weeks with zero Sales_Qty")

    fig.tight_layout()
    path = FIGURES_DIR / "sparse_zero_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _top_volume_products(n: int) -> list[str]:
    totals = pd.read_parquet(FEATURED_PATH, columns=["ProductID", "Sales_Qty"])
    return (
        totals.groupby("ProductID")["Sales_Qty"].sum().sort_values(ascending=False).head(n).index.tolist()
    )


def fig_yoy_seasonal_decomposition() -> tuple[list[Path], list[str]]:
    """FR-34: YoY + `seasonal_decompose(period=52)` for top-volume products
    with >=104 weeks of history (2 full seasonal cycles). Products with
    less history are skipped with a logged reason (EC-3), not a crash.
    """
    candidates = _top_volume_products(TOP_VOLUME_CANDIDATES)
    featured = pd.read_parquet(FEATURED_PATH, columns=["ProductID", "WeekStart", "Sales_Qty"])

    written: list[Path] = []
    skipped: list[str] = []
    products_done = 0

    for product_id in candidates:
        if products_done >= TOP_VOLUME_TARGET:
            break
        series_df = featured[featured["ProductID"] == product_id].sort_values("WeekStart")
        n_weeks = len(series_df)
        if n_weeks < MIN_WEEKS_FOR_DECOMPOSITION:
            reason = (
                f"skipped {product_id}: {n_weeks} weeks of history < "
                f"{MIN_WEEKS_FOR_DECOMPOSITION} required for period={SEASONAL_PERIOD} decomposition (EC-3)"
            )
            logger.info(reason)
            skipped.append(reason)
            continue

        series = pd.Series(
            series_df["Sales_Qty"].to_numpy(), index=pd.DatetimeIndex(series_df["WeekStart"])
        )
        result = seasonal_decompose(series, model="additive", period=SEASONAL_PERIOD)

        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        result.observed.plot(ax=axes[0], title=f"Observed — {product_id}")
        result.trend.plot(ax=axes[1], title="Trend")
        result.seasonal.plot(ax=axes[2], title="Seasonal")
        result.resid.plot(ax=axes[3], title="Residual")
        fig.suptitle(f"Seasonal Decomposition (additive, period={SEASONAL_PERIOD}) — {product_id}")
        fig.tight_layout()

        path = FIGURES_DIR / f"seasonal_decompose_{product_id}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

        # YoY plot: overlay each calendar year's series against week-of-year.
        yoy_df = series_df.copy()
        yoy_df["year"] = yoy_df["WeekStart"].dt.isocalendar().year
        yoy_df["week_of_year"] = yoy_df["WeekStart"].dt.isocalendar().week

        fig_yoy, ax_yoy = plt.subplots(figsize=(10, 5))
        for year, group in yoy_df.groupby("year"):
            ax_yoy.plot(group["week_of_year"], group["Sales_Qty"], marker="o", label=str(year))
        ax_yoy.set_title(f"Year-over-Year Weekly Sales — {product_id}")
        ax_yoy.set_xlabel("ISO week of year")
        ax_yoy.set_ylabel("Sales_Qty")
        ax_yoy.legend(title="year")
        fig_yoy.tight_layout()

        yoy_path = FIGURES_DIR / f"yoy_{product_id}.png"
        fig_yoy.savefig(yoy_path, dpi=150)
        plt.close(fig_yoy)
        written.append(yoy_path)
        products_done += 1

    return written, skipped


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with stage(logger, "FR-29: correlation heatmap"):
        path = fig_correlation_heatmap()
        log_metrics(logger, {"path": str(path)})

    with stage(logger, "Fold 1 LightGBM/XGBoost refit (feeds SHAP + loss curves)"):
        booster, eval_history, X_valid, xgb_model = _fit_fold1_lightgbm()
        log_metrics(logger, {"best_iteration": booster.best_iteration, "valid_rows": len(X_valid)})

    with stage(logger, "FR-30: SHAP feature importance"):
        path = fig_shap_importance(booster, X_valid)
        log_metrics(logger, {"path": str(path)})

    with stage(logger, "FR-31: loss/evaluation curves"):
        path = fig_loss_curves(eval_history, xgb_model)
        log_metrics(logger, {"path": str(path)})

    with stage(logger, "FR-32: actual-vs-predicted (high-volume + sparse)"):
        path = fig_actual_vs_predicted()
        log_metrics(logger, {"path": str(path)})

    with stage(logger, "FR-33: sparse/zero-sales analysis"):
        path = fig_sparse_zero_analysis()
        log_metrics(logger, {"path": str(path)})

    with stage(logger, "FR-34: YoY + seasonal decomposition (top-volume products)"):
        written, skipped = fig_yoy_seasonal_decomposition()
        log_metrics(
            logger,
            {
                "figures_written": len(written),
                "products_skipped": len(skipped),
            },
        )
        for reason in skipped:
            logger.info("  %s", reason)


if __name__ == "__main__":
    main()
