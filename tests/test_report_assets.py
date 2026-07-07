"""Regression test for spec 003 T011 — report-figure existence (T-7).

T-7: after running the generator, every file named in spec FR-29..FR-34 must
exist under `outputs/figures/`. This test invokes `src.report_assets.main()`
directly (same entrypoint as `uv run python -m src.report_assets`) and then
checks the filesystem, rather than re-deriving the figure logic inline.
"""
from __future__ import annotations

from pathlib import Path

from src.report_assets import FIGURES_DIR, main


def test_all_fr29_to_fr34_figures_exist_after_generation():
    main()

    assert (FIGURES_DIR / "correlation_heatmap.png").exists()  # FR-29
    assert (FIGURES_DIR / "feature_importance.png").exists()  # FR-30
    assert (FIGURES_DIR / "loss_curves.png").exists()  # FR-31
    assert (FIGURES_DIR / "actual_vs_predicted.png").exists()  # FR-32
    assert (FIGURES_DIR / "sparse_zero_analysis.png").exists()  # FR-33

    # FR-34: at least one YoY + one seasonal-decomposition figure for a
    # top-volume product with >=104 weeks of history (EC-3 skip is allowed
    # to reduce the count but not to eliminate it entirely).
    seasonal_figures = list(FIGURES_DIR.glob("seasonal_decompose_*.png"))
    yoy_figures = list(FIGURES_DIR.glob("yoy_*.png"))
    assert len(seasonal_figures) >= 1
    assert len(yoy_figures) >= 1
