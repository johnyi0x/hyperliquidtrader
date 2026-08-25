"""Offline crowd backtest orchestration (research data + Numba PnL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bt_tune import DEFAULT_GRID, TuneResult, load_tuned, save_tuned, tune_all
from .research_load import ResearchDataset, build_dataset

# Backward-compatible names used by tests / imports.
BACKTEST_PARAM_KEYS = DEFAULT_GRID.keys()


def run_backtest_suite(
    base_cfg: Any,
    data_dir: Path,
    *,
    max_days: int = 7,
    max_combos: int = 120,
    apply_path: Path | None = None,
    strategies: list[str] | None = None,
    two_stage: bool = True,
    top_k: int = 2,
    coarse_combos: int | None = None,
) -> tuple[ResearchDataset | None, list[TuneResult]]:
    ds, results = tune_all(
        base_cfg,
        data_dir,
        max_days=max_days,
        strategies=strategies,
        max_combos_per_strategy=max_combos,
        state_path=data_dir / "pmf_state.json",
        two_stage=two_stage,
        top_k=top_k,
        coarse_combos_per_strategy=coarse_combos,
    )
    if apply_path is not None and results:
        save_tuned(apply_path, results[0], dataset=ds)
    return ds, results


__all__ = [
    "BACKTEST_PARAM_KEYS",
    "DEFAULT_GRID",
    "ResearchDataset",
    "TuneResult",
    "build_dataset",
    "load_tuned",
    "run_backtest_suite",
    "save_tuned",
    "tune_all",
]
