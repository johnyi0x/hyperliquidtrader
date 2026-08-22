"""Tune speed profiles: fast (default) vs full (current thorough grids)."""

from __future__ import annotations

from typing import Any


def normalize_tune_profile(raw: str | None) -> str:
    p = str(raw or "fast").strip().lower()
    if p in ("full", "thorough", "slow", "max"):
        return "full"
    return "fast"


def _pick_balances(
    balance_grid: tuple[float, ...] | list[float],
    targets: tuple[float, ...],
) -> tuple[float, ...]:
    """Keep values from the user grid nearest to each target (unique)."""
    grid = sorted({float(x) for x in (balance_grid or ()) if float(x) > 0})
    if not grid:
        return targets
    picked: list[float] = []
    used: set[float] = set()
    for t in targets:
        best = min(grid, key=lambda g: (abs(g - t), g))
        if best not in used:
            used.add(best)
            picked.append(best)
    return tuple(picked) if picked else (grid[len(grid) // 2],)


def resolve_tune_profile(
    profile: str | None,
    *,
    balance_grid: tuple[float, ...] | list[float],
    screen_top_n: int,
) -> dict[str, Any]:
    """
    Returns effective tune knobs for the selected profile.

    fast  — same entry strategies screened; much smaller refine/consensus grids
    full  — current thorough search (unchanged quality/coverage)
    """
    mode = normalize_tune_profile(profile)
    if mode == "full":
        return {
            "profile": "full",
            "screen_top_n": max(1, int(screen_top_n)),
            "balance_grid": tuple(float(x) for x in balance_grid),
            "refine_profile": "full",
            "mtf_profile": "full",
        }
    return {
        "profile": "fast",
        "screen_top_n": max(1, min(6, int(screen_top_n))),
        "balance_grid": _pick_balances(balance_grid, (15.0, 30.0, 50.0, 75.0)),
        "refine_profile": "fast",
        "mtf_profile": "fast",
    }
