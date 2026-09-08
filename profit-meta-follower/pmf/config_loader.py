"""Apply PMF_PROFILE overrides onto the config module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config_profiles import PROFILES


def active_profile_name() -> str:
    return str(os.environ.get("PMF_PROFILE", "local") or "local").strip().lower()


def _tuned_params_path(profile: str) -> Path | None:
    root = Path(__file__).resolve().parent.parent
    override = os.environ.get("PMF_DATA_DIR", "").strip()
    if override:
        base = Path(override)
        if not base.is_absolute():
            base = root / base
    else:
        base = root / f"data-{profile}"
    # Committed file (Railway / git deploy).
    candidates = [
        root / "cloud_tuned.json",
        base / "tuned_cloud.json",
        base / "backtest_latest.json",
        root / "data-local" / "backtest_latest.json",
        root / "data-local" / "tuned_cloud.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _merge_tuned(overrides: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile != "cloud":
        return overrides
    path = _tuned_params_path(profile)
    if path is None:
        return overrides
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        params = raw.get("params") if isinstance(raw, dict) else None
        if isinstance(params, dict) and params:
            merged = dict(overrides)
            merged.update(params)
            merged["TUNED_STRATEGY"] = str(raw.get("strategy") or "")
            merged["TUNED_AT"] = float(raw.get("saved_at") or 0)
            merged["BACKTEST_LIVE_STRATEGY"] = merged["TUNED_STRATEGY"]
            return merged
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return overrides


def apply_profile(module_globals: dict[str, Any]) -> str:
    name = active_profile_name()
    overrides = PROFILES.get(name)
    if overrides is None:
        known = ", ".join(sorted(PROFILES))
        raise RuntimeError(f"Unknown PMF_PROFILE={name!r} — use one of: {known}")
    overrides = dict(overrides)
    env_mode = str(os.environ.get("PMF_RUN_MODE", "") or "").strip().lower()
    profile_mode = str(overrides.get("RUN_MODE") or "").strip().lower()
    majority = env_mode in ("majority", "majority_hold", "meta") or (
        profile_mode in ("majority", "majority_hold", "meta")
        and env_mode not in ("crowd", "copy", "copy_reverse")
    )
    if name == "cloud" and not majority:
        overrides = _merge_tuned(overrides, name)
    for key, val in overrides.items():
        module_globals[key] = val
    module_globals["PMF_PROFILE"] = name
    run_mode = env_mode
    if run_mode in ("crowd", "copy", "copy_reverse", "majority", "majority_hold", "meta"):
        module_globals["RUN_MODE"] = "majority" if run_mode in ("majority_hold", "meta") else run_mode
    env_single = str(os.environ.get("PMF_MAJORITY_SINGLE_PAIR", "") or "").strip().lower()
    if env_single in ("1", "true", "yes", "on"):
        module_globals["MAJORITY_SINGLE_PAIR"] = True
    elif env_single in ("0", "false", "no", "off"):
        module_globals["MAJORITY_SINGLE_PAIR"] = False
    return name
