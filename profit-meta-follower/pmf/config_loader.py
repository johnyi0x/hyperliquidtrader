"""Apply PMF_PROFILE overrides onto the config module."""

from __future__ import annotations

import os
from typing import Any

from config_profiles import PROFILES


def active_profile_name() -> str:
    return str(os.environ.get("PMF_PROFILE", "local") or "local").strip().lower()


def apply_profile(module_globals: dict[str, Any]) -> str:
    name = active_profile_name()
    overrides = PROFILES.get(name)
    if overrides is None:
        known = ", ".join(sorted(PROFILES))
        raise RuntimeError(f"Unknown PMF_PROFILE={name!r} — use one of: {known}")
    for key, val in overrides.items():
        module_globals[key] = val
    module_globals["PMF_PROFILE"] = name
    return name
