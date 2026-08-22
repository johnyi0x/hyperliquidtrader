from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, default=str)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        bak = path.with_suffix(path.suffix + ".corrupt")
        try:
            path.replace(bak)
        except OSError:
            pass
        return default


class StateStore:
    """Crash-safe bot state. Exchange positions always win over this file."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.path = data_dir / "pmf_state.json"
        self.heartbeat_path = data_dir / "heartbeat.json"
        self.data: dict[str, Any] = read_json(
            self.path,
            {
                "basket": [],
                "refresh": {},
                "managed_coins": [],
                "last_rebalance_at": 0.0,
                "fingerprints": {},
                "hyper_wallets": [],
                "last_targets": [],
            },
        )

    def save(self) -> None:
        self.data["saved_at"] = time.time()
        atomic_write_json(self.path, self.data)

    def heartbeat(self, extra: dict[str, Any] | None = None) -> None:
        payload = {"ts": time.time(), **(extra or {})}
        atomic_write_json(self.heartbeat_path, payload)
