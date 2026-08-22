"""
Append-only data file rotation so long-running bots do not grow forever.

Keeps small fixed state files as-is (params.json, trade_state.json, …).
Rotates growing jsonl/log-style files into timestamped archives and prunes old ones.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Soft limits — rotate before files get huge enough to hurt disk / editors.
DEFAULT_MAX_BYTES = 25 * 1024 * 1024  # 25 MB
DEFAULT_MAX_LINES = 100_000
DEFAULT_BACKUP_COUNT = 20


def _count_lines_fast(path: Path, *, sample_limit: int = DEFAULT_MAX_LINES + 1) -> int:
    """Count lines up to sample_limit (enough to know we should rotate)."""
    n = 0
    try:
        with open(path, "rb") as f:
            for _ in f:
                n += 1
                if n >= sample_limit:
                    return n
    except OSError:
        return 0
    return n


def rotate_if_needed(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_lines: int = DEFAULT_MAX_LINES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    logger: logging.Logger | None = None,
) -> bool:
    """
    If `path` exceeds size or line limits, rename it to a timestamped archive
    and leave a fresh empty file for new writes. Returns True if rotated.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= 0:
        return False

    over_size = size >= max_bytes
    over_lines = False
    if not over_size and max_lines > 0:
        over_lines = _count_lines_fast(path, sample_limit=max_lines + 1) > max_lines
    if not over_size and not over_lines:
        return False

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = path.parent / "archives"
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        archive_dir = path.parent

    archive = archive_dir / f"{path.stem}.{stamp}{path.suffix}"
    # Avoid collision if two rotates in the same second.
    if archive.exists():
        archive = archive_dir / f"{path.stem}.{stamp}_{int(time.time() * 1000)}{path.suffix}"

    log = logger or logging.getLogger("hl-multi")
    try:
        path.replace(archive)
        path.write_text("", encoding="utf-8")
        log.info(
            "Rotated data file %s → %s (size was %.1f MB)",
            path.name,
            archive.name,
            size / (1024 * 1024),
        )
    except OSError as exc:
        log.warning("Failed to rotate %s: %s", path, exc)
        return False

    prune_archives(path, backup_count=backup_count, logger=log)
    return True


def prune_archives(
    live_path: Path,
    *,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    logger: logging.Logger | None = None,
) -> None:
    """Keep only the newest `backup_count` archives for this stem."""
    live_path = Path(live_path)
    patterns = [
        live_path.parent / "archives" / f"{live_path.stem}.*{live_path.suffix}",
        live_path.parent / f"{live_path.stem}.*{live_path.suffix}",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        parent = pattern.parent
        if not parent.exists():
            continue
        for p in parent.glob(pattern.name):
            if p.resolve() == live_path.resolve():
                continue
            if p in seen:
                continue
            seen.add(p)
            files.append(p)
    if len(files) <= backup_count:
        return
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    log = logger or logging.getLogger("hl-multi")
    for old in files[backup_count:]:
        try:
            old.unlink(missing_ok=True)
            log.info("Pruned old archive %s", old.name)
        except OSError:
            pass


def append_jsonl(
    path: Path,
    row: dict[str, Any],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_lines: int = DEFAULT_MAX_LINES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    logger: logging.Logger | None = None,
) -> None:
    """Append one JSON object as a line, rotating the file first if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rotate_if_needed(
        path,
        max_bytes=max_bytes,
        max_lines=max_lines,
        backup_count=backup_count,
        logger=logger,
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")


def housekeep_data_dir(
    data_dir: Path,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """
    On startup / periodic: rotate known growing files if over limits.
    Safe to call often; no-ops when files are small.
    """
    data_dir = Path(data_dir)
    log = logger or logging.getLogger("hl-multi")
    growing = (
        "tuning.jsonl",
        "paper_trades.jsonl",
        "ema_dev_paper_trades.jsonl",
    )
    for name in growing:
        rotate_if_needed(data_dir / name, logger=log)

    # Prune any leftover archives even if current file is small.
    for name in growing:
        prune_archives(data_dir / name, logger=log)
