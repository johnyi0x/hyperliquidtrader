"""Flatten backtest specs to CSV rows and parse them back for live trading."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from .kernels import FEATURE_NAMES, N_FEAT

WEIGHT_COLS = [f"w_{name}" for name in FEATURE_NAMES]
LIVE_CSV_NAME = "rank_live.csv"
REPO_ROOT = Path(__file__).resolve().parent.parent

CSV_COLUMNS = (
    "sharpe",
    "return_pct",
    "fitness",
    "max_dd_pct",
    "final_equity",
    "round_trips",
    "trips_per_day",
    "avg_hold_h",
    "win_rate_pct",
    "fees",
    "data_from",
    "data_until",
    "n_hours",
    "n_bars",
    "n_coins",
    "family",
    "engine",
    "name",
    "step_h",
    "min_hold_h",
    "enter_top",
    "slots",
    "exec_lag",
    "use_lev",
    "gross_pct",
    "max_pair_share",
    "size_mode",
    "exposure_mode",
    "equity",
    "mode",
    "zscore",
    "enter_th",
    "exit_th",
    "lookback",
    "min_agree",
    "min_wallets",
    "max_abs_funding",
    "require_improve",
    "k",
    "a",
    "b",
    *WEIGHT_COLS,
    "tested_at",
    "run_id",
)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e12:
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value))) if abs(value) >= 1 else f"{value:.6g}"
        return f"{value:.8g}"
    return str(value)


def flatten_result(row: dict[str, Any], *, run_id: str = "") -> dict[str, Any]:
    spec = dict(row.get("spec") or {})
    weights = list(spec.get("weights") or [])
    while len(weights) < N_FEAT:
        weights.append(0.0)
    flat: dict[str, Any] = {key: "" for key in CSV_COLUMNS}
    for key in (
        "sharpe",
        "return_pct",
        "fitness",
        "max_dd_pct",
        "final_equity",
        "round_trips",
        "trips_per_day",
        "avg_hold_h",
        "win_rate_pct",
        "fees",
        "data_from",
        "data_until",
        "n_hours",
        "n_bars",
        "n_coins",
        "tested_at",
    ):
        if key in row:
            flat[key] = row[key]
    flat["run_id"] = run_id or row.get("run_id") or ""
    flat["family"] = spec.get("family") or row.get("family") or ""
    flat["engine"] = spec.get("engine") or "score"
    flat["name"] = spec.get("name") or flat["family"]
    for key in (
        "step_h",
        "min_hold_h",
        "enter_top",
        "slots",
        "exec_lag",
        "use_lev",
        "mode",
        "zscore",
        "enter_th",
        "exit_th",
        "lookback",
        "min_agree",
        "min_wallets",
        "max_abs_funding",
        "require_improve",
        "k",
        "a",
        "b",
        "gross_pct",
        "max_pair_share",
        "size_mode",
        "exposure_mode",
        "equity",
    ):
        if key in spec:
            flat[key] = spec[key]
        elif key in row:
            flat[key] = row[key]
    if not flat["gross_pct"]:
        flat["gross_pct"] = 95
    if not flat["max_pair_share"]:
        flat["max_pair_share"] = 0.7
    if flat["size_mode"] in ("", None):
        flat["size_mode"] = 0
    if flat["exposure_mode"] in ("", None):
        flat["exposure_mode"] = 0
    if not flat["equity"]:
        flat["equity"] = 1000
    if flat["exec_lag"] in ("", None):
        flat["exec_lag"] = 1
    for i, col in enumerate(WEIGHT_COLS):
        flat[col] = weights[i] if i < len(weights) else 0.0
    return flat


def spec_from_row(row: dict[str, Any]) -> dict[str, Any]:
    def _i(key: str, default: int = 0) -> int:
        raw = row.get(key)
        if raw is None or raw == "":
            return default
        return int(float(raw))

    def _f(key: str, default: float = 0.0) -> float:
        raw = row.get(key)
        if raw is None or raw == "":
            return default
        return float(raw)

    weights = [_f(col, 0.0) for col in WEIGHT_COLS]
    engine = str(row.get("engine") or "score").strip() or "score"
    spec: dict[str, Any] = {
        "family": str(row.get("family") or row.get("name") or "composite"),
        "engine": engine,
        "name": str(row.get("name") or row.get("family") or "composite"),
        "step_h": _i("step_h", 1),
        "min_hold_h": _i("min_hold_h", 0),
        "enter_top": _i("enter_top", 0),
        "slots": max(1, _i("slots", 5)),
        "exec_lag": _i("exec_lag", 1),
        "use_lev": _i("use_lev", 0),
        "gross_pct": _f("gross_pct", 95.0),
        "max_pair_share": _f("max_pair_share", 0.7),
        "size_mode": _i("size_mode", 0),
        "exposure_mode": _i("exposure_mode", 0),
        "equity": _f("equity", 1000.0),
        "mode": _i("mode", 0),
        "zscore": _i("zscore", 0),
        "enter_th": _f("enter_th", 0.0),
        "exit_th": _f("exit_th", 0.0),
        "lookback": _i("lookback", 4),
        "min_agree": _f("min_agree", 0.0),
        "min_wallets": _f("min_wallets", 0.0),
        "max_abs_funding": _f("max_abs_funding", 0.0),
        "require_improve": _i("require_improve", 0),
        "k": _i("k", 1),
        "a": _f("a", 0.0),
        "b": _f("b", 0.0),
        "weights": weights,
        "features": list(FEATURE_NAMES),
    }
    return spec


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k, "")) for k in CSV_COLUMNS})


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: _cell(row.get(k, "")) for k in CSV_COLUMNS})


def csv_header_line() -> str:
    return ",".join(CSV_COLUMNS)


def default_live_csv() -> Path:
    return REPO_ROOT / LIVE_CSV_NAME


def require_strategy_row(row: dict[str, str], source: str = LIVE_CSV_NAME) -> None:
    if str(row.get("engine") or "").strip():
        return
    if str(row.get("name") or "").strip() or str(row.get("family") or "").strip():
        return
    raise SystemExit(
        f"{source} has no strategy row. Copy one FULL row from a backtest CSV "
        f"(leaderboard.csv / by_return.csv / by_trades.csv) and paste it as row 2 in {LIVE_CSV_NAME}."
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_csv_row(path: Path, excel_row: int = 2) -> dict[str, str]:
    """excel_row is 1-based including header (row 2 = first / only strategy)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Paste one backtest row into {LIVE_CSV_NAME} (row 2, under the header)."
        )
    try:
        return parse_strategy_text(path.read_text(encoding="utf-8-sig"), excel_row=excel_row)
    except SystemExit as exc:
        msg = str(exc).strip() or "invalid strategy CSV"
        raise SystemExit(f"{path}: {msg}") from None


def load_strategy(
    *,
    csv_path: Path | None = None,
    excel_row: int = 2,
    paste: str | None = None,
) -> dict[str, str]:
    if paste and str(paste).strip():
        return parse_strategy_text(paste, excel_row=excel_row)
    if csv_path is not None and str(csv_path) == "-":
        return parse_strategy_text(sys.stdin.read(), excel_row=excel_row)
    path = Path(csv_path) if csv_path is not None else default_live_csv()
    return load_csv_row(path, excel_row)


def _split_fields(line: str) -> list[str]:
    line = line.strip().lstrip("\ufeff")
    best: list[str] = [line]
    best_score = -10_000
    want = len(CSV_COLUMNS)
    for delim in ("\t", ",", ";"):
        fields = [str(c).strip() for c in next(csv.reader([line], delimiter=delim))]
        while fields and fields[-1] == "":
            fields.pop()
        score = -abs(len(fields) - want)
        if len(fields) == want:
            score += 1000
        elif len(fields) > 1:
            score += min(len(fields), want)
        if score > best_score:
            best_score = score
            best = fields
    return best


def _row_from_header_data(header: list[str], data: list[str]) -> dict[str, str]:
    raw: dict[str, str] = {}
    for i, name in enumerate(header):
        key = str(name).strip()
        val = data[i] if i < len(data) else ""
        raw[key] = val
        raw[key.lower()] = val
    out = {k: str(raw.get(k, raw.get(k.lower(), "")) or "") for k in CSV_COLUMNS}
    if not out.get("engine") and not out.get("name") and not out.get("family") and data:
        for i, key in enumerate(CSV_COLUMNS):
            if i < len(data) and not out[key]:
                out[key] = data[i]
    return out


def parse_strategy_text(text: str, excel_row: int = 2) -> dict[str, str]:
    """Parse a copied CSV/TSV row. Header and data may use different delimiters (Excel paste)."""
    text = text.strip().lstrip("\ufeff")
    if not text:
        raise SystemExit("empty file — paste one backtest row as row 2")
    if text[0] in "{[":
        obj: Any = json.loads(text)
        if isinstance(obj, list):
            if not obj:
                raise SystemExit("Empty JSON strategy list")
            obj = obj[0]
        if isinstance(obj, dict) and "top" in obj and isinstance(obj["top"], list) and obj["top"]:
            obj = obj["top"][0]
        if not isinstance(obj, dict):
            raise SystemExit("Strategy JSON must be an object")
        flat = flatten_result(obj, run_id=str(obj.get("run_id") or ""))
        return {k: _cell(flat.get(k, "")) for k in CSV_COLUMNS}
    physical = [ln for ln in text.splitlines() if ln.strip()]
    if not physical:
        raise SystemExit("empty file — paste one backtest row as row 2")
    first = _split_fields(physical[0])
    if first and first[0].lower() == "sharpe":
        data_lines = physical[1:]
        if not data_lines:
            raise SystemExit("header only — paste one backtest row as row 2")
        idx = 0 if excel_row <= 1 else excel_row - 2
        if idx < 0 or idx >= len(data_lines):
            raise SystemExit(f"has {len(data_lines)} strategy rows; Excel row {excel_row} is out of range")
        return _row_from_header_data(first, _split_fields(data_lines[idx]))
    idx = 0 if excel_row <= 2 else excel_row - 2
    if idx < 0 or idx >= len(physical):
        raise SystemExit(f"has {len(physical)} pasted rows; row {excel_row} is out of range")
    return _row_from_header_data(list(CSV_COLUMNS), _split_fields(physical[idx]))


def export_search_dir(directory: Path) -> Path:
    """Turn an old jsonl/json search folder into sortable CSVs."""
    directory = Path(directory)
    if directory.is_file():
        directory = directory.parent
    jsonl = directory / "results.jsonl"
    leaderboard = directory / "leaderboard.json"
    rows: list[dict[str, Any]] = []
    run_id = directory.name
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(flatten_result(json.loads(line), run_id=run_id))
    elif leaderboard.exists():
        payload = json.loads(leaderboard.read_text(encoding="utf-8"))
        items = payload.get("top") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            items = [payload]
        for item in items:
            if isinstance(item, dict):
                rows.append(flatten_result(item, run_id=run_id))
    else:
        raise SystemExit(f"No results.jsonl or leaderboard.json in {directory}")
    if not rows:
        raise SystemExit(f"No strategies found in {directory}")
    write_csv(directory / "results.csv", rows)
    by_sharpe = sorted(rows, key=lambda r: (-float(r.get("sharpe") or 0), -float(r.get("return_pct") or 0)))
    by_return = sorted(rows, key=lambda r: (-float(r.get("return_pct") or 0), -float(r.get("sharpe") or 0)))
    by_trades = sorted(rows, key=lambda r: (-float(r.get("round_trips") or 0), -float(r.get("sharpe") or 0)))
    write_csv(directory / "leaderboard.csv", by_sharpe)
    write_csv(directory / "by_return.csv", by_return)
    write_csv(directory / "by_trades.csv", by_trades)
    return directory / "leaderboard.csv"
