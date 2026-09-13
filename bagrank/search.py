"""Endless random search over collector-feature strategies until Ctrl+C."""

from __future__ import annotations

import json
import logging
import math
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .kernels import (
    FEATURE_NAMES,
    MODE_BREAKOUT,
    MODE_CARRY,
    MODE_CROWD,
    MODE_DIVERGE,
    MODE_FADE,
    MODE_FLIP,
    MODE_GIVEBACK,
    MODE_STREAK,
    MODE_THRESH,
    MODE_TOP,
    N_FEAT,
    STRATEGY_NAMES,
    simulate,
)
from .engine import compute_targets
from .panel import RankPanel, resample_closed
from .specio import append_csv, flatten_result, write_csv
from .timeutil import to_iso

log = logging.getLogger("bagrank.search")

FAMILIES = (
    "composite",
    "rank_mom",
    "wallet_flow",
    "agree_spike",
    "funding_fade",
    "funding_follow",
    "oi_thrust",
    "premium_revert",
    "conviction_lead",
    "notional_flow",
    "thin_hold",
    "lev_crowding",
    "price_confirm",
    "vs_yesterday",
    "anti_rank",
    "streak",
    "breakout",
    "giveback",
    "zscore_composite",
    "flip_majority",
    "rank_accel",
    "ls_imbalance",
    "carry",
    "divergence",
    "crowd_fade",
    "agree_rank",
    "funding_oi",
    "concentrated",
    "spray",
    "named",
)

GROSS_CHOICES = (20, 30, 40, 50, 65, 80, 95, 110, 125, 140)
PAIR_CHOICES = (0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)
SIZE_MODES = (0, 1, 2, 3, 4, 5, 6)

_STOP = False


def _request_stop(_signum=None, _frame=None) -> None:
    global _STOP
    _STOP = True
    log.info("Stop requested — finishing current trial and saving")


def run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def range_tag(t0: int, t1: int) -> str:
    a = to_iso(t0).replace(":", "")
    b = to_iso(t1).replace(":", "")
    return f"from_{a}_until_{b}"


def run_folder_name(span: dict[str, Any], started: str) -> str:
    tag = range_tag(int(span["data_from_unix"]), int(span["data_until_unix"]))
    stamp = started.replace(":", "").replace("-", "")
    if stamp.endswith("Z") and "T" in stamp:
        stamp = stamp[:-1] + "Z"
    return f"run_{stamp}_{tag}"


def data_span(panel: RankPanel) -> dict[str, Any]:
    t0 = int(panel.cycle_unix[0]) if panel.n_times else 0
    t1 = int(panel.cycle_unix[-1]) if panel.n_times else 0
    hours = 0.0
    if panel.n_times >= 2:
        hours = float(t1 - t0) / 3600.0 + 1.0
    return {
        "data_from": to_iso(t0) if t0 else "",
        "data_until": to_iso(t1) if t1 else "",
        "data_from_unix": t0,
        "data_until_unix": t1,
        "n_hours": round(hours, 2),
        "n_bars": int(panel.n_times),
        "n_coins": int(panel.n_coins),
    }


def _sparse_weights(rng: random.Random, preset: np.ndarray | None = None) -> list[float]:
    w = np.zeros(N_FEAT, dtype=np.float64)
    if preset is not None:
        w[:] = preset
    else:
        keep = rng.randint(2, 7)
        idx = rng.sample(range(N_FEAT), keep)
        for i in idx:
            w[i] = rng.gauss(0.0, 1.0)
    nrm = float(np.linalg.norm(w))
    if nrm > 1e-9:
        w /= nrm
    return [round(float(x), 6) for x in w]


def _preset(name: str) -> np.ndarray:
    w = np.zeros(N_FEAT, dtype=np.float64)
    if name == "rank_mom":
        w[0] = 0.4
        w[1] = 1.0
    elif name == "wallet_flow":
        w[2] = 0.3
        w[3] = 1.0
    elif name == "agree_spike":
        w[5] = 0.4
        w[6] = 1.0
        w[1] = 0.3
    elif name == "funding_fade":
        w[10] = -1.0
        w[1] = 0.2
    elif name == "funding_follow":
        w[10] = 1.0
        w[2] = 0.4
    elif name == "oi_thrust":
        w[11] = 1.0
        w[1] = 0.5
        w[3] = 0.3
    elif name == "premium_revert":
        w[12] = -1.0
        w[13] = -0.4
    elif name == "conviction_lead":
        w[8] = 1.0
        w[1] = 0.4
    elif name == "notional_flow":
        w[9] = 1.0
        w[3] = 0.5
    elif name == "thin_hold":
        w[2] = 1.0
        w[4] = -1.0
    elif name == "lev_crowding":
        w[7] = 1.0
        w[2] = 0.4
        w[5] = 0.3
    elif name == "price_confirm":
        w[1] = 0.7
        w[13] = 0.7
        w[3] = 0.4
    elif name == "vs_yesterday":
        w[14] = 1.0
        w[1] = 0.5
    elif name == "anti_rank":
        w[0] = -1.0
        w[1] = -0.3
    elif name == "rank_accel":
        w[1] = 0.5
        w[17] = 1.0
        w[3] = 0.3
    elif name == "ls_imbalance":
        w[16] = 1.0
        w[5] = 0.4
        w[1] = 0.3
    elif name == "carry":
        w[20] = 1.0
        w[10] = 0.4
    elif name == "divergence":
        w[1] = 1.0
        w[13] = -0.8
        w[17] = 0.4
    elif name == "crowd_fade":
        w[21] = -1.0
        w[2] = -0.4
        w[0] = -0.3
    elif name == "agree_rank":
        w[0] = 0.6
        w[5] = 1.0
        w[6] = 0.5
    elif name == "funding_oi":
        w[10] = -0.7
        w[11] = 1.0
        w[3] = 0.3
    else:
        w[0] = 0.5
        w[1] = 0.5
        w[3] = 0.3
    return w


def sample_spec(rng: random.Random, *, named_ok: bool = True) -> dict[str, Any]:
    family = rng.choice(FAMILIES)
    if family == "named" and not named_ok:
        family = "composite"
    step = rng.choice((1, 2, 3, 4, 6, 8, 12))
    hold_h = rng.choice((0, 2, 4, 6, 8, 12, 24, 36, 48))
    top = rng.choice((0, 5, 8, 10, 12, 15, 20, 30))
    slots = rng.choice((1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24))
    gross = float(rng.choice(GROSS_CHOICES))
    pair = float(rng.choice(PAIR_CHOICES))
    size_mode = int(rng.choice(SIZE_MODES))
    exposure_mode = int(rng.choice((0, 0, 1, 1, 2, 3)))
    use_lev = int(rng.choice((0, 0, 0, 1)))
    if family == "concentrated":
        slots = rng.choice((1, 1, 2, 2, 3))
        gross = float(rng.choice((80, 95, 110, 125, 140)))
        pair = float(rng.choice((0.5, 0.65, 0.8, 1.0)))
        size_mode = rng.choice((2, 3, 6, 0))
        exposure_mode = rng.choice((0, 3))
    elif family == "spray":
        slots = rng.choice((10, 12, 16, 20, 24))
        gross = float(rng.choice((30, 40, 50, 65, 80)))
        pair = float(rng.choice((0.15, 0.2, 0.3, 0.4)))
        size_mode = rng.choice((1, 1, 4, 2))
        exposure_mode = rng.choice((0, 1, 2))
    if top > 0:
        slots = min(slots, top)
    exec_lag = rng.choice((1, 1, 1, 2))
    spec: dict[str, Any] = {
        "family": family,
        "step_h": step,
        "min_hold_h": hold_h,
        "enter_top": top,
        "slots": slots,
        "exec_lag": exec_lag,
        "use_lev": use_lev,
        "engine": "score",
        "gross_pct": gross,
        "max_pair_share": pair,
        "size_mode": size_mode,
        "exposure_mode": exposure_mode,
    }
    if family == "named":
        spec["engine"] = "named"
        spec["name"] = rng.choice(STRATEGY_NAMES)
        spec["k"] = rng.choice((1, 2, 3, 5, 8, 14))
        spec["lookback"] = rng.choice((1, 2, 4, 6, 8, 12, 24))
        spec["a"] = float(rng.choice((3, 4, 8, 12, 21, 25, 30)))
        spec["b"] = float(rng.choice((8, 12, 21, 26, 55, 70, 75)))
        if spec["b"] <= spec["a"]:
            spec["b"] = spec["a"] + 8
        return spec

    mode = MODE_TOP
    zscore = 1 if family == "zscore_composite" else rng.choice((0, 0, 1))
    if family == "streak":
        mode = MODE_STREAK
    elif family == "breakout":
        mode = MODE_BREAKOUT
    elif family == "giveback":
        mode = MODE_GIVEBACK
    elif family in ("anti_rank", "premium_revert", "funding_fade", "crowd_fade"):
        mode = rng.choice((MODE_FADE, MODE_TOP, MODE_THRESH, MODE_CROWD))
    elif family == "flip_majority":
        mode = MODE_FLIP
    elif family == "divergence":
        mode = MODE_DIVERGE
    elif family == "carry":
        mode = MODE_CARRY
    elif family == "crowd_fade":
        mode = MODE_CROWD
    elif rng.random() < 0.3:
        mode = rng.choice(
            (MODE_TOP, MODE_THRESH, MODE_FADE, MODE_STREAK, MODE_GIVEBACK, MODE_FLIP, MODE_DIVERGE, MODE_CROWD, MODE_CARRY)
        )

    if family == "composite" or family == "zscore_composite":
        weights = _sparse_weights(rng)
    else:
        weights = _sparse_weights(rng, _preset(family) + rng.gauss(0.0, 0.15) * np.ones(N_FEAT))
        for i in range(N_FEAT):
            if rng.random() < 0.12:
                weights[i] = 0.0
        weights = _sparse_weights(rng, np.asarray(weights, dtype=np.float64))

    spec.update(
        {
            "name": family,
            "weights": weights,
            "features": list(FEATURE_NAMES),
            "mode": int(mode),
            "zscore": int(zscore),
            "enter_th": round(rng.uniform(-0.5, 1.5), 4),
            "exit_th": round(rng.uniform(-1.5, 0.8), 4),
            "lookback": rng.choice((2, 3, 4, 6, 8, 12, 24)),
            "min_agree": round(rng.choice((0.0, 0.0, 0.0, 0.55, 0.65, 0.75)), 3),
            "min_wallets": float(rng.choice((0, 0, 0, 5, 10, 15, 20))),
            "max_abs_funding": round(rng.choice((0.0, 0.0, 0.0, 0.0005, 0.001, 0.002)), 6),
            "require_improve": int(rng.choice((0, 0, 1))),
        }
    )
    return spec


def mutate_spec(rng: random.Random, spec: dict[str, Any], *, scale: float = 0.25) -> dict[str, Any]:
    out = json.loads(json.dumps(spec))
    if rng.random() < 0.55:
        out["step_h"] = rng.choice((1, 2, 3, 4, 6, 8, 12))
    if rng.random() < 0.45:
        out["min_hold_h"] = rng.choice((0, 2, 4, 8, 12, 24, 36))
    if rng.random() < 0.5:
        out["slots"] = max(1, int(out.get("slots") or 5) + rng.choice((-3, -2, -1, 0, 1, 2, 3)))
    if rng.random() < 0.4:
        out["enter_top"] = rng.choice((0, 5, 8, 10, 15, 20, 30))
    if rng.random() < 0.5:
        out["gross_pct"] = float(rng.choice(GROSS_CHOICES))
    if rng.random() < 0.4:
        out["max_pair_share"] = float(rng.choice(PAIR_CHOICES))
    if rng.random() < 0.35:
        out["size_mode"] = int(rng.choice(SIZE_MODES))
    if rng.random() < 0.35:
        out["exposure_mode"] = int(rng.choice((0, 1, 2, 3)))
    if rng.random() < 0.2:
        out["use_lev"] = 1 - int(out.get("use_lev") or 0)
    top = int(out.get("enter_top") or 0)
    if top > 0:
        out["slots"] = min(max(1, int(out["slots"])), top)
    if out.get("engine") == "score" and out.get("weights"):
        w = np.asarray(out["weights"], dtype=np.float64)
        n_touch = 1 if scale < 0.3 else rng.randint(1, 4)
        for _ in range(n_touch):
            j = rng.randrange(N_FEAT)
            w[j] += rng.gauss(0.0, max(0.08, scale))
            if rng.random() < 0.12:
                w[j] = 0.0
        out["weights"] = _sparse_weights(rng, w)
        out["enter_th"] = round(float(out.get("enter_th") or 0) + rng.gauss(0, scale * 0.5), 4)
        out["exit_th"] = round(float(out.get("exit_th") or 0) + rng.gauss(0, scale * 0.5), 4)
        if rng.random() < 0.25:
            out["mode"] = int(rng.choice(
                (MODE_TOP, MODE_THRESH, MODE_FADE, MODE_STREAK, MODE_GIVEBACK, MODE_FLIP, MODE_DIVERGE, MODE_CROWD, MODE_CARRY)
            ))
    else:
        out["k"] = rng.choice((1, 2, 3, 5, 8, 14))
        out["lookback"] = rng.choice((1, 2, 4, 8, 12, 24))
    return out


def crossover_spec(rng: random.Random, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(a))
    for key in (
        "step_h",
        "min_hold_h",
        "slots",
        "enter_top",
        "gross_pct",
        "max_pair_share",
        "size_mode",
        "exposure_mode",
        "use_lev",
        "mode",
        "zscore",
        "lookback",
        "min_agree",
        "min_wallets",
        "exec_lag",
    ):
        if rng.random() < 0.5:
            out[key] = b.get(key, out.get(key))
    if out.get("engine") == "score" and b.get("engine") == "score":
        wa = list(out.get("weights") or [0.0] * N_FEAT)
        wb = list(b.get("weights") or [0.0] * N_FEAT)
        while len(wa) < N_FEAT:
            wa.append(0.0)
        while len(wb) < N_FEAT:
            wb.append(0.0)
        mixed = [wa[i] if rng.random() < 0.5 else wb[i] for i in range(N_FEAT)]
        out["weights"] = _sparse_weights(rng, np.asarray(mixed, dtype=np.float64))
        out["engine"] = "score"
        out["family"] = rng.choice((str(a.get("family") or "composite"), str(b.get("family") or "composite")))
        out["name"] = out["family"]
    return out


def fitness(row: dict[str, Any]) -> float:
    """Prefer real profit and enough trades. Clip 1-bar Sharpe spikes."""
    ret = float(row.get("return_pct") or 0)
    sh = float(row.get("sharpe") or 0)
    dd = max(float(row.get("max_dd_pct") or 0), 0.25)
    trips = int(row.get("round_trips") or 0)
    wr = float(row.get("win_rate_pct") or 0)
    if trips < 1:
        return -1e6 + ret
    sh_f = max(min(sh, 8.0), 0.0)
    ret_use = ret
    if trips < 3:
        frac = trips / 3.0
        sh_f *= frac
        ret_use = ret * frac
    calmar = ret_use / dd
    trip_f = math.log1p(trips)
    wr_f = max(wr - 40.0, 0.0) / 20.0
    return ret_use * 2.0 + calmar * 4.0 + sh_f * 1.2 + trip_f * 3.0 + wr_f


def spec_key(spec: dict[str, Any]) -> str:
    blob = {k: spec[k] for k in sorted(spec) if k != "features"}
    return json.dumps(blob, sort_keys=True, separators=(",", ":"))


def _hold_bars(min_hold_hours: int, step_hours: int) -> int:
    if min_hold_hours <= 0:
        return 0
    return max(1, int(math.ceil(float(min_hold_hours) / float(max(1, step_hours)))))


def _sharpe(equity: np.ndarray, bar_hours: float) -> float:
    if equity.size < 3:
        return 0.0
    prev = equity[:-1]
    nxt = equity[1:]
    ok = (prev > 1e-12) & np.isfinite(prev) & np.isfinite(nxt)
    if not bool(np.any(ok)):
        return 0.0
    rets = nxt[ok] / prev[ok] - 1.0
    sd = float(np.std(rets))
    if sd < 1e-12:
        return 0.0
    bars_year = (365.0 * 24.0) / max(float(bar_hours), 1.0)
    return float(np.mean(rets) / sd * np.sqrt(bars_year))


def run_one(
    hourly: RankPanel,
    spec: dict[str, Any],
    *,
    equity: float,
    gross_pct: float,
    max_pair_share: float,
) -> dict[str, Any] | None:
    step = max(1, int(spec.get("step_h") or 1))
    panel = resample_closed(hourly, step)
    if panel.n_times < 4:
        return None
    spec = dict(spec)
    gross = float(spec.get("gross_pct") or gross_pct)
    pair = float(spec.get("max_pair_share") or max_pair_share)
    spec["gross_pct"] = gross
    spec["max_pair_share"] = pair
    spec["equity"] = float(equity)
    spec["size_mode"] = int(spec.get("size_mode") or 0)
    spec["exposure_mode"] = int(spec.get("exposure_mode") or 0)
    slots = max(1, int(spec.get("slots") or 5))
    span = data_span(panel)
    tc, ts, tw, tl = compute_targets(panel, spec)
    ret, dd, trips, wr, fees, final, eq, avg_hold = simulate(
        panel.marks,
        tc,
        ts,
        tw,
        tl,
        0.0005,
        max(0.05, min(2.0, gross / 100.0)),
        float(equity),
        slots,
        float(pair),
        int(spec.get("exec_lag") or 1),
        _hold_bars(int(spec.get("min_hold_h") or 0), step),
        int(spec.get("use_lev") or 0),
        float(step),
        int(spec.get("exposure_mode") or 0),
    )
    span_h = float(span["n_hours"] or 0)
    trips_day = (float(trips) / (span_h / 24.0)) if span_h >= 1.0 else float(trips)
    row = {
        **span,
        "step_h": step,
        "tested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec": spec,
        "family": spec.get("family") or spec.get("name"),
        "return_pct": round(float(ret), 4),
        "max_dd_pct": round(float(dd), 4),
        "sharpe": round(_sharpe(eq, float(step)), 4),
        "round_trips": int(trips),
        "trips_per_day": round(trips_day, 3),
        "avg_hold_h": round(float(avg_hold), 3),
        "win_rate_pct": round(float(wr), 3),
        "fees": round(float(fees), 4),
        "final_equity": round(float(final), 2),
        "gross_pct": gross,
        "max_pair_share": pair,
        "size_mode": int(spec.get("size_mode") or 0),
        "exposure_mode": int(spec.get("exposure_mode") or 0),
        "equity": float(equity),
    }
    row["fitness"] = round(fitness(row), 4)
    return row


class ResultStore:
    def __init__(self, directory: Path, span: dict[str, Any]) -> None:
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.directory = directory / run_folder_name(span, started)
        n = 1
        while self.directory.exists():
            n += 1
            self.directory = directory / f"{run_folder_name(span, started)}_{n}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tag = self.directory.name
        self.jsonl = self.directory / "results.jsonl"
        self.leaderboard_path = self.directory / "leaderboard.json"
        self.meta_path = self.directory / "meta.json"
        self.csv_all = self.directory / "results.csv"
        self.csv_sharpe = self.directory / "leaderboard.csv"
        self.csv_return = self.directory / "by_return.csv"
        self.csv_trades = self.directory / "by_trades.csv"
        self.csv_fit = self.directory / "by_fitness.csv"
        self.span = span
        self.n = 0
        self.best: dict[str, Any] | None = None
        self.best_ret: dict[str, Any] | None = None
        self.best_fit: dict[str, Any] | None = None
        self.by_sharpe: list[dict[str, Any]] = []
        self.by_return: list[dict[str, Any]] = []
        self.by_trades: list[dict[str, Any]] = []
        self.by_fit: list[dict[str, Any]] = []
        meta = {
            **span,
            "search_started_at": started,
            "run_id": self.tag,
            "jsonl": str(self.jsonl),
            "results_csv": str(self.csv_all),
            "leaderboard_csv": str(self.csv_sharpe),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _consider(self, row: dict[str, Any]) -> None:
        trips = int(row.get("round_trips") or 0)
        if trips >= 2:
            if self.best is None or float(row.get("sharpe") or -1e9) > float(self.best.get("sharpe") or -1e9):
                self.best = row
        elif self.best is None:
            self.best = row
        if self.best_ret is None or float(row.get("return_pct") or -1e9) > float(self.best_ret.get("return_pct") or -1e9):
            self.best_ret = row
        if self.best_fit is None or float(row.get("fitness") or -1e9) > float(self.best_fit.get("fitness") or -1e9):
            self.best_fit = row

    def append(self, row: dict[str, Any]) -> None:
        with self.jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            fh.flush()
        self.n += 1
        self._consider(row)
        flat = flatten_result(row, run_id=self.tag)
        try:
            append_csv(self.csv_all, flat)
        except OSError as exc:
            log.warning("Could not append results.csv (close Excel if it is open): %s", exc)
        self._push_top(self.by_sharpe, flat, "sharpe")
        self._push_top(self.by_return, flat, "return_pct")
        self._push_top(self.by_trades, flat, "round_trips")
        self._push_top(self.by_fit, flat, "fitness")

    def _push_top(self, bucket: list[dict[str, Any]], row: dict[str, Any], key: str, keep: int = 500) -> None:
        bucket.append(row)
        bucket.sort(key=lambda r: -float(r.get(key) or 0))
        del bucket[keep:]

    def write_csvs(self) -> None:
        try:
            write_csv(self.csv_sharpe, self.by_sharpe)
            write_csv(self.csv_return, self.by_return)
            write_csv(self.csv_trades, self.by_trades)
            write_csv(self.csv_fit, self.by_fit)
        except OSError as exc:
            log.warning("Could not write ranked CSV (close Excel if it is open): %s", exc)

    def write_leaderboard(self, rows: list[dict[str, Any]], *, keep: int = 100) -> None:
        top = sorted(rows, key=lambda r: (-float(r.get("sharpe") or 0), -float(r.get("return_pct") or 0)))[:keep]
        payload = {
            **self.span,
            "run_id": self.tag,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trials": self.n,
            "top": top,
        }
        self.leaderboard_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.write_csvs()


def _push_elite(bucket: list[dict[str, Any]], row: dict[str, Any], key: str, keep: int) -> None:
    bucket.append(row)
    bucket.sort(key=lambda r: -float(r.get(key) or 0))
    del bucket[keep:]


def _next_spec(
    rng: random.Random,
    *,
    elite_fit: list[dict[str, Any]],
    elite_ret: list[dict[str, Any]],
    elite_sh: list[dict[str, Any]],
    explore_p: float,
    mut_scale: float,
) -> dict[str, Any]:
    roll = rng.random()
    if roll < explore_p or not elite_fit:
        return sample_spec(rng)
    pool = elite_fit + elite_ret + elite_sh
    if roll < explore_p + 0.12 and len(pool) >= 2:
        a = rng.choice(pool)["spec"]
        b = rng.choice(pool)["spec"]
        return crossover_spec(rng, a, b)
    if roll < explore_p + 0.12 + 0.22 and elite_ret:
        return mutate_spec(rng, rng.choice(elite_ret)["spec"], scale=mut_scale)
    if roll < explore_p + 0.12 + 0.22 + 0.18 and elite_sh:
        return mutate_spec(rng, rng.choice(elite_sh)["spec"], scale=mut_scale)
    return mutate_spec(rng, rng.choice(elite_fit)["spec"], scale=mut_scale)


def search_loop(
    hourly: RankPanel,
    *,
    out_dir: Path,
    equity: float = 1000.0,
    gross_pct: float = 95.0,
    max_pair_share: float = 0.70,
    seed: int | None = None,
) -> int:
    span = data_span(hourly)
    log.info(
        "Search data %s → %s (%s hours, %s coins, %s 1h bars)",
        span["data_from"],
        span["data_until"],
        span["n_hours"],
        span["n_coins"],
        span["n_bars"],
    )
    rng = random.Random(seed)
    store = ResultStore(out_dir, span)
    log.info(
        "Writing this run to %s | by_return.csv=profit  leaderboard.csv=Sharpe  by_fitness.csv=blend",
        store.directory,
    )
    seen: set[str] = set()
    elite_fit: list[dict[str, Any]] = []
    elite_ret: list[dict[str, Any]] = []
    elite_sh: list[dict[str, Any]] = []
    recent: list[dict[str, Any]] = []
    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)
    t0 = time.time()
    idle_ret = 0
    idle_fit = 0
    last_best_ret = -1e18
    last_best_fit = -1e18
    while not _STOP:
        elapsed = time.time() - t0
        explore_p = 0.20 + 0.55 * math.exp(-elapsed / 420.0)
        mut_scale = 0.18 + 0.35 * math.exp(-elapsed / 700.0)
        if idle_ret >= 80:
            explore_p = min(0.88, explore_p + 0.28)
            mut_scale = min(0.7, mut_scale + 0.25)
        if idle_ret >= 250:
            explore_p = 0.92
            mut_scale = 0.55
        spec = _next_spec(
            rng,
            elite_fit=elite_fit,
            elite_ret=elite_ret,
            elite_sh=elite_sh,
            explore_p=explore_p,
            mut_scale=mut_scale,
        )
        if spec.get("gross_pct") in (None, ""):
            spec["gross_pct"] = gross_pct
        if spec.get("max_pair_share") in (None, ""):
            spec["max_pair_share"] = max_pair_share
        key = spec_key(spec)
        if key in seen:
            spec = mutate_spec(rng, spec, scale=max(0.3, mut_scale))
            key = spec_key(spec)
            if key in seen:
                continue
        seen.add(key)
        row = run_one(
            hourly,
            spec,
            equity=equity,
            gross_pct=float(spec.get("gross_pct") or gross_pct),
            max_pair_share=float(spec.get("max_pair_share") or max_pair_share),
        )
        if row is None:
            continue
        store.append(row)
        recent.append(row)
        if len(recent) > 200:
            del recent[:50]
        trips = int(row.get("round_trips") or 0)
        _push_elite(elite_fit, row, "fitness", 36)
        if trips >= 2:
            _push_elite(elite_ret, row, "return_pct", 24)
            _push_elite(elite_sh, row, "sharpe", 24)
        ret_now = float((store.best_ret or {}).get("return_pct") or -1e18)
        fit_now = float((store.best_fit or {}).get("fitness") or -1e18)
        if ret_now > last_best_ret + 1e-9:
            last_best_ret = ret_now
            idle_ret = 0
        else:
            idle_ret += 1
        if fit_now > last_best_fit + 1e-9:
            last_best_fit = fit_now
            idle_fit = 0
        else:
            idle_fit += 1
        if store.n % 25 == 0:
            store.write_leaderboard(elite_fit + elite_ret + recent[-40:])
            br = store.best_ret or {}
            bs = store.best or {}
            bf = store.best_fit or {}
            rate = store.n / max(time.time() - t0, 1e-6)
            log.info(
                "trials=%s (%.1f/s)  BEST RET %s%% sh=%s trips=%s gross=%s%%  |  best sh=%s ret=%s%%  |  fit=%s  idle_ret=%s  %s",
                store.n,
                rate,
                br.get("return_pct"),
                br.get("sharpe"),
                br.get("round_trips"),
                (br.get("spec") or {}).get("gross_pct"),
                bs.get("sharpe"),
                bs.get("return_pct"),
                bf.get("fitness"),
                idle_ret,
                store.directory.name,
            )
    store.write_leaderboard(elite_fit + elite_ret + recent[-40:])
    log.info(
        "Stopped after %s trials. Open these CSVs and sort columns in Excel:\n  %s\n  %s\n  %s\n  %s\nCopy one full row into rank_live.csv then python run_rank.py",
        store.n,
        store.csv_return,
        store.csv_sharpe,
        store.csv_fit,
        store.csv_trades,
    )
    return 0
