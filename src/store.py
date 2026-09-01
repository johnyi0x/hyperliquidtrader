"""Persist winning setups for live trading."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import entry_signal, exit_signal
from .registry import STRATEGY_BY_ID


@dataclass(frozen=True)
class LiveSetup:
    coin: str
    interval: str
    sid: int
    name: str
    side: int
    p0: float
    p1: float
    p2: float
    aux: float
    tp_pct: float
    sl_pct: float
    balance_pct: float
    use_tpsl: bool
    use_exit_signal: bool
    use_max_hold: bool
    exit_eid: int
    exit_name: str
    ex_p0: float
    ex_aux: float
    dca_enabled: bool
    dca_trigger_pct: float
    dca_max_adds: int
    dca_size_mult: float
    score: float = 0.0
    win_rate_pct: float = 0.0
    trades_per_day: float = 0.0
    rank_score: float = 0.0
    max_hold_bars: int = 48
    mode: str = "legacy"  # "mtf" | "legacy"
    mtf_intervals: tuple[str, ...] = ()
    mtf_ema: int = 50
    mtf_min_agree: int = 3
    mtf_min_score: float = 0.35
    mtf_weight_power: float = 0.5

    @property
    def is_mtf(self) -> bool:
        return str(self.mode).lower() == "mtf"

    def label(self) -> str:
        side = "L" if self.side > 0 else "S"
        legs = 1 + int(self.dca_max_adds) if self.dca_enabled and self.dca_max_adds > 0 else 1
        leg_pct = self.balance_pct / legs
        if self.is_mtf:
            ivs = ",".join(self.mtf_intervals) if self.mtf_intervals else "?"
            return (
                f"MTF@{self.interval}:{self.name}[{side}] "
                f"agree≥{self.mtf_min_agree} score≥{self.mtf_min_score:g} "
                f"ema={self.mtf_ema} TFs=[{ivs}] "
                f"bal={self.balance_pct:g}% total (~{leg_pct:g}%×{legs}) "
                f"tp/sl={self.tp_pct:g} exit={self.exit_name} "
                f"dca={'Y' if self.dca_enabled else 'N'} "
                f"wr={self.win_rate_pct:.0f}% ~{self.trades_per_day:.1f}/d"
            )
        return (
            f"{self.interval}:{self.name}[{side}] "
            f"bal={self.balance_pct:g}% total (~{leg_pct:g}%×{legs}) "
            f"tp/sl={self.tp_pct:g} exit={self.exit_name} "
            f"dca={'Y' if self.dca_enabled else 'N'} "
            f"wr={self.win_rate_pct:.0f}% ~{self.trades_per_day:.1f}/d"
        )


def _row_to_setup(coin: str, row: dict[str, Any]) -> LiveSetup | None:
    try:
        sid = int(row["sid"])
        raw_ivs = row.get("mtf_intervals") or []
        if isinstance(raw_ivs, str):
            mtf_ivs = tuple(x.strip() for x in raw_ivs.split(",") if x.strip())
        else:
            mtf_ivs = tuple(str(x) for x in raw_ivs)
        mode = str(row.get("mode") or ("mtf" if mtf_ivs else "legacy"))
        return LiveSetup(
            coin=str(coin),
            interval=str(row.get("interval", "5m")),
            sid=sid,
            name=str(row.get("name") or STRATEGY_BY_ID[sid].name),
            side=int(row["side"]),
            p0=float(row["p0"]),
            p1=float(row.get("p1", 0)),
            p2=float(row.get("p2", 0)),
            aux=float(row.get("aux", 0)),
            tp_pct=float(row.get("tp_pct", 0)),
            sl_pct=float(row.get("sl_pct", 0)),
            balance_pct=float(row.get("balance_pct", 100)),
            use_tpsl=bool(row.get("use_tpsl", False)),
            use_exit_signal=bool(row.get("use_exit_signal", False)),
            use_max_hold=bool(row.get("use_max_hold", True)),
            exit_eid=int(row.get("exit_eid", -1)),
            exit_name=str(row.get("exit_name", "none")),
            ex_p0=float(row.get("ex_p0", 0)),
            ex_aux=float(row.get("ex_aux", 0)),
            dca_enabled=bool(row.get("dca_enabled", False)),
            dca_trigger_pct=float(row.get("dca_trigger_pct", 0)),
            dca_max_adds=int(row.get("dca_max_adds", 0)),
            dca_size_mult=float(row.get("dca_size_mult", 1)),
            score=float(row.get("score", 0)),
            win_rate_pct=float(row.get("win_rate_pct", 0)),
            trades_per_day=float(row.get("trades_per_day", 0)),
            rank_score=float(row.get("rank_score", 0)),
            max_hold_bars=int(row.get("max_hold_bars", 48)),
            mode=mode,
            mtf_intervals=mtf_ivs,
            mtf_ema=int(row.get("mtf_ema", 50) or 50),
            mtf_min_agree=int(row.get("mtf_min_agree", 3) or 3),
            mtf_min_score=float(row.get("mtf_min_score", 0.35) or 0.35),
            mtf_weight_power=float(row.get("mtf_weight_power", 0.5) or 0.5),
        )
    except (KeyError, TypeError, ValueError):
        return None


class SetupStore:
    def __init__(
        self,
        data_dir: Path,
        logger: logging.Logger,
        *,
        refresh_hours: float = 24.0,
    ) -> None:
        self.log = logger
        self.refresh_hours = max(1.0, float(refresh_hours))
        self._path = Path(data_dir) / "params.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = self._load()
        self._last_attempt_ts = 0.0
        self._retry_cooldown_s = 900.0

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"schema": 2, "per_coin": {}, "updated_at": 0.0}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"schema": 2, "per_coin": {}, "updated_at": 0.0}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        payload = json.dumps(self._state, indent=2)
        last_err: OSError | None = None
        for attempt in range(5):
            try:
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(self._path)
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
        if last_err is not None:
            raise last_err

    def last_updated(self) -> float:
        return float(self._state.get("updated_at", 0.0))

    def mark_attempt(self) -> None:
        """Record a tune attempt even when no winners (starts retry cooldown)."""
        self._last_attempt_ts = time.time()

    def config_mismatch(
        self,
        pair_mode: str,
        reverse_orders: bool,
        *,
        mover_tune: str | None = None,
    ) -> str | None:
        """Why saved setups don't match current pair-selection / reverse flags."""
        saved_mode = str(self._state.get("pair_selection_mode") or "").strip().lower()
        want_mode = str(pair_mode or "").strip().lower()
        if saved_mode != want_mode:
            return f"pair_mode {saved_mode or 'unset'} → {want_mode}"
        if mover_tune:
            saved_lock = str(self._state.get("mover_tune") or "")
            if saved_lock != str(mover_tune):
                return f"mover_tune {saved_lock or 'unset'} → {mover_tune}"
            return None
        if "reverse_orders" not in self._state:
            return "reverse stamp missing"
        if bool(self._state.get("reverse_orders")) != bool(reverse_orders):
            return f"reverse {self._state.get('reverse_orders')} → {reverse_orders}"
        return None

    def refresh_due(self) -> bool:
        if time.time() - self._last_attempt_ts < self._retry_cooldown_s:
            return False
        return time.time() - self.last_updated() >= self.refresh_hours * 3600.0

    def setups_for(self, coin: str) -> list[LiveSetup]:
        row = (self._state.get("per_coin") or {}).get(coin)
        if not row:
            return []
        if isinstance(row, dict) and isinstance(row.get("setups"), list):
            out = []
            for item in row["setups"]:
                s = _row_to_setup(coin, item)
                if s:
                    out.append(s)
            return out
        return []

    def setup_for(self, coin: str) -> LiveSetup | None:
        setups = self.setups_for(coin)
        if not setups:
            return None
        return max(setups, key=lambda s: s.rank_score)

    def setups(self) -> dict[str, LiveSetup]:
        out: dict[str, LiveSetup] = {}
        for coin in self.watch_coins():
            s = self.setup_for(coin)
            if s:
                out[coin] = s
        return out

    def watch_coins(self) -> list[str]:
        coins = self._state.get("watch_coins") or list(
            (self._state.get("per_coin") or {}).keys()
        )
        return [str(c) for c in coins if str(c).strip()]

    def describe(self) -> str:
        if not self.watch_coins():
            return "(no setups)"
        when = self._state.get("updated_at_iso", "never")
        parts = []
        for c in self.watch_coins():
            labels = [s.label() for s in self.setups_for(c)]
            parts.append(f"{c} {{{' ; '.join(labels)}}}")
        return f"{' | '.join(parts)} (updated={when})"

    def save_results(
        self,
        results: dict[str, list[dict[str, Any]]],
        *,
        leverage: int,
        pair_selection_mode: str | None = None,
        reverse_orders: bool | None = None,
        mover_tune: str | None = None,
    ) -> None:
        self._last_attempt_ts = time.time()
        now = time.time()
        per_coin: dict[str, Any] = {}
        for coin, rows in results.items():
            setups_out = []
            for row in rows:
                setups_out.append(dict(row))
            if setups_out:
                per_coin[coin] = {"setups": setups_out}
        if not per_coin:
            self.log.warning("Empty tune results — keeping previous")
            return
        self._state = {
            "schema": 2,
            "updated_at": now,
            "updated_at_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "leverage": leverage,
            "per_coin": per_coin,
            "watch_coins": list(per_coin.keys()),
        }
        if pair_selection_mode is not None:
            self._state["pair_selection_mode"] = str(pair_selection_mode)
        if reverse_orders is not None:
            self._state["reverse_orders"] = bool(reverse_orders)
        if mover_tune is not None:
            self._state["mover_tune"] = str(mover_tune)
        self._save()
        self.log.info("Setups saved: %s", ", ".join(per_coin.keys()))


def signal_from_candles(setup: LiveSetup, candles: list[dict]) -> int:
    return entry_signal(setup, candles)


def exit_from_candles(
    setup: LiveSetup,
    candles: list[dict],
    entry_price: float,
    *,
    position_side: int | None = None,
) -> bool:
    return exit_signal(
        setup,
        candles,
        avg_entry_px=entry_price,
        position_side=position_side,
    )
