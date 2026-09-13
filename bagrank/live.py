"""Live/paper trader: run one CSV row with the same kernels as the backtest."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .dsn import default_live_sqlite_path, load_env
from .engine import lagged_holdings
from .hlcycle import LiveBoard
from .kernels import NAME_TO_ID
from .panel import panel_from_sqlite
from .prices import fill_panel_from_collector
from .specio import default_live_csv, load_strategy, require_strategy_row, spec_from_row
from .store import connect
from .timeutil import to_iso

log = logging.getLogger("bagrank.live")

_STOP = False
FEE = 0.0005


def _request_stop(_signum=None, _frame=None) -> None:
    global _STOP
    _STOP = True
    log.info("Stop requested")


def load_panel(sqlite_path: Path, venue: str = "hyperliquid"):
    conn = connect(sqlite_path)
    try:
        panel = panel_from_sqlite(conn, venue=venue)
        fill_panel_from_collector(conn, panel, venue=venue)
        return panel
    finally:
        conn.close()


def _weight_targets(
    holds: list[dict[str, Any]],
    *,
    equity: float,
    gross_pct: float,
    max_pair_share: float,
    use_lev: int,
    slots: int = 0,
    exposure_mode: int = 0,
) -> list[dict[str, Any]]:
    wsum = sum(max(float(h["wallets"]), 1e-9) for h in holds)
    if wsum <= 0:
        return []
    n = len(holds)
    slot_n = max(int(slots or n), 1)
    gross = max(0.05, min(2.0, float(gross_pct) / 100.0))
    if int(exposure_mode) == 1:
        gross *= n / float(slot_n)
    elif int(exposure_mode) == 2:
        gross *= (n / float(slot_n)) ** 0.5
    gross = max(0.05, min(2.0, gross))
    out: list[dict[str, Any]] = []
    for h in holds:
        px = float(h.get("px") or 0)
        if px <= 0:
            continue
        weight = float(h["wallets"]) / wsum
        if max_pair_share > 0:
            weight = min(weight, float(max_pair_share))
        lev = float(h["lev"]) if use_lev else 1.0
        if lev < 1.0:
            lev = 1.0
        notional = equity * gross * weight * lev
        if notional <= 0:
            continue
        size = notional / px
        out.append({**h, "notional": notional, "size": size, "weight": weight})
    return out


class PaperBook:
    def __init__(self, cash: float) -> None:
        self.cash = float(cash)
        self.positions: dict[str, dict[str, Any]] = {}

    def equity(self, marks: dict[str, float]) -> float:
        eq = self.cash
        for coin, pos in self.positions.items():
            px = marks.get(coin) or float(pos.get("entry") or 0)
            entry = float(pos.get("entry") or px)
            signed = float(pos["size"]) if pos["side"] == "long" else -float(pos["size"])
            eq += signed * (px - entry)
        return eq

    def close(self, coin: str, px: float) -> None:
        pos = self.positions.pop(coin, None)
        if not pos:
            return
        signed = float(pos["size"]) if pos["side"] == "long" else -float(pos["size"])
        entry = float(pos.get("entry") or px)
        pnl = signed * (px - entry)
        fee = abs(float(pos["size"]) * px) * FEE
        self.cash += pnl - fee
        log.info("PAPER close %s %s pnl=%.2f", coin, pos["side"], pnl - fee)

    def open(self, coin: str, side: str, size: float, px: float) -> None:
        fee = abs(size * px) * FEE
        self.cash -= fee
        self.positions[coin] = {"side": side, "size": size, "entry": px}
        log.info("PAPER open %s %s sz=%.6f @ %.6f", coin, side, size, px)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"opened": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"opened": {}}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _can_exit(state: dict[str, Any], coin: str, min_hold_h: int, now: float) -> bool:
    if min_hold_h <= 0:
        return True
    opened = (state.get("opened") or {}).get(coin)
    if not opened:
        return True
    return (now - float(opened)) >= min_hold_h * 3600.0


def _info_only():
    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    return Info(constants.MAINNET_API_URL, skip_ws=True)


def _make_client():
    from src.exchange_client import HyperliquidClient

    wallet = (os.environ.get("HYPE_WALLET_ADDRESS") or "").strip()
    key = (os.environ.get("HYPE_PRIVATE_KEY") or "").strip()
    if not wallet or not key:
        raise SystemExit("Set HYPE_WALLET_ADDRESS and HYPE_PRIVATE_KEY in .env for --live")
    return HyperliquidClient(wallet, key, "BTC", log)


def _mids(client) -> dict[str, float]:
    try:
        raw = client.info.all_mids()
    except Exception as exc:
        log.warning("all_mids failed: %s", exc)
        return {}
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for coin, px in raw.items():
            try:
                v = float(px)
            except (TypeError, ValueError):
                continue
            if v > 0:
                out[str(coin)] = v
    return out


def _pos_side(pos: Any) -> str:
    if isinstance(pos, dict):
        return str(pos.get("side") or "")
    return str(getattr(pos, "side", "") or "")


def _occupied_slots(
    have: dict[str, Any],
    want: dict[str, dict[str, Any]],
    state: dict[str, Any],
    min_hold_h: int,
    now: float,
) -> list[str]:
    """Coins that still fill a slot (wanted, or locked by min-hold). Same as backtest."""
    held: list[str] = []
    for coin, pos in have.items():
        side = _pos_side(pos)
        keep = coin in want and want[coin]["side"] == side
        if keep or not _can_exit(state, coin, min_hold_h, now):
            held.append(coin)
    return held


def _apply_live(client, desired: list[dict[str, Any]], state: dict[str, Any], min_hold_h: int, slots: int = 1) -> None:
    from src.market_resolver import resolve_market
    from src.pricing import round_size

    ok, positions = client.fetch_open_positions(force=True)
    if not ok:
        log.warning("Position query failed — skip this cycle")
        return
    have = {coin: pos for coin, pos in positions}
    want = {str(d["coin"]): d for d in desired}
    now = time.time()
    equity = float(client.get_account_value(force=True) or 0)
    snap = (round(equity, 2), tuple(want), tuple(have))
    if getattr(_apply_live, "_last_snap", None) != snap:
        log.info("Live equity $%.2f want %s have %s", equity, list(want), list(have))
        _apply_live._last_snap = snap

    for coin, pos in list(have.items()):
        keep = coin in want and want[coin]["side"] == pos.side
        if keep:
            continue
        if not _can_exit(state, coin, min_hold_h, now):
            log.info("Min-hold: keep %s", coin)
            continue
        try:
            market = resolve_market(client.info, coin)
            client.apply_market(market)
            client.place_market_close()
            (state.setdefault("opened", {})).pop(coin, None)
            log.info("LIVE close %s", coin)
        except Exception as exc:
            log.warning("Close %s failed: %s", coin, exc)

    ok, positions = client.fetch_open_positions(force=True)
    have = {coin: pos for coin, pos in (positions if ok else [])}
    occupied = _occupied_slots(have, want, state, min_hold_h, now)
    free = max(0, max(int(slots or 1), 1) - len(occupied))
    mids = _mids(client)
    for coin, d in want.items():
        if coin in have and _pos_side(have[coin]) == d["side"]:
            continue
        if free <= 0:
            continue
        px = mids.get(coin) or float(d.get("px") or 0)
        if px <= 0:
            log.warning("No price for %s — skip open", coin)
            continue
        size = float(d["size"]) if d.get("size") else float(d["notional"]) / px
        try:
            market = resolve_market(client.info, coin)
            client.apply_market(market)
            size = round_size(size, client.sz_decimals)
            if size <= 0:
                continue
            lev = max(1, int(d.get("lev") or 1))
            try:
                client.set_leverage(lev)
            except Exception:
                pass
            client.place_market_open(d["side"] == "long", size)
            state.setdefault("opened", {})[coin] = now
            free -= 1
            log.info("LIVE open %s %s sz=%s", coin, d["side"], size)
        except Exception as exc:
            log.warning("Open %s failed: %s", coin, exc)


def _apply_paper(
    book: PaperBook,
    desired: list[dict[str, Any]],
    state: dict[str, Any],
    min_hold_h: int,
    marks: dict[str, float],
    slots: int = 1,
) -> None:
    now = time.time()
    want = {str(d["coin"]): d for d in desired}
    for coin in list(book.positions):
        pos = book.positions[coin]
        keep = coin in want and want[coin]["side"] == pos["side"]
        if keep:
            continue
        if not _can_exit(state, coin, min_hold_h, now):
            log.info("Min-hold: keep %s", coin)
            continue
        book.close(coin, marks.get(coin) or float(pos.get("entry") or 0))
        (state.setdefault("opened", {})).pop(coin, None)
    occupied = _occupied_slots(book.positions, want, state, min_hold_h, now)
    free = max(0, max(int(slots or 1), 1) - len(occupied))
    for coin, d in want.items():
        if coin in book.positions and book.positions[coin]["side"] == d["side"]:
            continue
        if free <= 0:
            continue
        px = marks.get(coin) or float(d.get("px") or 0)
        if px <= 0:
            continue
        size = float(d.get("size") or 0)
        if size <= 0 and d.get("notional"):
            size = float(d["notional"]) / px
        book.open(coin, d["side"], size, px)
        state.setdefault("opened", {})[coin] = now
        free -= 1


def run_live(
    spec: dict[str, Any],
    *,
    sqlite_path: Path,
    mode: str,
    poll_s: float = 30.0,
    refresh_board: bool = True,
    state_path: Path | None = None,
) -> int:
    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)
    state_path = state_path or (sqlite_path.parent / "live_state.json")
    state = _load_state(state_path)
    last_signal = ""
    client = None
    book = None
    if mode == "live":
        client = _make_client()
    else:
        book = PaperBook(float(spec.get("equity") or 1000.0))
    info = client.info if client is not None else None
    if info is None:
        try:
            info = _info_only()
        except Exception as exc:
            log.warning("HL info client failed; public HTTP will still gather the board: %s", exc)
    keep_hours = max(1, int(spec.get("lookback") or 1) * max(1, int(spec.get("step_h") or 1)))
    try:
        from .backup import seed_recent_board

        seeded = seed_recent_board(sqlite_path, keep_hours)
        if seeded:
            log.info("Neon backfill once: %s hours. Neon will not be queried again.", seeded)
    except Exception as exc:
        log.warning("Neon backfill skipped: %s", exc)
    board = LiveBoard(sqlite_path, info, keep_hours=keep_hours) if refresh_board else None
    log.info("Hourly board file %s | rolling window %sh", sqlite_path, keep_hours)
    log.info(
        "Trading %s | engine=%s family=%s step=%sh hold=%sh slots=%s lag=%s lookback=%s",
        mode,
        spec.get("engine"),
        spec.get("family"),
        spec.get("step_h"),
        spec.get("min_hold_h"),
        spec.get("slots"),
        spec.get("exec_lag"),
        spec.get("lookback"),
    )
    logged_wait = False
    while not _STOP:
        have = 0 if panel is None else panel.n_times
        if have >= keep_hours:
            holds = lagged_holdings(panel, spec)
            equity = float(spec.get("equity") or 1000.0)
            marks = {h["coin"]: float(h["px"]) for h in holds if h.get("px")}
            if client is not None:
                try:
                    equity = float(client.get_account_value(force=True) or equity)
                    marks.update(_mids(client))
                    for h in holds:
                        if h["coin"] in marks:
                            h["px"] = marks[h["coin"]]
                except Exception as exc:
                    log.warning("Live marks/equity failed: %s", exc)
            elif book is not None:
                equity = book.equity(marks)
            sized = _weight_targets(
                holds,
                equity=equity,
                gross_pct=float(spec.get("gross_pct") or 95),
                max_pair_share=float(spec.get("max_pair_share") or 0.7),
                use_lev=int(spec.get("use_lev") or 0),
                slots=int(spec.get("slots") or 0),
                exposure_mode=int(spec.get("exposure_mode") or 0),
            )
            key = json.dumps(
                [(d["coin"], d["side"]) for d in sized],
                separators=(",", ":"),
            )
            sig_ts = to_iso(holds[0]["signal_unix"]) if holds else ""
            bar_ts = to_iso(holds[0]["bar_unix"]) if holds else ""
            changed = key != last_signal
            if changed:
                log.info(
                    "Targets bar=%s signal=%s n=%s %s",
                    bar_ts,
                    sig_ts,
                    len(sized),
                    [(d["coin"], d["side"], round(d.get("notional", 0), 1)) for d in sized],
                )
                last_signal = key
            if mode == "dry":
                pass
            elif mode == "paper" and book is not None:
                _apply_paper(
                    book,
                    sized,
                    state,
                    int(spec.get("min_hold_h") or 0),
                    marks,
                    slots=int(spec.get("slots") or 1),
                )
                if changed:
                    log.info("PAPER equity $%.2f positions %s", book.equity(marks), list(book.positions))
            elif mode == "live" and client is not None:
                _apply_live(
                    client,
                    sized,
                    state,
                    int(spec.get("min_hold_h") or 0),
                    slots=int(spec.get("slots") or 1),
                )
            state["last_bar"] = bar_ts
            state["last_signal"] = sig_ts
            _save_state(state_path, state)
        elif not logged_wait:
            log.info("Waiting for %s lookback hours (have %s). Set NEON_DATABASE to backfill once.", keep_hours, have)
            logged_wait = True
        if board is not None:
            try:
                board.maybe_refresh()
            except Exception as exc:
                log.warning("HL board gather failed: %s", exc)
        time.sleep(max(5.0, float(poll_s)))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    load_env()
    p = argparse.ArgumentParser(
        description="Trade the strategy in rank_live.csv (one copied backtest row)."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Defaults to rank_live.csv in the repo root",
    )
    p.add_argument("--row", type=int, default=2, help="Excel row including header (2 = the pasted strategy)")
    p.add_argument("--db", type=Path, default=default_live_sqlite_path())
    p.add_argument("--paper", action="store_true", help="Simulated fills (default when not on Railway)")
    p.add_argument("--live", action="store_true", help="Send real Hyperliquid orders")
    p.add_argument("--dry-run", action="store_true", help="Print targets only")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument(
        "--no-refresh",
        action="store_true",
        help="Do not snapshot Hyperliquid wallets; reuse hours already in local sqlite",
    )
    p.add_argument(
        "--no-sync",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    csv_path = args.csv or default_live_csv()
    raw = load_strategy(csv_path=csv_path, excel_row=args.row)
    require_strategy_row(raw, str(csv_path))
    spec = spec_from_row(raw)
    if spec.get("engine") == "named" and spec.get("name") not in NAME_TO_ID:
        raise SystemExit(f"Unknown strategy name {spec.get('name')!r} in {csv_path}")
    on_railway = bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RAILWAY_PROJECT_ID"))
    if args.dry_run:
        mode = "dry"
    elif args.paper:
        mode = "paper"
    elif args.live or on_railway:
        mode = "live"
    else:
        mode = "paper"
    log.info(
        "Loaded %s row %s | %s %s sharpe=%s ret=%s",
        csv_path.name,
        args.row,
        spec.get("family"),
        spec.get("engine"),
        raw.get("sharpe"),
        raw.get("return_pct"),
    )
    return run_live(
        spec,
        sqlite_path=args.db,
        mode=mode,
        poll_s=args.poll,
        refresh_board=not (args.no_refresh or args.no_sync),
    )


if __name__ == "__main__":
    raise SystemExit(main())
