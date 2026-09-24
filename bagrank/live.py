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

from .dsn import default_live_sqlite_path, default_pnl_sqlite_path, load_env
from .engine import hourly_board_rank1, lagged_holdings
from .hlcycle import LiveBoard
from .kernels import NAME_TO_ID
from .lev import DEFAULT_LEV_X, DEFAULT_MAX_LEV, apply_exchange_max_lev, fetch_exchange_max_lev, pair_size_lev
from .panel import panel_from_sqlite
from .prices import fill_panel_from_collector
from .specio import default_live_csv, load_strategy, require_strategy_row, spec_from_row
from .store import connect
from .timeutil import to_iso

log = logging.getLogger("bagrank.live")

_STOP = False
FEE = 0.0005


def spec_board(spec: dict[str, Any]) -> str:
    return "pnl" if str(spec.get("board") or "roi").strip().lower() == "pnl" else "roi"


def live_hour_windows(spec: dict[str, Any]) -> tuple[int, int]:
    """Hours of closed board needed to trade, and hours kept on disk.

    Same lookback + exec_lag the backtest kernel sees. Keep at least 72h on a
    PnL board so a missed gather still has recent closed hours.
    """
    step = max(1, int(spec.get("step_h") or 1))
    look = max(1, int(spec.get("lookback") or 1))
    lag = max(0, int(spec.get("exec_lag") or 1))
    need = max(2, (look + lag) * step)
    keep = need
    if spec_board(spec) == "pnl":
        keep = max(keep, 72)
    return need, keep


def normalize_follow_rank1_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """follow_rank1 live: one slot on latest HL week-PnL board #1."""
    if str(spec.get("name") or "") != "follow_rank1":
        return spec
    spec["engine"] = "named"
    spec["family"] = "follow_rank1"
    spec["board"] = "pnl"
    spec["slots"] = 1
    spec["enter_top"] = 1
    spec["max_pair_share"] = 1.0
    return spec


def reset_follow_rank1_boot_gate(state: dict[str, Any]) -> None:
    """Clear follow_* boot flags. Does not touch exchange positions or opened{}."""
    for key in list(state.keys()):
        if str(key).startswith("follow_"):
            state.pop(key, None)


def gate_follow_rank1_holds(
    holds: list[dict[str, Any]],
    panel: Any,
    spec: dict[str, Any],
    state: dict[str, Any],
    *,
    held: dict[str, str] | None = None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Hold exactly the latest finished-hour HL week-PnL board #1 (1 coin).

    Redeploy-safe:
    - If exchange already has any position → always target current #1 (keep if
      it matches, flip if not). Never flatten a still-#1 BTC on restart.
    - If flat → seed boot #1 and stay flat until hourly #1 changes, then enter.

    Ignores lagged kernel / CSV step_h / exec_lag (backtest-only params).
    """
    del holds, spec, now
    lead = hourly_board_rank1(panel)
    if lead is None:
        return []
    coin = str(lead["coin"])
    side = str(lead["side"])
    bar_unix = int(lead.get("bar_unix") or 0)
    stamp = (coin, side, bar_unix)
    if state.get("follow_last_logged") != stamp:
        log.info(
            "follow_rank1: HL week-PnL board #1 %s %s @ %s",
            coin,
            side,
            to_iso(bar_unix) if bar_unix else "?",
        )
        state["follow_last_logged"] = stamp

    held_map = {str(c): str(s) for c, s in (held or {}).items() if c and s}

    # Already in the market (e.g. Railway redeploy with BTC open): track #1.
    if held_map:
        state["follow_armed"] = True
        if not state.get("follow_seed_coin"):
            state["follow_seed_coin"] = coin
            state["follow_seed_side"] = side
        matching = coin in held_map and held_map[coin] == side
        if matching and len(held_map) == 1:
            return [lead]
        if not matching:
            log.info(
                "follow_rank1: exchange has %s but #1 is %s %s — will flip",
                list(held_map.items()),
                coin,
                side,
            )
        elif len(held_map) > 1:
            log.info(
                "follow_rank1: extra positions %s — keep only #1 %s %s",
                list(held_map),
                coin,
                side,
            )
        return [lead]

    # Flat: wait for a fresh #1 change before first entry.
    seed_coin = state.get("follow_seed_coin")
    if not seed_coin:
        state["follow_seed_coin"] = coin
        state["follow_seed_side"] = side
        state["follow_armed"] = False
        log.info(
            "follow_rank1: flat seed #1 %s %s — wait for hourly #1 change before entry",
            coin,
            side,
        )
        return []

    if not state.get("follow_armed"):
        if coin == str(seed_coin) and side == str(state.get("follow_seed_side") or side):
            return []
        state["follow_armed"] = True
        log.info(
            "follow_rank1: #1 changed to %s %s (was %s %s) — enter and follow #1",
            coin,
            side,
            seed_coin,
            state.get("follow_seed_side"),
        )

    return [lead]


def _held_sides_from_exchange(client) -> dict[str, str] | None:
    """Return coin→side, or None if the query failed (do not treat as flat)."""
    try:
        ok, positions = client.fetch_open_positions(force=True)
    except Exception as exc:
        log.warning("Position query failed: %s", exc)
        return None
    if not ok:
        log.warning("Position query failed — skip targeting this cycle")
        return None
    return {str(coin): str(pos.side) for coin, pos in positions}


def _held_sides_from_book(book: "PaperBook") -> dict[str, str]:
    return {str(c): str(p.get("side") or "") for c, p in book.positions.items()}


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
    lev_x: float = DEFAULT_LEV_X,
    max_lev: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    wsum = sum(max(float(h["wallets"]), 1e-9) for h in holds)
    if wsum <= 0:
        return []
    n = len(holds)
    slot_n = max(int(slots or n), 1)
    # Same gross clamp as search.run_one / simulate_nb (0.05 .. 2.0).
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
        if use_lev:
            lev = float(h["lev"]) if h.get("lev") else 1.0
            if lev < 1.0:
                lev = 1.0
        else:
            mx = 0.0
            if max_lev:
                mx = float(max_lev.get(h["coin"]) or 0)
            if mx <= 0:
                mx = float(h.get("max_lev") or 0)
            if mx <= 0:
                mx = DEFAULT_MAX_LEV
            lev = float(int(pair_size_lev(mx, lev_x)))
        notional = equity * gross * weight * lev
        if notional <= 0:
            continue
        size = notional / px
        out.append({**h, "notional": notional, "size": size, "weight": weight, "lev": lev})
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


def _mids_from_info(info) -> dict[str, float]:
    if info is None:
        return {}
    try:
        raw = info.all_mids()
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


def _mids(client) -> dict[str, float]:
    return _mids_from_info(getattr(client, "info", None))


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
    *,
    respect_min_hold: bool = True,
) -> list[str]:
    """Coins that still fill a slot (wanted, soft-open, or locked by min-hold)."""
    held: list[str] = []
    for coin, pos in have.items():
        side = _pos_side(pos)
        keep = coin in want and want[coin]["side"] == side
        locked = respect_min_hold and not _can_exit(state, coin, min_hold_h, now)
        if keep or locked:
            held.append(coin)
    # Exchange lag: treat a just-opened want coin as occupying until it appears in have.
    for coin, opened_at in list((state.get("opened") or {}).items()):
        if coin in held:
            continue
        if coin not in want:
            continue
        if coin in have and _pos_side(have[coin]) == want[coin]["side"]:
            continue
        if (now - float(opened_at or 0)) <= 180.0:
            held.append(coin)
    return held


def _maker_wait_s() -> float:
    try:
        return max(5.0, float(os.environ.get("BAGRANK_MAKER_WAIT_S") or 20))
    except (TypeError, ValueError):
        return 20.0


def _maker_attempts() -> int:
    try:
        return max(1, int(os.environ.get("BAGRANK_MAKER_ATTEMPTS") or 5))
    except (TypeError, ValueError):
        return 5


def _order_executor(client):
    """Post-only mid → wait → reprice → market fallback (maker fees when possible)."""
    from src.order_executor import OrderExecutor

    wait_s = _maker_wait_s()
    attempts = _maker_attempts()
    return OrderExecutor(
        client,
        wait_seconds=int(wait_s),
        max_attempts=attempts,
        logger=log,
        use_market_orders=False,
        mid_limit_then_market=True,
        mid_limit_wait_seconds=wait_s,
        mid_limit_attempts=attempts,
    )


def _live_maker_close(client, coin: str) -> bool:
    """Fully close coin via post-only mid limits, market only if unfilled."""
    from src.market_resolver import resolve_market

    market = resolve_market(client.info, coin)
    client.apply_market(market)
    try:
        client.cancel_all_orders_for_coin()
    except Exception as exc:
        log.warning("Cancel orders on %s before close: %s", coin, exc)
    pos = client.get_position(force=True)
    if pos is None:
        return True
    log.info(
        "LIVE maker-close %s %s sz=%s (post-only mid x%s @ %.0fs, then market)",
        coin,
        pos.side,
        pos.size,
        _maker_attempts(),
        _maker_wait_s(),
    )
    ex = _order_executor(client)
    ok = ex.execute_mid_close_full()
    if not ok:
        log.warning("Maker-close %s incomplete — will retry next cycle", coin)
    return ok


def _fit_open_size(client, size: float, px: float, lev: int) -> float:
    """Cap size so isolated margin (notional/lev) fits ~90% of free equity."""
    from src.pricing import floor_size

    equity = float(client.get_account_value(force=True) or 0)
    lev_i = max(1, int(lev))
    px_f = float(px or 0)
    if equity <= 0 or px_f <= 0:
        return 0.0
    max_ntl = equity * 0.90 * float(lev_i)
    fitted = min(float(size), max_ntl / px_f)
    out = floor_size(fitted, client.sz_decimals)
    if out <= 0:
        from src.pricing import round_size

        out = round_size(fitted, client.sz_decimals)
    if fitted + 1e-12 < float(size):
        log.info(
            "Size capped to margin: equity=$%.2f lev=%sx max_ntl=$%.1f sz %.6f -> %.6f",
            equity,
            lev_i,
            max_ntl,
            size,
            out,
        )
    return out


def _live_maker_open(client, coin: str, side: str, size: float, lev: int) -> bool:
    """Open via post-only mid limits, market only if unfilled after retries."""
    from src.market_resolver import resolve_market
    from src.pricing import floor_size, round_size

    market = resolve_market(client.info, coin)
    client.apply_market(market)
    try:
        mids = _mids(client)
        px = float(mids.get(coin) or 0)
    except Exception:
        px = 0.0
    if px <= 0:
        try:
            l2 = client.l2_book()
            bids, asks = l2["levels"][0], l2["levels"][1]
            px = (float(bids[0]["px"]) + float(asks[0]["px"])) / 2.0
        except Exception as exc:
            log.warning("No mid for %s: %s", coin, exc)
            return False

    lev_i = max(1, int(lev))
    try:
        client.set_leverage(lev_i)
    except Exception as exc:
        log.warning("set_leverage %s failed (continuing): %s", coin, exc)

    size = _fit_open_size(client, size, px, lev_i)
    if size <= 0:
        log.warning("Open %s size rounds to zero after margin fit — skip", coin)
        return False

    try:
        client.cancel_entry_orders_for_coin()
    except Exception:
        pass

    is_buy = str(side) == "long"
    # Shrink + retry on insufficient margin (equity/buffer edge cases).
    for shrink in range(4):
        sz = floor_size(size * (0.85 ** shrink), client.sz_decimals)
        if sz <= 0:
            sz = round_size(size * (0.85 ** shrink), client.sz_decimals)
        if sz <= 0:
            break
        log.info(
            "LIVE maker-open %s %s sz=%s (post-only mid x%s @ %.0fs, then market)%s",
            coin,
            side,
            sz,
            _maker_attempts(),
            _maker_wait_s(),
            f" shrink={shrink}" if shrink else "",
        )
        try:
            ex = _order_executor(client)
            ok = ex.execute_mid_open(is_buy, sz)
            if ok:
                return True
            log.warning("Maker-open %s incomplete — will retry next cycle", coin)
            return False
        except Exception as exc:
            msg = str(exc).lower()
            if "insufficient margin" in msg:
                log.warning(
                    "Open %s insufficient margin at sz=%s — shrinking: %s",
                    coin,
                    sz,
                    exc,
                )
                time.sleep(0.5)
                continue
            log.warning("Open %s failed: %s", coin, exc)
            return False
    log.warning("Open %s failed after margin shrinks — retry next cycle", coin)
    return False


def _apply_live(
    client,
    desired: list[dict[str, Any]],
    state: dict[str, Any],
    min_hold_h: int,
    slots: int = 1,
    *,
    force_flip: bool = False,
) -> None:
    ok, positions = client.fetch_open_positions(force=True)
    if not ok:
        log.warning("Position query failed — skip this cycle")
        return
    have = {coin: pos for coin, pos in positions}
    if force_flip and desired:
        desired = desired[:1]
    want = {str(d["coin"]): d for d in desired}
    now = time.time()
    equity = float(client.get_account_value(force=True) or 0)
    snap = (round(equity, 2), tuple(want), tuple(have))
    if getattr(_apply_live, "_last_snap", None) != snap:
        log.info("Live equity $%.2f want %s have %s", equity, list(want), list(have))
        _apply_live._last_snap = snap

    slot_n = 1 if force_flip else max(int(slots or 1), 1)

    for coin, pos in list(have.items()):
        keep = coin in want and want[coin]["side"] == pos.side
        if keep:
            # Soft: matching #1 — do not resize, cancel, or re-enter.
            continue
        if not force_flip and not _can_exit(state, coin, min_hold_h, now):
            log.info("Min-hold: keep %s", coin)
            continue
        try:
            if _live_maker_close(client, coin):
                (state.setdefault("opened", {})).pop(coin, None)
                log.info("LIVE close %s done%s", coin, " (flip)" if force_flip else "")
            else:
                log.warning("Close %s not flat yet — retry next cycle", coin)
        except Exception as exc:
            log.warning("Close %s failed: %s", coin, exc)

    ok, positions = client.fetch_open_positions(force=True)
    have = {coin: pos for coin, pos in (positions if ok else [])}
    for coin in list((state.get("opened") or {})):
        if coin not in have and coin not in want:
            (state.setdefault("opened", {})).pop(coin, None)
    occupied = _occupied_slots(
        have,
        want,
        state,
        min_hold_h,
        now,
        respect_min_hold=not force_flip,
    )
    stray = [c for c in have if c not in want or _pos_side(have[c]) != want[c]["side"]]
    if stray:
        log.info("Waiting to clear stray positions before open: %s", stray)
        return
    if slot_n <= 1 and have:
        for coin, d in want.items():
            if coin in have and _pos_side(have[coin]) == d["side"]:
                return
        if have:
            log.info("Slot full with %s — skip open until flat", list(have))
            return
    free = max(0, slot_n - len(occupied))
    mids = _mids(client)
    for coin, d in want.items():
        if coin in have and _pos_side(have[coin]) == d["side"]:
            continue
        if coin in occupied:
            continue
        if free <= 0:
            continue
        px = mids.get(coin) or float(d.get("px") or 0)
        if px <= 0:
            log.warning("No price for %s — skip open", coin)
            continue
        size = float(d["size"]) if d.get("size") else float(d["notional"]) / px
        lev = max(1, int(d.get("lev") or 1))
        try:
            if _live_maker_open(client, coin, str(d["side"]), size, lev):
                state.setdefault("opened", {})[coin] = now
                free -= 1
                occupied.append(coin)
                log.info("LIVE open %s %s done", coin, d["side"])
            else:
                ok2, positions2 = client.fetch_open_positions(force=True)
                if ok2 and any(c == coin for c, _ in positions2):
                    state.setdefault("opened", {})[coin] = now
                    occupied.append(coin)
                    free = max(0, free - 1)
                    log.warning(
                        "Open %s partial fill — holding soft slot until reconciled",
                        coin,
                    )
                else:
                    log.warning("Open %s failed — retry next cycle", coin)
        except Exception as exc:
            log.warning("Open %s failed: %s", coin, exc)
        if force_flip:
            break


def _apply_paper(
    book: PaperBook,
    desired: list[dict[str, Any]],
    state: dict[str, Any],
    min_hold_h: int,
    marks: dict[str, float],
    slots: int = 1,
    *,
    force_flip: bool = False,
) -> None:
    now = time.time()
    if force_flip and desired:
        desired = desired[:1]
    want = {str(d["coin"]): d for d in desired}
    slot_n = 1 if force_flip else max(int(slots or 1), 1)
    for coin in list(book.positions):
        pos = book.positions[coin]
        keep = coin in want and want[coin]["side"] == pos["side"]
        if keep:
            continue
        if not force_flip and not _can_exit(state, coin, min_hold_h, now):
            log.info("Min-hold: keep %s", coin)
            continue
        book.close(coin, marks.get(coin) or float(pos.get("entry") or 0))
        (state.setdefault("opened", {})).pop(coin, None)
    occupied = _occupied_slots(
        book.positions,
        want,
        state,
        min_hold_h,
        now,
        respect_min_hold=not force_flip,
    )
    stray = [
        c
        for c in book.positions
        if c not in want or book.positions[c]["side"] != want[c]["side"]
    ]
    if stray:
        return
    free = max(0, slot_n - len(occupied))
    for coin, d in want.items():
        if coin in book.positions and book.positions[coin]["side"] == d["side"]:
            continue
        if coin in occupied or free <= 0:
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
        occupied.append(coin)



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
    log.info(
        "Signals match the backtest kernel (lagged closed bars, min-hold, CSV slots/gross). "
        "family=%s engine=%s name=%s",
        spec.get("family"),
        spec.get("engine"),
        spec.get("name"),
    )
    if mode == "live":
        log.info(
            "LIVE orders: post-only mid limits x%s every %.0fs, then market leftover "
            "(BAGRANK_MAKER_WAIT_S / BAGRANK_MAKER_ATTEMPTS)",
            _maker_attempts(),
            _maker_wait_s(),
        )
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
    board_kind = spec_board(spec)
    need_hours, keep_hours = live_hour_windows(spec)
    lev_x = float(spec.get("lev_x") if spec.get("lev_x") not in (None, "") else DEFAULT_LEV_X)
    spec["lev_x"] = lev_x
    spec["board"] = board_kind
    max_lev_map = fetch_exchange_max_lev()
    if board_kind == "pnl":
        try:
            from .backup import seed_local_hours, seed_recent_board

            seeded = seed_recent_board(sqlite_path, need_hours, kind="pnl")
            if seeded:
                log.info(
                    "Neon PnL seed once: %s hours (lookback only). Neon will not be queried again.",
                    seeded,
                )
            else:
                seeded = seed_local_hours(default_pnl_sqlite_path(), sqlite_path, need_hours)
                if seeded:
                    log.info("Local PnL backup seed: %s hours. Neon was not used.", seeded)
        except Exception as exc:
            log.warning("PnL lookback seed skipped: %s", exc)
    else:
        try:
            from .backup import seed_recent_board

            seeded = seed_recent_board(sqlite_path, need_hours, kind="roi")
            if seeded:
                log.info("Neon backfill once: %s hours. Neon will not be queried again.", seeded)
        except Exception as exc:
            log.warning("Neon backfill skipped: %s", exc)
    board = LiveBoard(sqlite_path, info, keep_hours=keep_hours, rank_by=board_kind) if refresh_board else None
    log.info(
        "Hourly board file %s | trade after %sh | retain %sh | rank_by=%s",
        sqlite_path,
        need_hours,
        keep_hours,
        board_kind,
    )
    log.info(
        "Trading %s | board=%s engine=%s family=%s step=%sh hold=%sh slots=%s lag=%s lookback=%s | "
        "lev_x=%s (pair uses floor(maxLev/%s) integer, 1x if maxLev<%s) | HL maxLev for %s coins",
        mode,
        board_kind,
        spec.get("engine"),
        spec.get("family"),
        spec.get("step_h"),
        spec.get("min_hold_h"),
        spec.get("slots"),
        spec.get("exec_lag"),
        spec.get("lookback"),
        lev_x,
        lev_x,
        lev_x,
        len(max_lev_map),
    )
    logged_wait = False
    while not _STOP:
        try:
            panel = load_panel(sqlite_path) if sqlite_path.exists() else None
            if panel is not None:
                apply_exchange_max_lev(panel, max_lev_map)
            have = 0 if panel is None else panel.n_times
            if have >= need_hours:
                holds = lagged_holdings(panel, spec)
                equity = float(spec.get("equity") or 1000.0)
                marks = {h["coin"]: float(h["px"]) for h in holds if h.get("px")}
                try:
                    marks.update(_mids_from_info(info))
                except Exception as exc:
                    log.warning("HL mids failed: %s", exc)
                for h in holds:
                    if h["coin"] in marks:
                        h["px"] = marks[h["coin"]]
                if client is not None:
                    try:
                        equity = float(client.get_account_value(force=True) or equity)
                    except Exception as exc:
                        log.warning("Live equity failed: %s", exc)
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
                    lev_x=lev_x,
                    max_lev=max_lev_map,
                )
                key = json.dumps(
                    [(d["coin"], d["side"]) for d in sized],
                    separators=(",", ":"),
                )
                sig_ts = to_iso(holds[0]["signal_unix"]) if holds else ""
                bar_ts = to_iso(holds[0]["bar_unix"]) if holds else ""
                changed = key != last_signal
                log.info(
                    "Cycle hours=%s closed_bar=%s signal_bar=%s lag=%s gross=%s%% slots=%s hold=%sh targets=%s",
                    have,
                    bar_ts or "-",
                    sig_ts or "-",
                    spec.get("exec_lag"),
                    spec.get("gross_pct"),
                    spec.get("slots"),
                    spec.get("min_hold_h"),
                    [
                        (d["coin"], d["side"], round(float(d.get("notional") or 0), 1), f"{d.get('lev')}x")
                        for d in sized
                    ]
                    or "FLAT",
                )
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
                    try:
                        _apply_paper(
                            book,
                            sized,
                            state,
                            int(spec.get("min_hold_h") or 0),
                            marks,
                            slots=int(spec.get("slots") or 1),
                        )
                    except Exception as exc:
                        log.warning("Paper apply failed (will retry next poll): %s", exc)
                    if changed:
                        log.info(
                            "PAPER equity $%.2f positions %s",
                            book.equity(marks),
                            list(book.positions),
                        )
                elif mode == "live" and client is not None:
                    try:
                        _apply_live(
                            client,
                            sized,
                            state,
                            int(spec.get("min_hold_h") or 0),
                            slots=int(spec.get("slots") or 1),
                        )
                    except Exception as exc:
                        log.warning("Live apply failed (will retry next poll): %s", exc)
                state["last_bar"] = bar_ts
                state["last_signal"] = sig_ts
                try:
                    _save_state(state_path, state)
                except Exception as exc:
                    log.warning("State save failed: %s", exc)
            elif not logged_wait:
                if board_kind == "pnl":
                    hint = "Optional NEON_DATABASE_PNL warm-start; else HL gathers hours."
                else:
                    hint = "Set NEON_DATABASE to backfill once."
                log.info("Waiting for %s lookback hours (have %s). %s", need_hours, have, hint)
                logged_wait = True
            if board is not None:
                try:
                    board.maybe_refresh()
                except Exception as exc:
                    log.warning("HL board gather failed (will retry): %s", exc)
        except Exception as exc:
            log.exception("Live loop error (continuing): %s", exc)
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
        help="Strategy CSV. Default: rank_live.csv in the repo root (the one committed row).",
    )
    p.add_argument("--row", type=int, default=2, help="Excel row including header (2 = top by_live row)")
    p.add_argument("--db", type=Path, default=None, help="Local hour sqlite (default follows board=roi/pnl)")
    p.add_argument("--paper", action="store_true", help="Simulated fills (default). Same signals as --live.")
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
    excel_row = args.row
    raw = load_strategy(csv_path=csv_path, excel_row=excel_row)
    require_strategy_row(raw, str(csv_path))
    spec = spec_from_row(raw)
    if spec.get("engine") == "named" and spec.get("name") not in NAME_TO_ID:
        raise SystemExit(f"Unknown strategy name {spec.get('name')!r} in {csv_path}")
    board_kind = spec_board(spec)
    sqlite_path = args.db or default_live_sqlite_path(board_kind)
    if args.dry_run:
        mode = "dry"
    elif args.live:
        mode = "live"
    else:
        mode = "paper"
    log.info(
        "Loaded %s row %s | board=%s family=%s engine=%s name=%s | "
        "sharpe=%s ret=%s%% dd=%s%% trips=%s | step=%sh hold=%sh slots=%s lag=%s "
        "lookback=%s gross=%s%% use_lev=%s lev_x=%s size_mode=%s exposure=%s "
        "enter_top=%s mode=%s enter_th=%s exit_th=%s | run=%s",
        csv_path,
        excel_row,
        board_kind,
        spec.get("family"),
        spec.get("engine"),
        spec.get("name"),
        raw.get("sharpe"),
        raw.get("return_pct"),
        raw.get("max_dd_pct"),
        raw.get("round_trips"),
        spec.get("step_h"),
        spec.get("min_hold_h"),
        spec.get("slots"),
        spec.get("exec_lag"),
        spec.get("lookback"),
        spec.get("gross_pct"),
        spec.get("use_lev"),
        spec.get("lev_x"),
        spec.get("size_mode"),
        spec.get("exposure_mode"),
        spec.get("enter_top"),
        spec.get("mode"),
        spec.get("enter_th"),
        spec.get("exit_th"),
        raw.get("run_id"),
    )
    return run_live(
        spec,
        sqlite_path=sqlite_path,
        mode=mode,
        poll_s=args.poll,
        refresh_board=not (args.no_refresh or args.no_sync),
    )


if __name__ == "__main__":
    raise SystemExit(main())
