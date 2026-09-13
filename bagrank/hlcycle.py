"""Build the bag-rank board live from Hyperliquid (same tally as the collector).

Database/Neon is not used. Hours are stored locally so the kernels have history.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .store import connect, upsert_rows
from .timeutil import floor_hour, to_iso

log = logging.getLogger("bagrank.hlcycle")

LEADERBOARD_URLS = (
    "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard",
    "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard.json",
)
WINDOW_ALIAS = {"month": "perpMonth", "week": "perpWeek", "day": "perpDay"}
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
BASKET_SIZE = 200
RANK_WINDOW = "week"
MIN_NOTIONAL = 50.0
SNAP_GAP_S = 0.12
BOARD_REFRESH_S = 3600.0

_CLEARINGHOUSE_META_KEYS = {
    "assetPositions",
    "marginSummary",
    "crossMarginSummary",
    "withdrawable",
    "time",
    "agentAddress",
    "cumLedger",
    "perpDexStates",
}


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def coin_key(coin: str, dex: str | None = None) -> str:
    raw = str(coin or "").strip()
    if ":" in raw:
        left, right = raw.split(":", 1)
        dex_s = left.strip()
        sym = right.strip()
        return f"{dex_s}:{sym}" if dex_s else sym
    dex_s = (dex or "").strip()
    return f"{dex_s}:{raw}" if dex_s else raw


def in_scope(coin: str, dex_scope: str = "include") -> bool:
    scope = (dex_scope or "include").strip().lower()
    dex = str(coin).split(":", 1)[0].strip() if ":" in str(coin) else ""
    if scope == "native" and dex:
        return False
    if scope == "xyz_only" and not dex:
        return False
    return True


def tally_holds(
    snaps: list[dict[str, Any]],
    *,
    dex_scope: str = "include",
    min_notional_usd: float = MIN_NOTIONAL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Same majority rank as the collector: (-wallets, -hold_pct, coin)."""
    ok: list[dict[str, Any]] = []
    empty = 0
    errors = 0
    for snap in snaps:
        if not snap.get("ok"):
            errors += 1
            continue
        ok.append(snap)
        if not snap.get("positions"):
            empty += 1
    n_ok = len(ok)
    long_n: dict[str, int] = defaultdict(int)
    short_n: dict[str, int] = defaultdict(int)
    long_lev: dict[str, list[int]] = defaultdict(list)
    short_lev: dict[str, list[int]] = defaultdict(list)
    long_conv: dict[str, list[float]] = defaultdict(list)
    short_conv: dict[str, list[float]] = defaultdict(list)
    long_ntl: dict[str, float] = defaultdict(float)
    short_ntl: dict[str, float] = defaultdict(float)
    for snap in ok:
        seen: set[str] = set()
        for pos in snap.get("positions") or []:
            coin = str(pos.get("coin") or "")
            if not coin or not in_scope(coin, dex_scope):
                continue
            if float(pos.get("notional") or 0) < min_notional_usd:
                continue
            if coin in seen:
                continue
            seen.add(coin)
            lev = max(1, int(pos.get("leverage") or 1))
            conv = abs(float(pos.get("conviction") or 0))
            ntl = float(pos.get("notional") or 0)
            if pos.get("side") == "long":
                long_n[coin] += 1
                long_lev[coin].append(lev)
                long_conv[coin].append(conv)
                long_ntl[coin] += ntl
            else:
                short_n[coin] += 1
                short_lev[coin].append(lev)
                short_conv[coin].append(conv)
                short_ntl[coin] += ntl
    rows: list[dict[str, Any]] = []
    for coin in set(long_n) | set(short_n):
        ln = int(long_n.get(coin, 0))
        sn = int(short_n.get(coin, 0))
        if ln >= sn and ln > 0:
            side, n, levs, convs, ntl = "long", ln, long_lev[coin], long_conv[coin], long_ntl[coin]
        else:
            side, n, levs, convs, ntl = "short", sn, short_lev[coin], short_conv[coin], short_ntl[coin]
        both = ln + sn
        rows.append(
            {
                "coin": coin,
                "side": side,
                "wallets": n,
                "hold_pct": (n / n_ok) if n_ok else 0.0,
                "agreement": (n / both) if both else 0.0,
                "long_n": ln,
                "short_n": sn,
                "median_leverage": int(round(float(statistics.median(levs)))) if levs else 1,
                "mean_leverage": float(sum(levs) / len(levs)) if levs else 1.0,
                "avg_conviction": float(sum(convs) / len(convs)) if convs else 0.0,
                "notional_usd": ntl,
            }
        )
    rows.sort(key=lambda r: (-r["wallets"], -r["hold_pct"], r["coin"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    stats = {
        "snapped": len(snaps),
        "ok": n_ok,
        "empty": empty,
        "errors": errors,
        "with_pos": n_ok - empty,
        "coins": len(rows),
    }
    return rows, stats


def parse_leaderboard(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw_rows = (
            payload.get("leaderboardRows")
            or payload.get("leaderboard_rows")
            or payload.get("leaderboard")
            or []
        )
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        raw_rows = []
    out: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        addr = str(row.get("ethAddress") or row.get("eth_address") or "").strip().lower()
        if not addr.startswith("0x") or len(addr) != 42:
            continue
        windows: dict[str, dict[str, float]] = {}
        raw = row.get("windowPerformances") or row.get("window_performances") or []
        items = raw.items() if isinstance(raw, dict) else raw
        for item in items:
            name = ""
            block: Any = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name, block = str(item[0]), item[1]
            elif isinstance(item, dict):
                name = str(item.get("window") or item.get("name") or "")
                block = item
            if name and isinstance(block, dict):
                windows[name] = {
                    "pnl": _f(block.get("pnl")),
                    "roi": _f(block.get("roi")),
                    "volume": _f(block.get("vlm", block.get("volume"))),
                }
        out.append(
            {
                "address": addr,
                "account_value": _f(row.get("accountValue") or row.get("account_value")),
                "windows": windows,
            }
        )
    return out


def shortlist_top_roi(rows: list[dict[str, Any]], *, rank_window: str = RANK_WINDOW, limit: int = BASKET_SIZE) -> list[str]:
    key = WINDOW_ALIAS.get(str(rank_window).strip(), str(rank_window).strip())
    scored: list[tuple[float, str]] = []
    for row in rows:
        block = (row.get("windows") or {}).get(key) or (row.get("windows") or {}).get(rank_window)
        if not isinstance(block, dict):
            continue
        scored.append((float(block.get("roi") or 0), str(row["address"])))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [addr for _roi, addr in scored[: max(1, int(limit))]]


def fetch_leaderboard(timeout_s: float = 90.0) -> list[dict[str, Any]]:
    last: Exception | None = None
    for url in LEADERBOARD_URLS:
        try:
            log.info("Downloading HL leaderboard %s", url)
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            rows = parse_leaderboard(resp.json())
            if rows:
                return rows
        except Exception as exc:
            last = exc
            log.warning("Leaderboard fetch failed (%s): %s", url, exc)
    raise RuntimeError(f"Could not download HL leaderboard: {last}")


def _info_post(info: Any, body: dict[str, Any]) -> Any:
    if info is None:
        resp = requests.post(HL_INFO_URL, json=body, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    post = getattr(info, "post", None)
    if post is None:
        raise RuntimeError("HL info client has no post()")
    try:
        return post("/info", body)
    except TypeError:
        return post(body)


def _iter_states(raw: Any, default_dex: str = "") -> list[tuple[str, dict[str, Any]]]:
    tagged: list[tuple[str, dict[str, Any]]] = []
    seen: set[int] = set()

    def add(dex: str, state: dict[str, Any]) -> None:
        if "assetPositions" not in state:
            return
        sid = id(state)
        if sid in seen:
            return
        seen.add(sid)
        tagged.append((str(dex or ""), state))

    def walk(obj: Any, dex_hint: str) -> None:
        if obj is None:
            return
        if (
            isinstance(obj, (list, tuple))
            and len(obj) == 2
            and isinstance(obj[0], str)
            and isinstance(obj[1], dict)
            and "assetPositions" in obj[1]
        ):
            add(obj[0], obj[1])
            return
        if isinstance(obj, dict):
            if "assetPositions" in obj:
                add(dex_hint, obj)
            for key, val in obj.items():
                if key == "assetPositions":
                    continue
                next_dex = dex_hint
                if (
                    isinstance(key, str)
                    and key not in _CLEARINGHOUSE_META_KEYS
                    and isinstance(val, dict)
                    and "assetPositions" in val
                ):
                    next_dex = key
                walk(val, next_dex)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item, dex_hint)

    walk(raw, default_dex or "")
    return tagged


def _equity(states: list[tuple[str, dict[str, Any]]]) -> float:
    best = 0.0
    for _dex, state in states:
        for key in ("marginSummary", "crossMarginSummary"):
            block = state.get(key) or {}
            if isinstance(block, dict):
                best = max(best, _f(block.get("accountValue")))
    return best


def _positions(states: list[tuple[str, dict[str, Any]]], account_value: float) -> list[dict[str, Any]]:
    raw: list[tuple[str, str, float, float, float | None, int]] = []
    seen: set[str] = set()
    for dex, state in states:
        for ap in state.get("assetPositions") or []:
            pos = ap.get("position", ap) if isinstance(ap, dict) else {}
            szi = _f(pos.get("szi"))
            if abs(szi) < 1e-12:
                continue
            coin = coin_key(str(pos.get("coin") or ""), dex)
            if not coin or coin in seen:
                continue
            seen.add(coin)
            lev_raw = pos.get("leverage") or {}
            if isinstance(lev_raw, dict):
                lev = max(1, int(_f(lev_raw.get("value"), 1)))
            else:
                lev = max(1, int(_f(lev_raw, 1)))
            notional = abs(_f(pos.get("positionValue")))
            if notional <= 0:
                entry = _f(pos.get("entryPx"))
                notional = abs(szi) * entry if entry > 0 else 0.0
            raw.append((coin, "long" if szi > 0 else "short", abs(szi), notional, _f(pos.get("entryPx")) or None, lev))
    equity = max(account_value, sum(p[3] for p in raw), 1e-9)
    out: list[dict[str, Any]] = []
    for coin, side, size, notional, entry_px, lev in raw:
        signed = 1.0 if side == "long" else -1.0
        out.append(
            {
                "coin": coin,
                "side": side,
                "size": size,
                "notional": notional,
                "entry_px": entry_px,
                "leverage": lev,
                "conviction": (notional / equity) * signed,
            }
        )
    return out


def snapshot_wallet(info: Any, address: str, *, flags: dict[str, bool] | None = None) -> dict[str, Any]:
    addr = address.lower()
    now = time.time()
    flags = flags if flags is not None else {"all_dexes_ok": True}
    try:
        states: list[tuple[str, dict[str, Any]]] = []
        if flags.get("all_dexes_ok", True):
            try:
                raw = _info_post(info, {"type": "clearinghouseState", "user": addr, "dex": "ALL_DEXES"})
                states = _iter_states(raw)
            except Exception:
                flags["all_dexes_ok"] = False
                states = []
        if not states:
            seen: set[int] = set()
            for dex in ("", "xyz"):
                body: dict[str, Any] = {"type": "clearinghouseState", "user": addr}
                if dex:
                    body["dex"] = dex
                try:
                    raw = _info_post(info, body)
                except Exception:
                    continue
                for item in _iter_states(raw, default_dex=dex):
                    sid = id(item[1])
                    if sid in seen:
                        continue
                    seen.add(sid)
                    states.append(item)
        equity = _equity(states)
        positions = _positions(states, equity)
        return {"address": addr, "positions": positions, "ok": True, "error": "", "fetched_at": now}
    except Exception as exc:
        return {"address": addr, "positions": [], "ok": False, "error": str(exc)[:300], "fetched_at": now}


def fetch_ctx_map(info: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dex in ("", "xyz"):
        body: dict[str, Any] = {"type": "metaAndAssetCtxs"}
        if dex:
            body["dex"] = dex
        try:
            raw = _info_post(info, body)
        except Exception as exc:
            log.warning("metaAndAssetCtxs dex=%r failed: %s", dex or "native", exc)
            continue
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        meta = raw[0] if isinstance(raw[0], dict) else {}
        ctxs = raw[1] if isinstance(raw[1], list) else []
        universe = meta.get("universe") or []
        for i, asset in enumerate(universe):
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name") or "").strip()
            if not name:
                continue
            ctx = ctxs[i] if i < len(ctxs) and isinstance(ctxs[i], dict) else {}
            key = coin_key(name, dex or None)
            out[key] = ctx
            out[name] = ctx
    return out


def gather_cycle(info: Any, *, basket: int = BASKET_SIZE) -> dict[str, Any]:
    started = time.time()
    cycle_ts = to_iso(floor_hour(datetime.now(timezone.utc)))
    rows = fetch_leaderboard()
    addrs = shortlist_top_roi(rows, limit=basket)
    if not addrs:
        raise RuntimeError("HL leaderboard shortlist empty")
    log.info("Snapshotting %s top-week wallets from Hyperliquid", len(addrs))
    books: list[dict[str, Any]] = []
    flags = {"all_dexes_ok": True}
    for i, addr in enumerate(addrs, start=1):
        books.append(snapshot_wallet(info, addr, flags=flags))
        if i == len(addrs):
            log.info("Snapped %s/%s", i, len(addrs))
        time.sleep(SNAP_GAP_S)
    index_rows, stats = tally_holds(books)
    ctxs = fetch_ctx_map(info)
    prices: list[dict[str, Any]] = []
    now_iso = to_iso(datetime.now(timezone.utc))
    for row in index_rows:
        coin = str(row["coin"])
        ctx = ctxs.get(coin) or ctxs.get(coin.split(":")[-1]) or {}
        prices.append(
            {
                "cycle_ts": cycle_ts,
                "coin": coin,
                "mark_px": _f(ctx.get("markPx")),
                "mid_px": _f(ctx.get("midPx")),
                "oracle_px": _f(ctx.get("oraclePx")),
                "funding": _f(ctx.get("funding")),
                "open_interest": _f(ctx.get("openInterest")),
                "prev_day_px": _f(ctx.get("prevDayPx")),
                "day_ntl_vlm": _f(ctx.get("dayNtlVlm")),
                "premium": _f(ctx.get("premium")),
                "source": "hl",
                "fetched_at": now_iso,
            }
        )
    ok = int(stats["ok"])
    listed = len(addrs)
    coverage = (ok / listed) if listed else 0.0
    return {
        "cycle_ts": cycle_ts,
        "listed": listed,
        "snapped_ok": ok,
        "snapped_err": int(stats["errors"]),
        "empty_books": int(stats["empty"]),
        "coverage": round(coverage, 6),
        "duration_s": round(time.time() - started, 2),
        "meta_index": index_rows,
        "coin_prices": prices,
        "status": "ok" if coverage >= 0.7 else ("partial" if ok else "failed"),
    }


def prune_old_hours(conn: Any, *, venue: str, keep_hours: int) -> int:
    """Keep only the newest keep_hours cycle timestamps."""
    keep = max(1, int(keep_hours))
    rows = conn.execute(
        """
        SELECT DISTINCT cycle_ts FROM collector_runs
        WHERE venue = ?
        ORDER BY cycle_ts DESC
        """,
        (venue,),
    ).fetchall()
    drop = [str(r[0]) for r in rows[keep:]]
    if not drop:
        return 0
    placeholders = ",".join("?" * len(drop))
    args = (*drop, venue)
    for table in (
        "meta_index",
        "coin_prices",
        "collector_runs",
        "wallet_books",
        "wallet_positions",
        "cohort_members",
    ):
        conn.execute(
            f"DELETE FROM {table} WHERE cycle_ts IN ({placeholders}) AND venue = ?",
            args,
        )
    return len(drop)


def persist_cycle(
    sqlite_path: Path,
    payload: dict[str, Any],
    *,
    venue: str = "hyperliquid",
    keep_hours: int | None = None,
) -> None:
    cycle = str(payload["cycle_ts"])
    conn = connect(sqlite_path)
    try:
        conn.execute("DELETE FROM meta_index WHERE cycle_ts = ? AND venue = ?", (cycle, venue))
        conn.execute("DELETE FROM coin_prices WHERE cycle_ts = ? AND venue = ?", (cycle, venue))
        now = to_iso(datetime.now(timezone.utc))
        upsert_rows(
            conn,
            "collector_runs",
            [
                {
                    "cycle_ts": cycle,
                    "started_at": now,
                    "finished_at": now,
                    "status": payload.get("status") or "ok",
                    "listed": int(payload.get("listed") or 0),
                    "snapped_ok": int(payload.get("snapped_ok") or 0),
                    "snapped_err": int(payload.get("snapped_err") or 0),
                    "empty_books": int(payload.get("empty_books") or 0),
                    "coverage": payload.get("coverage"),
                    "leaderboard_refreshed": 1,
                    "error": "",
                    "duration_s": payload.get("duration_s"),
                    "cohort": f"top{BASKET_SIZE}_{RANK_WINDOW}",
                }
            ],
            venue=venue,
        )
        meta = [dict(r, cycle_ts=cycle) for r in (payload.get("meta_index") or [])]
        if meta:
            upsert_rows(conn, "meta_index", meta, venue=venue)
        prices = list(payload.get("coin_prices") or [])
        if prices:
            upsert_rows(conn, "coin_prices", prices, venue=venue)
        if keep_hours is not None:
            prune_old_hours(conn, venue=venue, keep_hours=keep_hours)
        conn.commit()
    finally:
        conn.close()


class LiveBoard:
    def __init__(
        self,
        sqlite_path: Path,
        info: Any,
        *,
        refresh_s: float = BOARD_REFRESH_S,
        keep_hours: int = 6,
    ) -> None:
        self.sqlite_path = sqlite_path
        self.info = info
        self.refresh_s = float(refresh_s)
        self.keep_hours = max(1, int(keep_hours))
        self.last_at = 0.0

    def maybe_refresh(self) -> bool:
        now = time.time()
        hour_key = to_iso(floor_hour(datetime.now(timezone.utc)))
        if getattr(self, "last_hour", "") == hour_key and self.last_at > 0:
            return False
        payload = gather_cycle(self.info)
        persist_cycle(self.sqlite_path, payload, keep_hours=self.keep_hours)
        self.last_at = now
        self.last_hour = str(payload.get("cycle_ts") or hour_key)
        log.info(
            "HL board hour %s | coins=%s coverage=%.0f%% snapped=%s/%s in %.1fs",
            payload["cycle_ts"],
            len(payload.get("meta_index") or []),
            float(payload.get("coverage") or 0) * 100.0,
            payload.get("snapped_ok"),
            payload.get("listed"),
            payload.get("duration_s"),
        )
        return True
