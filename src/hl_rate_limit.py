"""Hyperliquid REST rate limits (2026 docs): 1200 weight/min per IP, no daily info cap."""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import types
from pathlib import Path
from typing import Any

from hyperliquid.api import API
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.error import ClientError

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

# https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits
IP_WEIGHT_PER_MINUTE = 1200
# Stay under budget so browser + 2–3 bots on same IP can coexist
IP_WEIGHT_RESERVE = 250


def _is_transient_network_error(exc: BaseException) -> bool:
    """True for dropped connections / timeouts that are safe to retry."""
    if requests is not None and isinstance(
        exc,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    name = type(exc).__name__
    if name in (
        "ConnectionError",
        "Timeout",
        "ProtocolError",
        "RemoteDisconnected",
        "ReadTimeoutError",
        "ConnectTimeoutError",
    ):
        return True
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "connection aborted",
            "remote end closed",
            "remotely disconnected",
            "connection reset",
            "timed out",
            "temporarily unavailable",
            "name resolution",
            "failed to establish",
        )
    )

WEIGHT_LIGHT_INFO = 2  # l2Book, allMids, clearinghouseState, orderStatus, ...
WEIGHT_DEFAULT_INFO = 20
WEIGHT_EXCHANGE_BASE = 1


def info_request_weight(req_type: str, response_item_count: int = 0) -> int:
    if req_type in (
        "l2Book",
        "allMids",
        "clearinghouseState",
        "orderStatus",
        "spotClearinghouseState",
        "exchangeStatus",
    ):
        w = WEIGHT_LIGHT_INFO
    elif req_type == "userRole":
        w = 60
    elif req_type == "candleSnapshot":
        w = WEIGHT_DEFAULT_INFO + (response_item_count // 60)
    elif req_type in (
        "recentTrades",
        "historicalOrders",
        "userFills",
        "userFillsByTime",
        "fundingHistory",
        "userFunding",
        "userNonFundingLedgerUpdates",
        "nonUserFundingUpdates",
    ):
        w = WEIGHT_DEFAULT_INFO + (response_item_count // 20)
    else:
        w = WEIGHT_DEFAULT_INFO
    return w


class SharedIpBudget:
    """Optional cross-process budget file (same PC + IP = shared 1200/min)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> tuple[int, int]:
        if not self.path.exists():
            return 0, 0
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return int(data.get("minute", 0)), int(data.get("used", 0))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return 0, 0

    def _write(self, minute: int, used: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"minute": minute, "used": used}), encoding="utf-8")
        tmp.replace(self.path)

    def acquire(self, weight: int, logger: logging.Logger) -> None:
        while True:
            with self._lock:
                minute_bucket = int(time.time()) // 60
                stored_minute, used = self._read()
                if stored_minute != minute_bucket:
                    used = 0
                cap = IP_WEIGHT_PER_MINUTE - IP_WEIGHT_RESERVE
                if used + weight <= cap:
                    self._write(minute_bucket, used + weight)
                    return
                wait_s = 60 - (time.time() % 60) + 0.05
            logger.warning(
                "IP weight budget %s/%s used; waiting %.0fs (shared limit with browser/other bots)",
                used,
                IP_WEIGHT_PER_MINUTE,
                wait_s,
            )
            time.sleep(min(wait_s, 5.0))


class RequestGuard:
    def __init__(
        self,
        min_interval_s: float = 0.12,
        max_429_retries: int = 8,
        logger: logging.Logger | None = None,
        shared_budget: SharedIpBudget | None = None,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.max_429_retries = max_429_retries
        self.log = logger or logging.getLogger(__name__)
        self.shared = shared_budget
        self._lock = threading.Lock()
        self._last_send = 0.0
        self._local_minute = 0
        self._local_used = 0

    def _acquire_weight(self, weight: int) -> None:
        if self.shared is not None:
            self.shared.acquire(weight, self.log)
            return
        with self._lock:
            minute_bucket = int(time.time()) // 60
            if minute_bucket != self._local_minute:
                self._local_minute = minute_bucket
                self._local_used = 0
            cap = IP_WEIGHT_PER_MINUTE - IP_WEIGHT_RESERVE
            while self._local_used + weight > cap:
                wait_s = 60 - (time.time() % 60) + 0.05
                self.log.warning(
                    "Local IP weight %s/%s; pausing %.0fs",
                    self._local_used,
                    IP_WEIGHT_PER_MINUTE,
                    wait_s,
                )
                time.sleep(min(wait_s, 5.0))
                minute_bucket = int(time.time()) // 60
                if minute_bucket != self._local_minute:
                    self._local_minute = minute_bucket
                    self._local_used = 0
            self._local_used += weight

    def wait_turn(self, weight: int = 1) -> None:
        self._acquire_weight(weight)
        with self._lock:
            now = time.monotonic()
            wait_s = self._last_send + self.min_interval_s - now
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_send = time.monotonic()


class ThrottledInfo(Info):
    def __init__(
        self,
        base_url: str | None = None,
        skip_ws: bool = False,
        meta: Any = None,
        spot_meta: Any = None,
        perp_dexs: Any = None,
        timeout: float | None = None,
        *,
        guard: RequestGuard,
    ) -> None:
        self._hl_guard = guard
        super().__init__(base_url, skip_ws, meta, spot_meta, perp_dexs, timeout)

    def post(self, url_path: str, payload: Any = None) -> Any:
        payload = payload or {}
        req_type = payload.get("type", "")
        est = info_request_weight(req_type, 0)
        last_err: BaseException | None = None
        for attempt in range(self._hl_guard.max_429_retries):
            self._hl_guard.wait_turn(est)
            try:
                result = super().post(url_path, payload)
                if req_type == "candleSnapshot" and isinstance(result, list):
                    extra = info_request_weight(req_type, len(result)) - WEIGHT_DEFAULT_INFO
                    if extra > 0:
                        self._hl_guard._acquire_weight(extra)
                return result
            except ClientError as exc:
                last_err = exc
                if exc.status_code == 429:
                    delay = min(45.0, 1.0 * (2**attempt) + random.uniform(0, 0.3))
                    self._hl_guard.log.warning(
                        "Hyperliquid 429 (%s), backoff %.1fs",
                        req_type or url_path,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise
            except Exception as exc:
                last_err = exc
                if _is_transient_network_error(exc):
                    delay = min(30.0, 1.5 * (2**attempt) + random.uniform(0, 0.5))
                    self._hl_guard.log.warning(
                        "Hyperliquid network glitch (%s) attempt %s/%s — retry in %.1fs: %s",
                        req_type or url_path,
                        attempt + 1,
                        self._hl_guard.max_429_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                raise
        assert last_err is not None
        raise last_err


def bind_throttled_exchange_post(exchange: Exchange, guard: RequestGuard) -> None:
    def _post(self: Exchange, url_path: str, payload: Any = None) -> Any:
        last_err: BaseException | None = None
        for attempt in range(guard.max_429_retries):
            guard.wait_turn(WEIGHT_EXCHANGE_BASE)
            try:
                return API.post(self, url_path, payload)
            except ClientError as exc:
                last_err = exc
                if exc.status_code == 429:
                    delay = min(45.0, 1.0 * (2**attempt) + random.uniform(0, 0.3))
                    guard.log.warning("Hyperliquid 429 on exchange, backoff %.1fs", delay)
                    time.sleep(delay)
                    continue
                raise
            except Exception as exc:
                last_err = exc
                if _is_transient_network_error(exc):
                    delay = min(30.0, 1.5 * (2**attempt) + random.uniform(0, 0.5))
                    guard.log.warning(
                        "Hyperliquid network glitch (exchange) attempt %s/%s — retry in %.1fs: %s",
                        attempt + 1,
                        guard.max_429_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                raise
        assert last_err is not None
        raise last_err

    exchange.post = types.MethodType(_post, exchange)
    exchange.info = ThrottledInfo(
        exchange.base_url,
        True,
        None,
        None,
        None,
        exchange.timeout,
        guard=guard,
    )


def default_shared_budget() -> SharedIpBudget:
    env_path = os.environ.get("HL_RATE_BUDGET_FILE")
    if env_path:
        return SharedIpBudget(Path(env_path))
    return SharedIpBudget(
        Path(os.environ.get("TEMP", ".")) / "hyperliquid_rsi_bot_ip_budget.json"
    )
