"""Resolve a read-only Neon URL without touching the collector repo."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from dotenv import load_dotenv

log = logging.getLogger("bagrank.dsn")

REPO = Path(__file__).resolve().parent.parent
COLLECTOR_ENV = (
    REPO.parent
    / "portfolio_data_platform"
    / "portfolio_data_collection"
    / ".env"
)

CONNECT_TIMEOUT_S = 60


def load_env() -> None:
    # This repo's NEON_BAGRANK always wins over a stale shell var.
    load_dotenv(REPO / ".env", override=True)
    if COLLECTOR_ENV.exists():
        load_dotenv(COLLECTOR_ENV, override=False)


def neon_endpoint_id(host: str) -> str:
    label = (host or "").split(".")[0]
    if label.endswith("-pooler"):
        label = label[: -len("-pooler")]
    return label if label.startswith("ep-") else ""


def pooler_dsn(url: str) -> str:
    """PgBouncer host for the same endpoint (fallback if direct TCP stalls)."""
    host = urlparse(url).hostname or ""
    if not host or "-pooler." in host:
        return url
    first, _, rest = host.partition(".")
    if not rest or first.endswith("-pooler"):
        return url
    return url.replace(host, f"{first}-pooler.{rest}", 1)


def redacted_dsn(url: str) -> str:
    """Host/db/ssl flags only — never user or password."""
    parsed = urlparse(url)
    host = parsed.hostname or "?"
    db = (parsed.path or "/").strip("/") or "?"
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    flags = []
    for key in ("sslmode", "gssencmode", "channel_binding"):
        if q.get(key):
            flags.append(f"{key}={q[key]}")
    if q.get("options"):
        flags.append(f"options={q['options']}")
    return f"{host}/{db} {' '.join(flags)}".strip()


def prepare_dsn(url: str) -> str:
    """Direct Neon host + SSL. Disable GSS so Windows libpq does not hang up."""
    out = url.strip().strip("'").strip('"')
    if not out:
        return ""
    if "-pooler." in out:
        out = out.replace("-pooler.", ".", 1)
    parsed = urlparse(out)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    q["sslmode"] = "require"
    q["gssencmode"] = "disable"
    q["channel_binding"] = "require"
    q["connect_timeout"] = str(CONNECT_TIMEOUT_S)
    epid = neon_endpoint_id(parsed.hostname or "")
    existing = q.get("options") or ""
    if epid and "endpoint=" not in existing.lower() and "endpoint%3d" not in existing.lower():
        q["options"] = f"endpoint={epid}"
    return urlunparse(parsed._replace(query=urlencode(q)))


def open_neon(dsn: str):
    import psycopg
    from psycopg.rows import dict_row

    prepared = prepare_dsn(dsn)
    if not prepared:
        raise ValueError("empty Neon DSN")
    last: Exception | None = None
    candidates = [prepared]
    pooled = pooler_dsn(prepared)
    if pooled != prepared:
        candidates.append(pooled)
    for idx, candidate in enumerate(candidates):
        host = urlparse(candidate).hostname or ""
        label = "pooler" if idx else "direct"
        log.info("Neon %s handshake %s", label, redacted_dsn(candidate))
        try:
            return psycopg.connect(
                candidate,
                row_factory=dict_row,
                autocommit=True,
                sslmode="require",
                gssencmode="disable",
                connect_timeout=CONNECT_TIMEOUT_S,
            )
        except ImportError:
            raise
        except Exception as exc:
            last = exc
            log.warning("Neon %s connect failed: %s", label, exc)
    raise last or RuntimeError("Neon connect failed")


def resolve_database_url(explicit: str = "") -> tuple[str, str]:
    """Return (prepared_dsn, source_label). Source is an env name or --dsn."""
    if explicit.strip():
        return prepare_dsn(explicit), "--dsn"
    load_env()
    raw = (os.environ.get("NEON_BAGRANK") or "").strip()
    if raw:
        return prepare_dsn(raw), "NEON_BAGRANK"
    return "", ""


def database_url(explicit: str = "") -> str:
    url, _src = resolve_database_url(explicit)
    return url


def default_sqlite_path() -> Path:
    override = (os.environ.get("BAGRANK_SQLITE") or "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else REPO / p
    return REPO / "data" / "bagrank" / "bagrank.sqlite"


def default_live_sqlite_path() -> Path:
    override = (os.environ.get("BAGRANK_LIVE_SQLITE") or "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else REPO / p
    return REPO / "data" / "bagrank" / "live.sqlite"
