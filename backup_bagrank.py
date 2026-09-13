"""CLI: incremental read-only backup of collector Neon → local sqlite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bagrank.backup import main

if __name__ == "__main__":
    raise SystemExit(main())
