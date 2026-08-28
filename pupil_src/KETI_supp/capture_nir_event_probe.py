#!/usr/bin/env python3
"""KETI support entry point for the live Eye0/DAVIS checkerboard probe."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_nir_event_checkerboard_probe import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
