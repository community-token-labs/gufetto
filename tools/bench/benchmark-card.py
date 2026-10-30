#!/usr/bin/env python3
"""Render a Gufo-only Markdown benchmark card to stdout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.benchmark_card import main

if __name__ == "__main__":
    raise SystemExit(main())
