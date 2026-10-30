"""Reuse only the existing fake runner, not its unrelated multi-thousand-line suites."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
# Compatibility for old build trees; current targets include the shared header.
a.output.write_text(a.source.with_name('component_text_cache_fixture.hpp').read_text())
