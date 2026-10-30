"""Reuse only the existing fake runner, not its unrelated multi-thousand-line suites."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = a.source.read_text()
marker = '\nvoid ExecutionReclamationWaitsForActiveSourcesAndUnbinds()'
assert source.count(marker) == 1
fixture = source[:source.index(marker)]
assert fixture.count('class Runner : public TextModelRunner') == 1
assert fixture.rstrip().endswith('};')
a.output.write_text(fixture + '\n}  // namespace\n')
