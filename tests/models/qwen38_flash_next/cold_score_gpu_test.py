"""CTest wrapper: an expected native refusal is not an arbitrary failed test."""
import argparse
import json
import math
import subprocess
import sys


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', required=True)
    p.add_argument('--mode', choices=('off', 'capture', 'refusal'), required=True)
    a = p.parse_args()
    run = subprocess.run([a.binary, '--mode', a.mode], capture_output=True, text=True)
    print(run.stdout, end='')
    print(run.stderr, end='', file=sys.stderr)
    if run.returncode == 77:
        return 77
    records = []
    for line in run.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('schema') == 'gufo-full-logits-core-native-v1':
            records.append(value)
    require(len(records) == 1, 'missing or duplicate native result')
    r = records[0]
    require(r['complete'] is True and r['mode'] == a.mode, 'incomplete or wrong-mode result')
    if a.mode == 'refusal':
        require(run.returncode == 42 and r['status'] == 'expected_refusal', 'wrong refusal exit/status')
        e, x = r['required_bytes'], r['ceiling_bytes']
        require(e == x + 1 and e > 0, 'wrong refusal capacity witness')
        require(r['error'] == (f'--full-logits-api needs {e} bytes; snapshot ceiling is {x} bytes; '
                               'lower --context or --sessions, free memory, or restart without --full-logits-api'),
                'wrong refusal message')
    else:
        require(run.returncode == 0 and r['status'] == 'pass', 'native cell failed')
        if a.mode == 'off':
            require(r['off_geometry_equal'] is True and r['max_logit_rows'] == r['max_speculative'],
                    'OFF geometry changed')
        else:
            c = r['capture']
            require(all(c[k] is True for k in ('last_row_equal', 'pass_boundary_equal', 'cancel_recovery_equal')),
                    'strict capture witness failed')
            require(c['cancel_completed_passes'] == 1 and c['cancel_native_checks'] == 4,
                    'wrong cancellation counters')
            require(all(math.isfinite(c['l5'][k]) and c['l5'][k] >= 0 for k in ('max_abs', 'max_rel')),
                    'invalid L5 measurement')
    return 0


if __name__ == '__main__':
    sys.exit(main())
