"""CPU negatives for the native-test receipt wrapper; no GPU evidence."""
import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import cold_score_gpu_test as wrapper


class Contract(unittest.TestCase):
    def run_record(self, mode, fields, code=0):
        value = dict(schema='gufo-full-logits-core-native-v1', mode=mode,
                     complete=True, status='pass', **fields)
        run = subprocess.CompletedProcess([], code, json.dumps(value) + '\n', '')
        with patch.object(sys, 'argv', ['wrapper', '--binary', 'unused', '--mode', mode]), \
             patch.object(wrapper.subprocess, 'run', return_value=run), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return wrapper.main()

    def test_off(self):
        self.assertEqual(self.run_record('off', dict(off_geometry_equal=True, max_logit_rows=1, max_speculative=1)), 0)

    def test_capture_and_measured_difference(self):
        fields = dict(capture=dict(last_row_equal=True, pass_boundary_equal=True,
                      cancel_recovery_equal=True, cancel_completed_passes=1, cancel_native_checks=4,
                      l5=dict(max_abs=1e-5, max_rel=2e-4, byte_equal=False, argmax_equal=True)))
        self.assertEqual(self.run_record('capture', fields), 0)
        fields['capture']['l5']['max_abs'] = float('nan')
        with self.assertRaises(AssertionError):
            self.run_record('capture', fields)

    def test_refusal_requires_exact_message_and_nonzero(self):
        fields = dict(required_bytes=4096, ceiling_bytes=4095,
                      error='--full-logits-api needs 4096 bytes; snapshot ceiling is 4095 bytes; '
                            'lower --context or --sessions, free memory, or restart without --full-logits-api')
        value = dict(schema='gufo-full-logits-core-native-v1', mode='refusal',
                     complete=True, status='expected_refusal', **fields)
        for code, message, valid in ((42, fields['error'], True), (0, fields['error'], False), (42, 'wrong', False)):
            value['error'] = message
            run = subprocess.CompletedProcess([], code, json.dumps(value) + '\n', '')
            with patch.object(sys, 'argv', ['wrapper', '--binary', 'unused', '--mode', 'refusal']), \
                 patch.object(wrapper.subprocess, 'run', return_value=run), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if valid:
                    self.assertEqual(wrapper.main(), 0)
                else:
                    with self.assertRaises(AssertionError):
                        wrapper.main()

    def test_incomplete(self):
        run = subprocess.CompletedProcess([], 0, '{"stage":"readiness"}\n', '')
        with patch.object(sys, 'argv', ['wrapper', '--binary', 'unused', '--mode', 'capture']), \
             patch.object(wrapper.subprocess, 'run', return_value=run), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(AssertionError):
                wrapper.main()


if __name__ == '__main__':
    unittest.main()
