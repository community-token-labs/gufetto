"""CPU checks for the focused HTTP harness, not model/admission evidence."""

import base64
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from vision_admission import body, exercise


class VisionAdmissionHarnessTests(unittest.TestCase):
    def test_request_keeps_text_before_images_and_explicit_controls(self):
        value = body(["one", "two"], "prefix")
        self.assertEqual(value["messages"][0]["content"][0],
                         {"type": "text", "text": "prefix"})
        self.assertEqual(value["messages"][0]["content"][2]["image_url"],
                         {"url": "two", "detail": "auto"})
        self.assertEqual(value["max_tokens"], 16)
        self.assertFalse(value["cache_prompt"])

    def run_budget(self, policy, accept_overflow=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # encoded_variant only inserts before the final PNG trailer; these
            # stand-ins are sufficient for testing request assembly, not pixels.
            for index in range(3):
                (root / f"screenshot-256x256-{index}.png").write_bytes(bytes(32))
            report = {"fixtures": {}, "checks": {}}
            recorder = SimpleNamespace(case="")
            limit = 1 if policy == "cap64" else 2

            def fake_request(base, endpoint, payload, *, timeout=600):
                self.assertEqual(timeout, 600)
                images = sum(part.get("type") == "image_url"
                             for message in payload["messages"]
                             if isinstance(message["content"], list)
                             for part in message["content"])
                if images > limit and not accept_overflow:
                    raise urllib.error.HTTPError(
                        base, 400, "bad request", {}, io.BytesIO(b'vision budget exceeded'))
                tokens = 17 + 64 * images
                if endpoint.endswith("/input_tokens"):
                    return {"object": "response.input_tokens", "input_tokens": tokens}, 1.0
                return {"usage": {"prompt_tokens": tokens, "completion_tokens": 1},
                        "choices": [{"message": {"content": "OK"},
                                     "finish_reason": "stop"}]}, 1.0

            with patch("vision_admission.request", fake_request):
                exercise("http://127.0.0.1", root, policy, report, recorder)
            return report

    def test_cap_and_pool_include_history_encoding_and_recovery(self):
        for policy in ("cap64", "pool128"):
            with self.subTest(policy=policy):
                checks = self.run_budget(policy)["checks"]
                for case in ("never-fit-count", "never-fit", "history-overflow",
                             "alternate-encoding-overflow"):
                    self.assertEqual(checks[case]["status"], 400)
                self.assertIn("recovery", checks)

    def test_default_control_and_explicit_deadline(self):
        for control in (False, True):
            report = {"fixtures": {}, "checks": {}}
            recorder = SimpleNamespace(case="")

            def fixture(path):
                return (b"large" if "4096x4096" in path.name else b"small") + path.name.encode()

            def fake_request(base, endpoint, payload, *, timeout):
                self.assertEqual(timeout, 3600)
                rows = sum(16384 if base64.b64decode(part["image_url"]["url"].split(",", 1)[1]).startswith(b"large") else 64
                           for part in payload["messages"][0]["content"]
                           if part.get("type") == "image_url")
                if rows > 65536 and not control:
                    raise urllib.error.HTTPError(base, 400, "bad request", {},
                                                 io.BytesIO(b"vision budget exceeded"))
                if endpoint.endswith("/input_tokens"):
                    return {"object": "response.input_tokens", "input_tokens": rows + 17}, 1.0
                return {"usage": {"prompt_tokens": rows + 17, "completion_tokens": 1,
                                  "prompt_tokens_details": {"cached_tokens": 0}},
                        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}, 1.0

            with patch.object(Path, "read_bytes", fixture), patch("vision_admission.request", fake_request):
                exercise("http://127.0.0.1", Path("fixtures"), "default", report, recorder,
                         request_timeout=3600, independent_control=control)
            self.assertEqual("control_exclusion" in report, control)
            self.assertEqual("five-max" in report["checks"], not control)
            self.assertIn("reencode-A-replay", report["checks"])

    def test_unexpected_overflow_acceptance_fails(self):
        with self.assertRaisesRegex(AssertionError, "never-fit request accepted"):
            self.run_budget("cap64", accept_overflow=True)


if __name__ == "__main__":
    unittest.main()
