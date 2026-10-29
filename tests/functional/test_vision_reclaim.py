import copy
import unittest

from vision_reclaim import FILL_REQUESTS, compare, digest, fill_body


class ReclaimEvidenceTests(unittest.TestCase):
    def setUp(self):
        response = {"choices": [{"message": {"content": "image"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 90,
                              "prompt_tokens_details": {"cached_tokens": 0}}}
        self.pressure = {"status": "requests_complete", "phase": "pressure",
                         "source_revision": "candidate", "binary_sha256": "binary",
                         "harness_sha256": "harness", "fixture_sha256": "fixture",
                         "image_body_sha256": "body",
                         "fill_body_sha256": [digest(fill_body(i)) for i in range(FILL_REQUESTS)],
                         "checks": {"image": {"response": response, "wall_ms": 100}},
                         "requests": [{"case": "image", "request_id": "r97"}]}
        self.reference = copy.deepcopy(self.pressure)
        self.reference.update(phase="reference", fill_body_sha256=[])
        self.log = "\n".join([
            "request=r97 event=received method=POST",
            "event=retention action=2 reason=1 tokens=4096",
            "event=image_backing_reclaim_before committed_bytes=33554432 rss_mib=40",
            "event=image_backing_reclaim_after released_bytes=25165824 committed_bytes=8388608 rss_mib=16",
            "event=image_claim_admitted outstanding_bytes=4155048128",
            "event=image_claim_completed resident_bytes=655360 outstanding_bytes=0",
            "request=r97 event=completed draft_rounds=3 draft_proposed=6",
            "event=vision_pool encodes=1 reencodes=0",
        ])

    def test_complete_physical_proof(self):
        result = compare(self.pressure, self.reference, self.log)
        self.assertEqual(result["released_bytes"], 25165824)
        self.assertEqual(result["status"], "passed")

    def test_rejects_missing_or_false_memory_evidence(self):
        for original, replacement in [
            ("event=retention action=2 reason=1", "event=retention action=0 reason=0"),
            ("rss_mib=16", "rss_mib=40"),
            ("committed_bytes=8388608", "committed_bytes=33554432"),
            ("resident_bytes=655360", "resident_bytes=0"),
            ("outstanding_bytes=0", "outstanding_bytes=3471376384"),
            ("encodes=1", "encodes=0"),
            ("draft_proposed=6", "draft_proposed=0"),
        ]:
            with self.subTest(original=original), self.assertRaises(AssertionError):
                compare(self.pressure, self.reference, self.log.replace(original, replacement))

    def test_rejects_output_and_history_changes(self):
        self.reference["checks"]["image"]["response"]["choices"][0]["message"]["content"] = "changed"
        with self.assertRaises(AssertionError):
            compare(self.pressure, self.reference, self.log)
        self.reference = copy.deepcopy(self.pressure)
        self.reference.update(phase="reference", fill_body_sha256=[])
        self.pressure["fill_body_sha256"].pop()
        with self.assertRaises(AssertionError):
            compare(self.pressure, self.reference, self.log)

    def test_rejects_mismatched_binary(self):
        self.reference["binary_sha256"] = "other"
        with self.assertRaises(AssertionError):
            compare(self.pressure, self.reference, self.log)

    def test_fixed_unique_fill(self):
        self.assertEqual(FILL_REQUESTS, 96)
        self.assertEqual(len({digest(fill_body(i)) for i in range(96)}), 96)
        self.assertEqual(fill_body(0)["messages"][0]["content"][0]["text"].count(" word"), 4096)
        self.assertTrue(fill_body(0)["cache_prompt"])
        self.assertEqual(fill_body(0)["max_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
