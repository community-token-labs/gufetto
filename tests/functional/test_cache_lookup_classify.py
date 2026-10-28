"""CPU-only classification, anti-excuse and receipt sanitization checks."""
from copy import deepcopy
import unittest

from cache_lookup import ProbeError, generation_metrics, lookup_metrics
from cache_lookup_classify import Evidence, classify, fixture_results, summarize


def pair(matched=531, cached=0, version=10, admitted=10, failures=0):
    return {"case": "pair", "lookup": {"matched_tokens": matched, "index_version": version,
                                     "observation_id": 1},
            "actual": {"cached_tokens": cached, "component_cache": {
                "admission_index_version": admitted, "acquisition_id": 2,
                "restore_attempts": 1, "restore_failures": failures}}}


def evidence(version=10, retirement=None, state="published", finish=10):
    lines = [f"schema=cache-lookup-v1 event=lookup lookup_id=1 entry_id=7 index_version={version} "
             f"matched_tokens=531 version_start=10 version_finish={finish} state={state}"]
    if retirement is not None:
        lines.append(f"schema=cache-lookup-v1 event=retire entry_id=7 tokens=531 "
                     f"retirement_version={retirement} reason=erased")
    return Evidence(lines)


class Classification(unittest.TestCase):
    def test_control(self):
        self.assertEqual(classify(pair(cached=531), evidence()), "exact")

    def test_retirement(self):
        self.assertEqual(classify(pair(admitted=14), evidence(retirement=12)),
                         "advisory_retirement")

    def test_later_version(self):
        self.assertEqual(classify(pair(admitted=14), evidence()), "advisory_later_version")

    def test_no_future_retirement_excuse(self):
        self.assertEqual(classify(pair(), evidence(retirement=12)), "unresolved_overreport")

    def test_no_self_admission_excuse(self):
        self.assertEqual(classify(pair(), evidence()), "unresolved_overreport")

    def test_missing_failure_counters_not_excused(self):
        value = pair(admitted=14)
        del value["actual"]["component_cache"]["restore_failures"]
        self.assertEqual(classify(value, evidence(retirement=12)), "unresolved_overreport")

    def test_restore_failure_beats_later_version_even_when_exact(self):
        self.assertEqual(classify(pair(admitted=14, failures=1), evidence(retirement=12)),
                         "bug_restore_failure")
        self.assertEqual(classify(pair(cached=531, failures=1), evidence()), "bug_restore_failure")

    def test_stale_publication_beats_later_admission(self):
        self.assertEqual(classify(pair(admitted=14), evidence(retirement=10)),
                         "bug_stale_publication")

    def test_r2_underreport_odd_version(self):
        self.assertEqual(classify(pair(matched=0, cached=685, version=7011)),
                         "conservative_mutation_underreport")

    def test_age_zero_not_evidence(self):
        value = pair(matched=0, cached=531)
        value["lookup"]["snapshot_age_ms"] = 0
        self.assertEqual(classify(value), "unexplained_underreport")

    def test_completed_overlap_even_version(self):
        logs = Evidence(["schema=cache-lookup-v1 event=lookup lookup_id=1 entry_id=0 "
                         "index_version=14 matched_tokens=0 version_start=10 version_finish=14 "
                         "state=mutation_overlap"])
        self.assertEqual(classify(pair(matched=0, cached=531, version=14), logs),
                         "conservative_mutation_underreport")

    def test_wrong_lookup_join_not_excused(self):
        self.assertEqual(classify(pair(version=12, admitted=14), evidence()), "invalid_evidence")

    def test_restore_ms_zero_does_not_prove_absence(self):
        value = pair()
        value["actual"] = {"cached_tokens": 0, "cache_restore_ms": 0}
        self.assertEqual(classify(value), "unresolved_overreport")

    def test_unpaired_restore_failure_still_fails_trial(self):
        logs = evidence()
        logs.restore_failures.append({"acquisition_id": 99})
        self.assertTrue(summarize([pair(cached=531)], logs)["hard_nonpass"])

    def test_missing_diagnostics_counted(self):
        self.assertEqual(summarize([pair(cached=531)])["missing_diagnostic_pairs"], 1)

    def test_rates(self):
        summary = summarize([pair(cached=531), pair(),
                             pair(matched=0, cached=531, version=11)])
        self.assertEqual(summary["pairs"], 3)
        self.assertEqual(summary["overreport_rate"], 1 / 3)
        self.assertEqual(summary["underreport_rate"], 1 / 3)
        self.assertTrue(summary["hard_nonpass"])


class Sanitization(unittest.TestCase):
    def test_allowlisted_request_metrics(self):
        value = {"usage": {"prompt_tokens": 620, "completion_tokens": 1,
                           "prompt_tokens_details": {"cached_tokens": 0},
                           "gufo": {"component_cache": {
                               "acquisition_id": 2, "admission_index_version": 10,
                               "selected_entry_id": 7, "restore_attempts": 1,
                               "restore_failures": 1, "restore_failure_reason": "ram_transfer",
                               "private_prompt": "PRIVATE"}}},
                 "timings": {"prompt_n": 620, "cache_n": 0},
                 "_receipt_request_id": "r123"}
        actual = generation_metrics(value)
        self.assertEqual(actual["request_id"], "r123")
        self.assertEqual(actual["component_cache"]["restore_failures"], 1)
        self.assertNotIn("PRIVATE", str(actual))
        invalid = deepcopy(value)
        invalid["usage"]["gufo"]["component_cache"]["restore_failure_reason"] = "PRIVATE"
        with self.assertRaises(ProbeError):
            generation_metrics(invalid)
        invalid = deepcopy(value)
        invalid["usage"]["gufo"]["component_cache"]["restore_failures"] = 2
        with self.assertRaises(ProbeError):
            generation_metrics(invalid)

    def test_fixture_parser_uses_counters_not_label(self):
        lines = ["schema=cache-lookup-fixture-v1 event=begin case=control",
                 "schema=cache-lookup-v1 event=lookup lookup_id=1 entry_id=7 "
                 "index_version=10 matched_tokens=531",
                 "schema=component-cache-v1 event=acquire acquisition_id=2 selected_tokens=0 "
                 "restore_attempts=1 restore_failures=1 admission_index_version=10",
                 "schema=cache-lookup-fixture-v1 case=control lookup_id=1 acquisition_id=2 "
                 "total_tokens=620 actual_tokens=0"]
        self.assertEqual(fixture_results(lines)[0]["cases"][0]["classification"],
                         "bug_restore_failure")


if __name__ == "__main__":
    unittest.main()
