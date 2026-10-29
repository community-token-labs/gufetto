"""Synthetic CPU-only fixtures for pd_attribution; no model or host access."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pd_attribution as pd


MS = 1_000_000


def member(request=1, slot=0, generation=1, agent=7):
    return {"request_id": request, "slot": slot,
            "slot_generation": generation, "logical_agent": agent}


def event(kind, time, ordinal=0, burst_id=None, **identity):
    return {**member(**identity), "event": kind, "time_ns": time,
            "ordinal": ordinal,
            "burst_id": (ordinal + 100 if kind == "token" else 0)
            if burst_id is None else burst_id}


def work(work_id, start, end, events=(), kind="decode_single", ready=(),
         prefiller=9, run=1, pid=1):
    return {
        "schema": 1, "run_id": run, "pid": pid, "thread_id": 1,
        "kind": kind, "start_ns": start, "end_ns": end,
        "sessions": [{"request_id": prefiller, "slot": 5,
                      "attribution_request_id": prefiller}],
        "tokens_requested": 1024, "tokens_processed": 512,
        "phases": [{"name": "target", "start_ns": start, "end_ns": end}],
        "attribution": {"version": 1, "work_id": work_id, "dropped": 0,
                        "ready": list(ready), "events": list(events), "execution": []},
    }


def trace():
    return [
        work(1, 0, 0, [event("submit", 0), event("admit", 0), event("token", 0, 1)]),
        work(2, 100 * MS, 900 * MS, kind="prefill", ready=[member()]),
        work(3, 400 * MS, 1200 * MS, kind="prefill", prefiller=10, ready=[member()]),
        work(4, 1500 * MS, 1600 * MS,
             [event("token", 1500 * MS, 2), event("terminal", 1600 * MS, 2)]),
    ]


def request(result, request_id=1, run=1, pid=1):
    return next(r for r in result["requests"] if r["request_id"] == request_id
                and r["run_id"] == run and r["pid"] == pid)


class AttributionTests(unittest.TestCase):
    def test_overlapping_peer_union_and_gap_ending_after_prefill(self):
        result = pd.analyze(trace())
        r = request(result)
        self.assertEqual(r["peer_prefill_union_ns"], [[100 * MS, 1200 * MS]])
        self.assertEqual(r["gaps"][0]["duration_ns"], 1500 * MS)
        exposed = r["token_gaps"]["exposed"]
        self.assertEqual(exposed["completed_samples"], 1)
        self.assertEqual(exposed["total_weight_ns"], 1100 * MS)
        self.assertEqual(exposed["stall_area_ns"], 700 * MS)
        self.assertEqual(exposed["p95_ns"], 1500 * MS)
        self.assertEqual(r["token_gaps"]["all_time"]["stall_area_ns"], 1000 * MS)
        self.assertEqual(r["terminal_censored_silence"]["all_time_lower_bound_ns"], 100 * MS)
        self.assertEqual(r["terminal_censored_silence"]["exposed_lower_bound_ns"], 0)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["issues"], [])
        self.assertIsNone(result["U"])
        self.assertIsNone(result["L"])
        self.assertIsNone(result["H"])
        self.assertEqual(result["prefills"][0]["tokens_processed"], 512)
        self.assertEqual(result["prefills"][0]["phase_names"], ["target"])

    def test_nearest_rank_and_weighted_p95(self):
        self.assertEqual(pd.quantiles(list(range(1, 21))),
                         {"p50_ns": 10, "p95_ns": 19, "p99_ns": 20, "max_ns": 20})
        # A long gap with small overlap must not dominate the exposure weight.
        self.assertEqual(pd.weighted_p95([(100, 95), (1000, 5)]), 100)
        self.assertEqual(pd.weighted_p95([(100, 94), (1000, 6)]), 1000)
        self.assertIsNone(pd.weighted_p95([(100, 0)]))
        records = [
            work(1, 0, 0, [event("submit", 0), event("token", 0, 1)]),
            work(2, 0, 95 * MS, kind="prefill", ready=[member()]),
            work(3, 100 * MS, 100 * MS, [event("token", 100 * MS, 2)]),
            work(4, 100 * MS, 105 * MS, kind="prefill", ready=[member()]),
            work(5, 1100 * MS, 1100 * MS,
                 [event("token", 1100 * MS, 3), event("terminal", 1100 * MS, 3)]),
        ]
        r = request(pd.analyze(records))
        self.assertEqual(r["token_gaps"]["exposed"]["time_weighted_p95_ns"], 100 * MS)
        self.assertEqual(r["token_gaps"]["all_time"]["time_weighted_p95_ns"], 1000 * MS)

    def test_terminal_without_token_is_censored_not_a_sample(self):
        records = [work(1, 0, 0, [event("submit", 0), event("admit", 0)]),
                   work(2, 100 * MS, 1000 * MS, kind="prefill", ready=[member()]),
                   work(3, 1500 * MS, 1500 * MS, [event("terminal", 1500 * MS)])]
        r = request(pd.analyze(records))
        self.assertEqual(r["gaps"], [])
        self.assertIsNone(r["token_gaps"]["exposed"]["p95_ns"])
        self.assertIsNone(r["token_gaps"]["exposed"]["stall_area_ns"])
        silence = r["terminal_censored_silence"]
        self.assertEqual(silence["origin"], "admit")
        self.assertEqual(silence["all_time_lower_bound_ns"], 1500 * MS)
        self.assertEqual(silence["exposed_lower_bound_ns"], 900 * MS)
        self.assertEqual(silence["exposed_stall_lower_bound_ns"], 500 * MS)

    def test_same_slot_generation_different_requests_do_not_join(self):
        records = trace()
        records[0]["attribution"]["events"].append(event("submit", 0, request=2))
        records[3]["attribution"]["events"].extend([
            event("token", 1500 * MS, 1, request=2),
            event("terminal", 1600 * MS, 1, request=2)])
        result = pd.analyze(records)
        self.assertEqual(request(result, 1)["token_count"], 2)
        self.assertEqual(request(result, 2)["token_count"], 1)
        self.assertEqual(request(result, 2)["gaps"], [])

    def test_own_prefill_and_other_clock_domains_are_not_peers(self):
        records = trace()
        records[1]["sessions"][0]["attribution_request_id"] = 1
        records[2]["pid"] = 2
        r = request(pd.analyze(records))
        self.assertEqual(r["token_gaps"]["all_time"]["completed_samples"], 1)
        self.assertEqual(r["token_gaps"]["exposed"]["completed_samples"], 0)
        self.assertIsNone(r["token_gaps"]["exposed"]["max_ns"])

    def test_explicit_native_prefiller_identity_join(self):
        records = trace()
        records[1]["sessions"] = [{"request_id": 42, "slot": 0}]
        records[1]["attribution"]["execution"] = [
            {**member(), "native_request_id": 42, "native_session": 123}]
        result = pd.analyze(records)
        self.assertEqual(result["prefills"][0]["request_ids"], [1])
        self.assertEqual(request(result)["gaps"][0]["exposure_ns"], 800 * MS)
        records[1]["attribution"]["execution"][0]["slot"] = 2
        result = pd.analyze(records)
        self.assertTrue(any("prefiller identity join" in issue for issue in result["issues"]))

    def test_missing_execution_never_implies_trace_id_namespace(self):
        records = trace()
        # Scheduler ID collides with decoder trace ID but identifies a peer.
        records[1]["sessions"] = [{"request_id": 1, "slot": 5}]
        result = pd.analyze(records)
        self.assertTrue(any("prefiller identity join" in i for i in result["issues"]))
        self.assertEqual(request(result)["responsiveness"]["status"], "INCONCLUSIVE")
        records[1]["sessions"][0]["attribution_request_id"] = 9
        result = pd.analyze(records)
        self.assertEqual(result["issues"], [])
        self.assertEqual(request(result)["gaps"][0]["exposure_ns"], 1100 * MS)

    def test_missing_or_late_admission_blocks_sufficiency(self):
        records = [work(8001, 0, 21 * MS, [
            event("submit", 0, generation=0), event("admit", 0),
            *(event("token", i * MS, i + 1) for i in range(21)),
            event("terminal", 21 * MS, 21)]),
            work(5, 0, 20 * MS, kind="prefill", ready=[member()])]
        self.assertEqual(request(pd.analyze(records))["responsiveness"]["status"], "SUFFICIENT_SAMPLES")
        records[0]["attribution"]["events"][1]["time_ns"] = MS
        r = request(pd.analyze(records))
        self.assertIn("invalid admission lifecycle ordering", r["responsiveness"]["reasons"])
        del records[0]["attribution"]["events"][1]
        r = request(pd.analyze(records))
        self.assertIn("missing/duplicate admit event", r["responsiveness"]["reasons"])
        self.assertEqual(r["responsiveness"]["status"], "INCONCLUSIVE")

    def test_run_and_pid_scope_requests(self):
        records = trace()
        other = copy.deepcopy(records)
        for record in other:
            record["run_id"] = 2
        result = pd.analyze(records + other)
        self.assertEqual(request(result, run=1)["token_count"], 2)
        self.assertEqual(request(result, run=2)["token_count"], 2)
        self.assertEqual(result["issues"], [])

    def test_bursts_are_separate_from_inter_work_service(self):
        records = trace()
        records[3]["attribution"]["events"] = [
            event("token", 1500 * MS, 2, burst_id=801),
            event("token", 1500 * MS, 3, burst_id=801),
            event("token", 1500 * MS + 1, 4, burst_id=801),
            event("terminal", 1600 * MS, 4)]
        r = request(pd.analyze(records))
        self.assertEqual([g["duration_ns"] for g in r["gaps"]], [1500 * MS, 0, 1])
        self.assertEqual(r["inter_work_gaps"]["all_time"]["completed_samples"], 1)
        self.assertEqual(r["intra_work_gaps"]["all_time"]["completed_samples"], 2)
        self.assertEqual(r["intra_work_gaps"]["all_time"]["max_ns"], 1)
        self.assertEqual(r["intra_work_gaps"]["exposed"]["completed_samples"], 0)

    def test_sample_floor_per_request_not_aggregate(self):
        records = [work(1, 0, 0, [event("submit", 0), event("admit", 0), event("token", 0, 1)])]
        records.append(work(2, 0, 20 * MS, kind="prefill", ready=[member()]))
        for i in range(1, 21):
            records.append(work(i + 2, i * MS, i * MS, [event("token", i * MS, i + 1)]))
        records.append(work(23, 21 * MS, 21 * MS, [event("terminal", 21 * MS, 21)]))
        r = request(pd.analyze(records))
        self.assertEqual(r["responsiveness"]["status"], "SUFFICIENT_SAMPLES")
        records[-2]["attribution"]["events"] = []
        r = request(pd.analyze(records))
        self.assertEqual(r["token_gaps"]["exposed"]["completed_samples"], 19)
        self.assertEqual(r["responsiveness"]["status"], "INCONCLUSIVE")

    def test_burst_samples_do_not_qualify_service(self):
        records = trace()
        records[1]["attribution"]["events"] = [
            event("token", 100 * MS + i, i + 2, burst_id=707) for i in range(25)]
        records[3]["attribution"]["events"] = [
            event("token", 1500 * MS, 27), event("terminal", 1600 * MS, 27)]
        r = request(pd.analyze(records))
        self.assertGreaterEqual(r["token_gaps"]["exposed"]["completed_samples"], 20)
        self.assertIn("fewer than 20 completed exposed inter-burst service gaps",
                      r["responsiveness"]["reasons"])

    def test_two_explicit_bursts_in_one_work(self):
        records = [work(91, 0, 1600 * MS, [
            event("submit", 0, generation=0), event("admit", 0),
            event("token", 0, 1, burst_id=27),
            event("token", 1, 2, burst_id=27),
            event("token", 1500 * MS, 3, burst_id=9001),
            event("token", 1500 * MS + 1, 4, burst_id=9001),
            event("terminal", 1600 * MS, 4)]),
            work(150, 100 * MS, 1000 * MS, kind="prefill", ready=[member()])]
        result = pd.analyze(records)
        r = request(result)
        self.assertEqual(result["issues"], [])
        self.assertEqual([g["inter_burst"] for g in r["gaps"]], [False, True, False])
        self.assertEqual(r["inter_burst_gaps"]["exposed"]["completed_samples"], 1)
        self.assertEqual(r["inter_burst_gaps"]["exposed"]["p95_ns"], 1500 * MS - 1)
        self.assertEqual(r["inter_work_gaps"]["all_time"]["completed_samples"], 0)
        self.assertEqual(r["intra_burst_gaps"]["all_time"]["completed_samples"], 2)

    def test_service_sample_floor_uses_bursts_not_work_count(self):
        records = [work(8001, 0, 21 * MS, [
            event("submit", 0, generation=0), event("admit", 0),
            *(event("token", i * MS, i + 1, burst_id=97 * (i + 1))
              for i in range(21)), event("terminal", 21 * MS, 21)]),
            work(5, 0, 20 * MS, kind="prefill", ready=[member()])]
        r = request(pd.analyze(records))
        self.assertEqual(r["responsiveness"]["status"], "SUFFICIENT_SAMPLES")
        self.assertEqual(r["inter_burst_gaps"]["exposed"]["completed_samples"], 20)
        self.assertEqual(r["inter_work_gaps"]["exposed"]["completed_samples"], 0)

    def test_missing_or_invalid_burst_never_falls_back_to_work(self):
        for value in (None, 0, -1, True, "17"):
            records = trace()
            token = records[-1]["attribution"]["events"][0]
            if value is None:
                del token["burst_id"]
            else:
                token["burst_id"] = value
            with self.subTest(value=value):
                result = pd.analyze(records)
                r = request(result)
                self.assertEqual(result["status"], "INCONCLUSIVE")
                self.assertTrue(any("burst_id" in issue for issue in result["issues"]))
                self.assertIsNone(r["gaps"][0]["inter_burst"])
                self.assertEqual(r["inter_burst_gaps"]["all_time"]["completed_samples"], 0)
                self.assertIsNone(r["inter_burst_gaps"]["exposed"]["p95_ns"])
                self.assertEqual(r["inter_work_gaps"]["all_time"]["completed_samples"], 1)

    def test_lifecycle_bursts_are_zero_and_closed_bursts_cannot_resume(self):
        records = trace()
        records[0]["attribution"]["events"][0]["burst_id"] = 88
        self.assertTrue(pd.analyze(records)["issues"])
        records = trace()
        records[-1]["attribution"]["events"] = [
            event("token", 1500 * MS, 2, burst_id=8),
            event("token", 1501 * MS, 3, burst_id=101),
            event("terminal", 1600 * MS, 3)]
        self.assertIn("noncontiguous publications reuse burst_id",
                      request(pd.analyze(records))["responsiveness"]["reasons"])

    def test_diagnostic_sidecar_is_not_perf_or_u(self):
        records = trace()
        records.insert(2, {"schema": "pd-profile-v1", "diagnostic": True,
                           "kind": "packed_requests", "run_id": 1, "pid": 1,
                           "payload_bytes": 999999, "U": 100})
        result = pd.analyze(records)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["diagnostic_sidecar_records"], 1)
        self.assertIsNone(result["U"])
        self.assertIsNone(result["L"])
        self.assertIsNone(result["H"])
        self.assertTrue(pd.analyze([records[2]])["issues"])

    def candidate_records(self):
        records = trace()
        records[1]["attribution"]["work_id"] = 1001
        records[2]["attribution"]["work_id"] = 82
        records[2]["thread_id"] = 2
        records[3]["attribution"]["work_id"] = 9997
        records[1]["attribution"]["ready"] = [member(), member(request=2), member(request=3)]
        records[3]["attribution"]["execution"] = [member(), member(request=2)]
        records[3]["forwards"] = [
            {"forward": 71, "prefill": False, "sessions": [
                {"request_id": 1, "native_session": 123, "context_depth": 10, "rows": 1}]},
            {"forward": 79, "prefill": False, "sessions": [
                {"request_id": 2, "native_session": 456, "context_depth": 10, "rows": 1}]},
            {"forward": 90, "prefill": False, "sessions": [
                {"request_id": 1, "native_session": 123, "context_depth": 11, "rows": 3}]}]
        return records

    def test_ready_join_uses_first_actual_target_not_whole_frozen_roster(self):
        result = pd.analyze(self.candidate_records())
        join = result["prefills"][0]["candidate_join"]
        self.assertEqual(join["next_decode_work_id"], 9997)
        self.assertEqual(join["execution_request_ids"], [1, 2])
        self.assertEqual(join["matched_execution_request_ids"], [1, 2])
        self.assertEqual(join["unmatched_ready_request_ids"], [3])
        self.assertEqual(join["candidate_request_ids"], [1])
        self.assertEqual(join["outside_first_target_request_ids"], [2])
        self.assertEqual(join["first_target_forward"], 71)
        self.assertEqual(len(join["actual_target_forwards"]), 3)
        self.assertEqual(len(join["declines"]), 2)
        self.assertIsNone(result["U"])

    def test_unknown_first_target_does_not_borrow_from_later_round(self):
        for value in (None, 0, 999):
            records = self.candidate_records()
            records[-1]["forwards"][0]["sessions"][0]["request_id"] = value
            result = pd.analyze(records)
            join = result["prefills"][0]["candidate_join"]
            self.assertIsNone(join["candidate_request_ids"])
            self.assertEqual(join["first_target_forward"], 71)
            self.assertTrue(join["declines"])

    def test_ready_join_never_skips_next_decode_to_find_a_match(self):
        records = self.candidate_records()
        records.append(work(99999, 1700 * MS, 1800 * MS))
        records[-1]["attribution"]["execution"] = [member(request=3)]
        records[-1]["forwards"] = [{"forward": 100, "prefill": False, "sessions": [
            {"request_id": 3, "rows": 1, "context_depth": 10}]}]
        records[-2]["attribution"]["execution"][0]["slot_generation"] = 2
        join = pd.analyze(records)["prefills"][0]["candidate_join"]
        self.assertEqual(join["candidate_request_ids"], [])
        self.assertEqual(join["matched_execution_request_ids"], [2])
        self.assertEqual(join["unmatched_ready_request_ids"], [1, 3])
        self.assertEqual(join["next_decode_work_id"], 9997)

    def test_candidate_join_uses_host_order_not_input_or_global_ids(self):
        records = self.candidate_records()
        records.append(work(3, 1600 * MS, 1700 * MS))
        # Touching half-open work intervals are unambiguous; input order and
        # globally allocated IDs do not determine which decode executed next.
        result = pd.analyze(list(reversed(records)))
        prefill = next(p for p in result["prefills"] if p["work_id"] == 1001)
        self.assertEqual(prefill["candidate_join"]["next_decode_work_id"], 9997)
        self.assertEqual(prefill["candidate_join"]["candidate_request_ids"], [1])
        records[-1]["start_ns"] = 1500 * MS
        result = pd.analyze(records)
        self.assertIsNone(result["prefills"][0]["candidate_join"]["candidate_request_ids"])

    def test_missing_forward_evidence_is_null_not_roster_guess(self):
        records = self.candidate_records()
        del records[-1]["forwards"]
        join = pd.analyze(records)["prefills"][0]["candidate_join"]
        self.assertEqual(join["matched_execution_request_ids"], [1, 2])
        self.assertIsNone(join["candidate_request_ids"])
        self.assertIsNone(join["actual_target_forwards"])
        self.assertIn("missing/invalid actual target forwards", join["declines"])

    def test_target_cannot_be_claimed_by_two_prefills(self):
        records = self.candidate_records()
        records[2]["thread_id"] = 1
        result = pd.analyze(records)
        self.assertIsNone(result["prefills"][0]["candidate_join"]["candidate_request_ids"])
        self.assertIn("another prefill intervened before next decoder target",
                      result["prefills"][0]["candidate_join"]["declines"])

    def test_candidate_join_declines_incomplete_or_other_thread(self):
        records = self.candidate_records()
        records[-1]["thread_id"] = 9
        join = pd.analyze(records)["prefills"][0]["candidate_join"]
        self.assertIsNone(join["next_decode_work_id"])
        self.assertIsNone(join["candidate_request_ids"])
        records = self.candidate_records()
        records[-1]["attribution"]["dropped"] = 1
        join = pd.analyze(records)["prefills"][0]["candidate_join"]
        self.assertIsNone(join["candidate_request_ids"])
        self.assertIn("trace integrity incomplete", join["declines"])

    def test_early_terminal_is_visible(self):
        records = trace()
        records[-1] = work(4, 600 * MS, 600 * MS,
                           [event("token", 600 * MS, 2), event("terminal", 600 * MS, 2)])
        self.assertIn("ready decoder terminated before peer prefill ended",
                      request(pd.analyze(records))["responsiveness"]["reasons"])

    def test_missing_drop_and_duplicate_work_invalidate(self):
        for mutation in ("missing", "drop", "duplicate"):
            with self.subTest(mutation=mutation):
                records = trace()
                if mutation == "missing":
                    del records[1]["attribution"]
                elif mutation == "drop":
                    records[1]["attribution"]["dropped"] = 1
                else:
                    records.append(copy.deepcopy(records[1]))
                result = pd.analyze(records)
                self.assertEqual(result["status"], "INCONCLUSIVE")
                self.assertTrue(result["issues"])
                self.assertIn("trace integrity incomplete; see issues",
                              request(result)["responsiveness"]["reasons"])

    def test_ordinal_loss_or_duplicate_is_not_joined(self):
        for ordinal in (1, 3):
            with self.subTest(ordinal=ordinal):
                records = trace()
                records[-1]["attribution"]["events"][0]["ordinal"] = ordinal
                r = request(pd.analyze(records))
                self.assertEqual(r["gaps"], [])
                self.assertIn("duplicate/gapped token ordinals", r["responsiveness"]["reasons"])

    def test_missing_trailing_token_detected_by_terminal_count(self):
        records = trace()
        records[-1]["attribution"]["events"][1]["ordinal"] = 3
        r = request(pd.analyze(records))
        self.assertIn("terminal token count disagrees with publications",
                      r["responsiveness"]["reasons"])

    def test_duplicate_roster_and_conflicting_identity(self):
        records = trace()
        records[1]["attribution"]["ready"].append(member())
        records[-1]["attribution"]["events"][0]["logical_agent"] = 8
        result = pd.analyze(records)
        self.assertTrue(any("duplicate ready" in issue for issue in result["issues"]))
        self.assertIn("conflicting logical-agent identity",
                      request(result)["responsiveness"]["reasons"])

    def test_ready_roster_not_inferred_from_queue_counts(self):
        records = trace()
        for record in records:
            record["queue"] = {"decoders_due": 999}
        records[1]["attribution"]["ready"] = [member(request=3)]
        result = pd.analyze(records)
        self.assertEqual(len(result["requests"]), 2)
        self.assertEqual(request(result, 3)["token_count"], 0)
        self.assertIsNone(request(result, 3)["token_gaps"]["exposed"]["p95_ns"])

    def test_missing_terminal_and_missing_identity(self):
        records = trace()
        records[-1]["attribution"]["events"].pop()
        r = request(pd.analyze(records))
        self.assertIsNone(r["terminal_censored_silence"])
        self.assertTrue(any("missing/duplicate terminal" in reason
                            for reason in r["responsiveness"]["reasons"]))
        del records[-1]["attribution"]["events"][0]["slot_generation"]
        self.assertTrue(pd.analyze(records)["issues"])

    def test_integer_precision_beyond_json_double(self):
        records = trace()
        offset = 2 ** 60 + 3
        for record in records:
            record["start_ns"] += offset
            record["end_ns"] += offset
            for e in record["attribution"]["events"]:
                e["time_ns"] += offset
        r = request(pd.analyze(json.loads(json.dumps(records))))
        self.assertEqual(r["gaps"][0]["start_ns"], offset)
        self.assertEqual(r["gaps"][0]["exposure_ns"], 1100 * MS)
        self.assertEqual(r["gaps"][0]["duration_ns"], 1500 * MS)

    def test_half_open_intersections_and_disjoint_union(self):
        self.assertEqual(pd.union([(0, 3), (2, 4), (5, 6), (6, 8), (9, 9)]),
                         [[0, 4], [5, 8]])
        self.assertEqual(pd.intersection(4, 5, [[0, 4], [5, 8]]), 0)
        self.assertEqual(pd.intersection(3, 6, [[0, 4], [5, 8]]), 2)
        self.assertEqual(pd.intersection(10, 5, [[0, 100]]), 0)

    def test_malformed_fields_are_inconclusive(self):
        mutations = [lambda r: r.update(start_ns=1.5),
                     lambda r: r.update(schema=2),
                     lambda r: r["attribution"].update(events=None),
                     lambda r: r["attribution"].update(execution=None),
                     lambda r: r.update(sessions=None),
                     lambda r: r.update(phases=None)]
        for mutate in mutations:
            records = trace()
            mutate(records[1])
            with self.subTest(record=records[1]):
                self.assertTrue(pd.analyze(records)["issues"])
        self.assertEqual(pd.analyze([])["status"], "INCONCLUSIVE")

    def test_cli_truncated_json_and_no_prompt_or_token_ids(self):
        records = trace()
        records[0]["prompt"] = "SECRET PROMPT"
        records[0]["token_ids"] = [12345]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records) + '\n{"schema":',
                            encoding="utf-8")
            completed = subprocess.run([sys.executable, str(Path(pd.__file__)), str(path)],
                                       check=True, text=True, capture_output=True)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertTrue(any("truncated JSON" in issue for issue in result["issues"]))
        self.assertNotIn("SECRET PROMPT", completed.stdout)
        self.assertNotIn("token_ids", completed.stdout)
        self.assertEqual(request(result)["gaps"][0]["duration_ns"], 1500 * MS)


if __name__ == "__main__":
    unittest.main()
