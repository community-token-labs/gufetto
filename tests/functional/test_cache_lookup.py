"""CPU-only probe tests; all sockets are owned loopback fake servers, no model."""

import base64
from collections import OrderedDict
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import struct
import tempfile
import threading
import time
import unittest
import zlib

from cache_lookup import (Client, Config, Harness, ProbeError, distribution,
                          bracketed_tasks, generation_metrics, lookup_metrics, main,
                          png_url, processing_tasks)


class Engine:
    """Independent exact-key cache oracle; not a model/cache implementation test."""

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.epoch = 0
        self.calls = []
        self.lock = threading.Lock()
        self.generation_delay = 0
        self.lookup_delay = 0
        self.lookup_delta = 0
        self.cache_delta = 0
        self.total_delta = 0
        self.fail_lookup = False
        self.busy_lookup = False
        self.fail_generation = False
        self.invalid_lookup = None
        self.slots_value = [{"id": i, "id_task": 987654321 + i, "is_processing": True,
                             "prompt": "PRIVATE_SLOT_PROMPT", "model": "PRIVATE_MODEL"}
                            for i in range(8)]
        self.slots_error = False
        self.slots_delay = 0

    def get(self, endpoint):
        assert endpoint == "/slots"
        time.sleep(self.slots_delay)
        if self.slots_error:
            raise ProbeError("http_status", http_status=503, wall_ms=.1)
        return deepcopy(self.slots_value), .5

    def post(self, endpoint, body):
        body = deepcopy(body)
        if endpoint == "/v1/chat/completions":
            time.sleep(self.generation_delay)
        else:
            time.sleep(self.lookup_delay)
        with self.lock:
            self.calls.append((endpoint, body))
            rendered = json.dumps({k: body[k] for k in (
                "messages", "tools", "chat_template_kwargs", "reasoning_effort") if k in body}, sort_keys=True)
            # Image encoding/metadata length is not a visual token count.
            sized = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "<64x64 image>", rendered)
            total = 1 + len(sized) // 4
            cached = total if rendered in self.cache and body.get("cache_prompt") is not False else 0
            if endpoint == "/v1/cache/lookup":
                if self.fail_lookup:
                    raise ProbeError("http_status", http_status=503, wall_ms=.1)
                if self.busy_lookup:
                    raise ProbeError("cache_lookup_busy", http_status=503, wall_ms=.2)
                matched = max(0, cached - self.lookup_delta)
                result = {"total_tokens": total + self.total_delta,
                          "matched_tokens": matched, "tier": "ram" if matched else "none",
                          "epoch": "fixture-opaque-epoch-" + str(self.epoch), "lookup_ms": .125,
                          "index_version": self.epoch, "snapshot_age_ms": .25,
                          "restore_estimate_ms": None, "prefill_estimate_ms": 0,
                          "untrusted_extra": "PRIVATE_RESPONSE_AND_PROMPT"}
                if self.invalid_lookup:
                    result.update(self.invalid_lookup)
            elif endpoint == "/v1/chat/completions":
                if self.fail_generation:
                    raise ProbeError("http_status", http_status=500, wall_ms=.1)
                # Inject only when a hit exists, leaving ranges/accounting valid.
                cached = max(0, cached - self.cache_delta)
                self.cache[rendered] = True
                self.cache.move_to_end(rendered)
                while len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
                self.epoch += 1
                result = {"choices": [{"message": {"role": "assistant", "content": "PRIVATE_REPLY"},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": total, "completion_tokens": 2,
                                    "total_tokens": total + 2,
                                    "prompt_tokens_details": {"cached_tokens": cached}},
                          "timings": {"prompt_n": total - cached, "cache_n": cached}}
            else:
                raise AssertionError("unexpected endpoint")
        return result, .5


@contextmanager
def fake_http(engine, raw_response=None, redirect=False):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if redirect:
                self.send_response(302)
                self.send_header("Location", "/unexpected-redirect")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 200
            try:
                value, _ = engine.post(self.path, body)
                raw = json.dumps(value).encode() if raw_response is None else raw_response
            except ProbeError as error:
                status = error.metrics.get("http_status", 500)
                # A real server could echo sensitive content in its error object.
                raw = json.dumps({"error": {"message": "PRIVATE_RESPONSE_AND_PROMPT",
                                            "code": error.code}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            try:
                value, _ = engine.get(self.path)
                status, raw = 200, json.dumps(value).encode()
            except ProbeError as error:
                status = error.metrics.get("http_status", 500)
                raw = b'{"error":{"message":"PRIVATE_SLOT_ERROR"}}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 64

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class AccountingTests(unittest.TestCase):
    def test_lookup_schema_allows_missing_or_null_estimates_and_filters_extra_text(self):
        value, _ = Engine().post("/v1/cache/lookup", {"messages": []})
        parsed = lookup_metrics(value)
        self.assertNotIn("untrusted_extra", parsed)
        self.assertIsNone(parsed["restore_estimate_ms"])
        self.assertEqual(parsed["epoch"], value["epoch"])
        self.assertIsInstance(parsed["epoch"], str)
        del value["restore_estimate_ms"], value["prefill_estimate_ms"]
        self.assertNotIn("restore_estimate_ms", lookup_metrics(value))

    def test_lookup_rejects_invalid_counts_tiers_epochs_and_nonfinite_timings(self):
        baseline, _ = Engine().post("/v1/cache/lookup", {"messages": []})
        for change in ({"total_tokens": True}, {"matched_tokens": -1}, {"epoch": 7}, {"epoch": ""},
                       {"tier": "pin"}, {"lookup_ms": float("nan")},
                       {"restore_estimate_ms": -1}, {"prefill_estimate_ms": "PRIVATE"},
                       {"matched_tokens": 1000000}, {"tier": "live"}):
            with self.subTest(change=change), self.assertRaises(ProbeError):
                lookup_metrics({**baseline, **change})

    def test_snapshot_metadata_required_uint64_and_finite_age(self):
        baseline, _ = Engine().post("/v1/cache/lookup", {"messages": []})
        for version in (0, 2**64 - 1):
            for age in (0, 0.0, 12.5):
                parsed = lookup_metrics({**baseline, "index_version": version, "snapshot_age_ms": age})
                self.assertEqual((parsed["index_version"], parsed["snapshot_age_ms"]), (version, age))
        for key, bad in (("index_version", (-1, True, 1.0, 2**64, None, "PRIVATE")),
                         ("snapshot_age_ms", (-1, True, float("inf"), float("nan"), 10**400, None, "PRIVATE"))):
            for value in bad:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ProbeError, "lookup_invalid_" + key):
                    lookup_metrics({**baseline, key: value})
            missing = dict(baseline)
            del missing[key]
            with self.assertRaisesRegex(ProbeError, "lookup_invalid_" + key):
                lookup_metrics(missing)

    def test_slots_metadata_and_adjacent_brackets_reject_churn_errors_and_gaps(self):
        tasks = processing_tasks(Engine().slots_value)
        self.assertEqual(len(tasks), 8)
        polls = [(0, .001, tasks, None), (.05, .051, tasks, None)]
        self.assertEqual(bracketed_tasks(.01, .02, polls, .05), 8)
        churned = frozenset((slot, task + 1) for slot, task in tasks)
        self.assertEqual(bracketed_tasks(.01, .02, [polls[0], (.05, .051, churned, None)], .05), 0)
        self.assertEqual(bracketed_tasks(.01, .02, [polls[0], (.05, .051, None, "error")], .05), 0)
        self.assertEqual(bracketed_tasks(.01, .02, [polls[0], (.2, .201, tasks, None)], .05), 0)
        self.assertEqual(bracketed_tasks(.0005, .02, polls, .05), 0)
        self.assertEqual(bracketed_tasks(.01, .06, polls, .05), 0)
        for value in ({}, [{"id": 0, "is_processing": 1}],
                      [{"id": 0, "is_processing": True, "id_task": -1}],
                      [{"id": 0, "is_processing": False}] * 2):
            with self.assertRaisesRegex(ProbeError, "slots_invalid_metadata"):
                processing_tasks(value)

    def test_snapshot_stats_keep_age_distribution_and_do_not_assume_version_order(self):
        rows = [{"lookup": {"snapshot_age_ms": age, "index_version": version}}
                for version, age in ((9, 1), (7, 4), (9, 3))]
        stats = Harness.snapshot_stats(rows + [{"lookup_error_code": "cache_lookup_busy"}])
        self.assertEqual(stats["index_version"], {"n": 3, "min": 7, "max": 9, "distinct": 2})
        self.assertEqual(stats["snapshot_age_ms"], distribution([1, 4, 3]))
        self.assertFalse(stats["version_order_checked"])
        self.assertIsNone(Harness.snapshot_stats([])["index_version"]["min"])

    def test_generation_requires_both_cache_sources_and_total_prompt_accounting(self):
        value, _ = Engine().post("/v1/chat/completions", {"messages": []})
        self.assertEqual(generation_metrics(value)["total_tokens"], value["usage"]["prompt_tokens"])
        for target, key, replacement in (("timings", "cache_n", 1),
                                         ("timings", "prompt_n", 100000),
                                         ("usage", "completion_tokens", True)):
            altered = deepcopy(value)
            altered[target][key] = replacement
            with self.assertRaises(ProbeError):
                generation_metrics(altered)
        del value["usage"]["prompt_tokens_details"]
        with self.assertRaisesRegex(ProbeError, "generation_missing_accounting"):
            generation_metrics(value)

    def test_integral_float_generation_counts_are_normalized_but_fractions_rejected(self):
        value, _ = Engine().post("/v1/chat/completions", {"messages": []})
        value["timings"]["cache_n"] = 0.0
        value["timings"]["prompt_n"] = float(value["timings"]["prompt_n"])
        self.assertIsInstance(generation_metrics(value)["cache_n"], int)
        value["timings"]["prompt_n"] += .5
        with self.assertRaises(ProbeError):
            generation_metrics(value)

    def test_nearest_rank_quantiles_and_empty_samples(self):
        self.assertEqual(distribution(range(1, 101)),
                         {"n": 100, "p50": 50, "p95": 95, "p99": 99, "max": 100})
        self.assertEqual(distribution([7])["p99"], 7)
        self.assertEqual(distribution([])["n"], 0)
        self.assertIsNone(distribution([])["max"])


class WorkloadTests(unittest.TestCase):
    def runner(self, engine=None, **kwargs):
        config = Config("PRIVATE_MODEL", rounds=1, pressure_lineages=20,
                        system_lines=2, pressure_lines=3, latency_lookups=20,
                        lookup_interval_ms=1, load_timeout=3, **kwargs)
        return Harness(engine or Engine(), config)

    def test_hundreds_case_default_histories_match_and_keep_identical_pair_payloads(self):
        engine = Engine()
        runner = Harness(engine, Config("fixture", images=True))
        runner.sequential()
        report = runner.finish()
        self.assertEqual(report["summary"]["sequential_pairs"], 477)
        self.assertEqual(report["failures"], [])
        self.assertGreater(report["pressure"]["anchors_with_retention_drop"], 0)
        self.assertEqual(len(engine.calls), 2 * len(runner.rows))
        for lookup, generation in zip(engine.calls[::2], engine.calls[1::2]):
            self.assertEqual(lookup[0], "/v1/cache/lookup")
            self.assertEqual(generation[0], "/v1/chat/completions")
            self.assertEqual(lookup[1], generation[1])
        labels = {row["case"] for row in runner.rows}
        for suffix in ("seed", "append", "tool_append", "rewind", "regenerate", "newchat_shared",
                       "cache_prompt_false"):
            self.assertIn("history_000_" + suffix, labels)
        append = engine.calls[4][1]["messages"]
        self.assertEqual(append[2]["content"], "PRIVATE_REPLY")
        self.assertNotIn("PRIVATE_REPLY", json.dumps(report))

    def test_preserve_thinking_variants_use_same_reasoned_history_without_mutating_bodies(self):
        seen = []

        class RecordingHarness(Harness):
            def pair(self, label, body, strict=True):
                seen.append((label, body, deepcopy(body)))
                return super().pair(label, body, strict)

        engine = Engine()
        config = Config("fixture", rounds=2, pressure_lineages=0, system_lines=1)
        with fake_http(engine) as url:
            runner = RecordingHarness(Client(url, timeout=2), config)
            runner.sequential()
        bodies = {label: body for label, body, _ in seen}
        for index in range(config.rounds):
            label = f"history_{index:03d}"
            variants = [body for name, body, _ in seen if name.startswith(label + "_preserve_thinking_")]
            self.assertEqual(len(variants), 2)
            kept, dropped = variants
            self.assertIs(kept["chat_template_kwargs"]["preserve_thinking"], True)
            self.assertIs(dropped["chat_template_kwargs"]["preserve_thinking"], False)
            self.assertEqual(kept["messages"], dropped["messages"])
            self.assertIsNot(kept["messages"], dropped["messages"])
            self.assertIsNot(kept["messages"][-2], dropped["messages"][-2])
            historical = kept["messages"][-2]
            self.assertEqual(historical["role"], "assistant")
            self.assertEqual(historical["content"], "PRIVATE_REPLY")
            self.assertTrue(historical["reasoning_content"].strip())
            appended = bodies[label + "_append"]
            expected = deepcopy(appended)
            expected["messages"][-2]["reasoning_content"] = historical["reasoning_content"]
            for variant in variants:
                self.assertEqual(variant["reasoning_effort"], "none")
                self.assertEqual({k: v for k, v in variant.items() if k != "chat_template_kwargs"}, expected)
            self.assertNotIn("reasoning_content", appended["messages"][-2])
            for suffix in ("seed", "regenerate", "append", "tool_append", "rewind",
                           "rewind_regenerate", "newchat_shared", "cache_prompt_false"):
                self.assertNotIn("chat_template_kwargs", bodies[label + "_" + suffix])
        # References catch later mutation that the fake worker's deep copies alone
        # would conceal. Both HTTP requests must also receive each exact snapshot.
        self.assertEqual(len(engine.calls), 2 * len(seen))
        for (_, body, snapshot), lookup, generation in zip(seen, engine.calls[::2], engine.calls[1::2]):
            self.assertEqual(body, snapshot)
            self.assertEqual(lookup, ("/v1/cache/lookup", snapshot))
            self.assertEqual(generation, ("/v1/chat/completions", snapshot))
        report = runner.finish()
        self.assertEqual(report["failures"], [])
        self.assertNotIn(historical["reasoning_content"], json.dumps(report))
        # Editing either replay after dispatch cannot change its sibling or source.
        reasoning = historical["reasoning_content"]
        kept["messages"][-2]["reasoning_content"] = "changed only in this replay"
        self.assertEqual(dropped["messages"][-2]["reasoning_content"], reasoning)
        self.assertNotIn("reasoning_content", appended["messages"][-2])

    def test_strict_equality_and_concurrent_over_report_both_fail(self):
        engine = Engine()
        runner = self.runner(engine)
        body = runner.body("fixture")
        runner.pair("seed", body)
        engine.cache_delta = 1
        _, strict = runner.pair("strict", body)
        _, concurrent = runner.pair("concurrent", body, strict=False)
        self.assertIn("cached_tokens_mismatch", strict["failures"])
        self.assertEqual(concurrent["status"], "failed")
        self.assertIn("concurrent_cached_tokens_over_report", concurrent["failures"])
        self.assertEqual(concurrent["cache_comparison"], "over_report")
        concurrent.pop("_generation_interval")
        report = runner.finish()
        self.assertEqual(report["summary"]["concurrent_pair_comparisons"]["over_report_count"], 1)
        self.assertEqual(report["summary"]["sequential_cache_mismatches"], 1)

    def test_busy_is_recorded_without_retry_only_under_load(self):
        engine = Engine()
        engine.busy_lookup = True
        runner = self.runner(engine)
        body = runner.body("fixture")
        _, strict = runner.pair("strict_busy", body)
        _, concurrent = runner.pair("load_busy", body, strict=False)
        self.assertIn("cache_lookup_busy", strict["failures"])
        self.assertEqual(concurrent["status"], "busy")
        self.assertNotIn("failures", concurrent)
        self.assertEqual(concurrent["actual"]["completion_tokens"], 2)
        self.assertEqual(len(engine.calls), 4)
        self.assertEqual(runner.availability([concurrent])["busy_rate"], 1)
        latency = runner.latencies([concurrent])
        self.assertEqual(latency["http_503_busy_wall"]["n"], 1)
        self.assertEqual(latency["engine_lookup"]["n"], 0)
        self.assertIsNone(latency["engine_lookup"]["p50"])

    def test_busy_does_not_mask_generation_failures(self):
        engine = Engine()
        engine.busy_lookup, engine.fail_generation = True, True
        runner = self.runner(engine)
        _, row = runner.pair("busy_then_generation_failure", runner.body("fixture"), strict=False)
        self.assertTrue(row["lookup_busy"])
        self.assertEqual(row["status"], "failed")
        self.assertEqual(len(runner.failures), 1)
        self.assertEqual(runner.failures[0]["http_status"], 500)

    def test_concurrent_total_mismatch_still_fails(self):
        engine = Engine()
        engine.total_delta = 1
        runner = self.runner(engine)
        _, row = runner.pair("concurrent", runner.body("fixture"), strict=False)
        self.assertIn("total_tokens_mismatch", row["failures"])

    def test_all_lookup_and_generation_failures_survive_and_later_cases_continue(self):
        engine = Engine()
        engine.fail_lookup, engine.fail_generation = True, True
        runner = self.runner(engine)
        runner.sequential()
        report = runner.finish()
        self.assertGreater(len(report["failures"]), 50)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any(row["case"] == "anchor_2_after" for row in report["cases"]))
        self.assertEqual(len(engine.calls), len(report["cases"]) * 2)

    def test_no_eviction_and_missing_tier_are_unqualified_not_silent_pass(self):
        runner = self.runner(Engine(capacity=1000), required_tiers=("disk",))
        report = runner.run("exactness")
        self.assertEqual(report["status"], "unqualified")
        self.assertIn("eviction_not_observed_no_anchor_retention_drop", report["uncertainties"])
        self.assertIn("required_tier_not_observed_disk", report["uncertainties"])
        self.assertIsNone(report.get("load"))

    def test_image_fixture_changes_encoding_not_pixel_shape_and_covers_image_edits(self):
        def pixels(url):
            raw = base64.b64decode(url.split(",", 1)[1])
            offset, data = 8, b""
            while offset < len(raw):
                size = struct.unpack(">I", raw[offset:offset + 4])[0]
                if raw[offset + 4:offset + 8] == b"IDAT":
                    data += raw[offset + 8:offset + 8 + size]
                offset += size + 12
            return zlib.decompress(data)

        red, reencoded, blue = png_url((255, 0, 0)), png_url((255, 0, 0), True), png_url((0, 0, 255))
        self.assertNotEqual(red, reencoded)
        self.assertEqual(pixels(red), pixels(reencoded))
        self.assertNotEqual(pixels(red), pixels(blue))
        runner = self.runner(images=True)
        runner.image_cases()
        labels = {row["case"] for row in runner.rows}
        self.assertIn("image_same_pixels_new_encoding", labels)
        self.assertIn("image_changed_pixels", labels)
        self.assertIn("image_removed", labels)
        self.assertIn("image_appended_regenerate", labels)
        report = runner.finish()
        self.assertEqual(report["failures"], [])
        self.assertNotIn("data:image", json.dumps(report))

    def test_pin_depth_and_private_chat_payload_not_stored(self):
        runner = self.runner(pin_min_matched_tokens=1000000)
        runner.pin_body = runner.body("PRIVATE_PIN_PROMPT", tools=True)
        runner.sequential()
        report = runner.finish()
        self.assertEqual(sum(f["code"] == "pin_depth_not_reached" for f in report["failures"]), 4)
        self.assertNotIn("PRIVATE_PIN_PROMPT", json.dumps(report))
        self.assertNotIn("PRIVATE_MODEL", json.dumps(report))


class HttpTests(unittest.TestCase):
    def test_real_http_exact_pair_and_v1_url_normalization(self):
        engine = Engine()
        with fake_http(engine) as url:
            client = Client(url + "/v1/", timeout=2)
            runner = Harness(client, Config("fixture"))
            body = runner.body("PRIVATE_PROMPT")
            runner.pair("seed", body)
            _, row = runner.pair("warm", body)
        self.assertTrue(row["cache_equal"])
        self.assertGreater(row["lookup_wall_ms"], 0)
        self.assertGreater(row["lookup"]["matched_tokens"], 0)
        self.assertEqual(runner.failures, [])

    def test_http_error_body_and_extra_fields_never_leak(self):
        engine = Engine()
        engine.fail_lookup = True
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), Config("fixture"))
            runner.pair("error", runner.body("PRIVATE_PROMPT"))
        report = runner.finish()
        encoded = json.dumps(report)
        self.assertNotIn("PRIVATE", encoded)
        self.assertEqual(report["failures"][0]["http_status"], 503)
        self.assertEqual(report["cases"][0]["actual"]["completion_tokens"], 2)

    def test_unrecognized_503_is_still_failure_under_load(self):
        engine = Engine()
        engine.fail_lookup = True
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), Config("fixture"))
            _, row = runner.pair("unknown_503", runner.body("fixture"), strict=False)
        self.assertEqual(row["status"], "failed")
        self.assertNotIn("lookup_busy", row)
        self.assertEqual(len(runner.failures), 1)
        self.assertEqual(runner.availability([row])["other_failures"], 1)

    def test_invalid_json_is_redacted_and_redirect_is_not_followed(self):
        for options, code in (({"raw_response": b"PRIVATE not json"}, "invalid_response_json"),
                              ({"redirect": True}, "http_status")):
            engine = Engine()
            with fake_http(engine, **options) as url:
                with self.assertRaises(ProbeError) as caught:
                    Client(url, timeout=2).post("/v1/cache/lookup", {"messages": []})
            self.assertEqual(caught.exception.code, code)
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertLessEqual(len(engine.calls), 1)

    def test_http_concurrent_over_report_is_hard_failure_not_a_race_exception(self):
        engine = Engine()
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), Config("fixture"))
            body = runner.body("PRIVATE_PROMPT")
            runner.pair("seed", body)
            engine.cache_delta = 3
            _, row = runner.pair("over_report", body, strict=False)
        report = runner.finish()
        self.assertEqual(row["cache_delta_tokens"], 3)
        self.assertEqual(report["status"], "failed")
        self.assertIn("concurrent_cached_tokens_over_report", row["failures"])
        stats = report["summary"]["concurrent_pair_comparisons"]
        self.assertEqual((stats["over_report_count"], stats["over_report_rate"]), (1, 1))
        self.assertEqual(stats["under_report_count"], 0)
        self.assertEqual(engine.calls[-2][1], engine.calls[-1][1])
        self.assertNotIn("PRIVATE", json.dumps(report))

    def test_http_under_report_is_measured_with_compared_pair_denominator(self):
        engine = Engine()
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), Config("fixture"))
            body = runner.body("PRIVATE_PROMPT")
            runner.pair("seed", body)
            engine.lookup_delta = 2
            _, under = runner.pair("under_report", body, strict=False)
            # Idle equality remains strict in the under-report direction too.
            _, strict = runner.pair("idle_under_report", body)
            engine.lookup_delta = 0
            runner.pair("equal", body, strict=False)
            engine.busy_lookup = True
            runner.pair("uncompared", body, strict=False)
        report = runner.finish()
        stats = report["summary"]["concurrent_pair_comparisons"]
        self.assertEqual(under["cache_delta_tokens"], -2)
        self.assertEqual(under["status"], "passed")
        self.assertIn("cached_tokens_mismatch", strict["failures"])
        self.assertEqual(stats["pairs"], 3)
        self.assertEqual(stats["compared_pairs"], 2)
        self.assertEqual(stats["uncompared_pairs"], 1)
        self.assertEqual(stats["under_report_count"], 1)
        self.assertEqual(stats["under_report_rate"], .5)
        self.assertEqual(stats["over_report_count"], 0)
        self.assertNotIn("PRIVATE", json.dumps(report))

    def test_successful_only_latency_excludes_http_200_schema_and_json_errors(self):
        engine = Engine()
        engine.invalid_lookup = {"lookup_ms": "PRIVATE"}
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), Config("fixture"))
            _, schema_row = runner.pair("schema_error", runner.body("fixture"), strict=False)
        with fake_http(engine, raw_response=b"PRIVATE not json") as url:
            runner.client = Client(url, timeout=2)
            json_row = {"case": "json_error", "phase": "load_lookup"}
            runner.lookup(json_row, runner.body("fixture"), allow_busy=True)
        stats = runner.availability([schema_row, json_row])
        self.assertEqual(stats["available_200"], 0)
        self.assertEqual(stats["other_failures"], 2)
        self.assertEqual(stats["http_status_counts"], {"200": 2})
        self.assertEqual(stats["error_code_counts"], {"lookup_invalid_lookup_ms": 1,
                                                     "invalid_response_json": 1})
        latency = runner.latencies([schema_row, json_row])
        self.assertEqual(latency["http_wall"]["n"], 2)
        self.assertEqual(latency["http_200_wall"]["n"], 0)
        self.assertEqual(latency["engine_lookup"]["n"], 0)

    def test_fixed_cadence_dispatches_slow_http_concurrently(self):
        engine = Engine()
        engine.lookup_delay = .12
        config = Config("fixture", clients=0, latency_lookups=12, lookup_workers=8,
                        lookup_interval_ms=25, load_timeout=3)
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), config)
            report = runner.run("load")
        cadence = report["load"]["cadence"]
        self.assertEqual(cadence["scheduled_slots"], 12)
        self.assertEqual(cadence["attempted_slots"], 12)
        self.assertEqual(cadence["missed_slots"], 0)
        self.assertEqual(cadence["unlaunched_slots"], 0)
        self.assertEqual(cadence["target_hz"], 40)
        self.assertLess(cadence["observed_launch_intervals_ms"]["p50"], 70)
        self.assertGreater(report["load"]["latency_all_samples_ms"]["http_200_wall"]["p50"], 100)
        self.assertEqual([row["scheduled_offset_ms"] for row in runner.rows], list(range(0, 300, 25)))
        self.assertTrue(all(endpoint == "/v1/cache/lookup" for endpoint, _ in engine.calls))
        self.assertTrue(report["load"]["external_load_overlap_unverified"])
        self.assertEqual(report["load"]["samples_fully_covered_by_all_generations"], 0)

    def test_client_capacity_missed_slots_are_counted_not_queued(self):
        engine = Engine()
        engine.lookup_delay = .12
        config = Config("fixture", clients=0, latency_lookups=12, lookup_workers=1,
                        lookup_interval_ms=25, load_timeout=3)
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), config)
            report = runner.run("load")
        cadence = report["load"]["cadence"]
        self.assertGreater(cadence["missed_slots"], 0)
        self.assertEqual(cadence["attempted_slots"] + cadence["missed_slots"], 12)
        self.assertEqual(cadence["attempted_slots"], len(engine.calls))
        self.assertEqual(sum(cadence["missed_codes"].values()), cadence["missed_slots"])
        self.assertIn("client_capacity_missed", cadence["missed_codes"])
        self.assertIn("lookup_launch_cadence_missed", report["uncertainties"])
        self.assertEqual(sum(row["status"] == "missed" for row in runner.rows), cadence["missed_slots"])

    def test_launch_thread_delay_records_missed_cadence_without_burst_replay(self):
        class SlowBodyHarness(Harness):
            def body(self, *args, **kwargs):
                time.sleep(.06)
                return super().body(*args, **kwargs)

        config = Config("fixture", clients=0, latency_lookups=20,
                        lookup_interval_ms=10, load_timeout=2)
        with fake_http(Engine()) as url:
            report = SlowBodyHarness(Client(url, timeout=2), config).run("load")
        cadence = report["load"]["cadence"]
        self.assertGreater(cadence["missed_codes"]["client_cadence_missed"], 0)
        self.assertEqual(cadence["attempted_slots"] + cadence["missed_slots"], 20)
        self.assertIn("lookup_launch_cadence_missed", report["uncertainties"])

    def test_deadline_records_slots_never_launched(self):
        config = Config("fixture", clients=0, latency_lookups=20,
                        lookup_interval_ms=50, load_timeout=.12)
        with fake_http(Engine()) as url:
            report = Harness(Client(url, timeout=2), config).run("load")
        cadence = report["load"]["cadence"]
        self.assertGreater(cadence["unlaunched_slots"], 0)
        self.assertEqual(cadence["scheduled_slots"] + cadence["unlaunched_slots"], 20)
        self.assertIn("load_deadline_before_all_samples", report["uncertainties"])

    def test_eight_agent_load_records_actual_overlap_and_pair_outcomes(self):
        engine = Engine()
        engine.generation_delay = .12
        config = Config("fixture", clients=8, latency_lookups=20,
                        lookup_interval_ms=10, load_timeout=4, system_lines=1)
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), config)
            report = runner.run("load")
        load = report["load"]
        self.assertEqual(load["agents"], 8)
        self.assertEqual(len(load["completed_generations_with_tokens"]), 8)
        self.assertTrue(all(load["completed_generations_with_tokens"]))
        self.assertGreater(load["samples_fully_covered_by_all_generations"], 0)
        self.assertEqual(report["failures"], [])
        stats = load["concurrent_pair_comparisons"]
        self.assertGreaterEqual(stats["compared_pairs"], 8)
        self.assertEqual(stats["over_report_count"], 0)
        self.assertTrue(all(row["generation_agents_overlapping"] <= 8 for row in runner.rows
                            if row.get("lookup_attempted")))
        self.assertNotIn("PRIVATE", json.dumps(report))

    def test_four_agent_http_load_separates_latency_overlap_from_exactness(self):
        engine = Engine()
        engine.generation_delay = .08
        config = Config("fixture", rounds=1, pressure_lineages=1, latency_lookups=40,
                        lookup_interval_ms=10, load_timeout=4, system_lines=1)
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), config)
            runner.latency()
        report = runner.finish()
        load = report["load"]
        self.assertEqual(load["agents"], 4)
        self.assertTrue(all(load["completed_generations_with_tokens"]))
        self.assertEqual(load["latency_all_samples_ms"]["http_wall"]["n"] +
                         load["cadence"]["missed_slots"], 40)
        self.assertGreater(load["samples_fully_covered_by_all_generations"], 0)
        self.assertEqual(report["failures"], [])
        self.assertTrue(all("_generation_interval" not in row for row in report["cases"]))
        self.assertTrue(all(not row.get("failures") for row in report["cases"]
                            if row.get("cache_equal") is False))

    def test_four_agent_busy_http_load_retains_503_latency_and_does_not_fail_generation(self):
        engine = Engine()
        engine.busy_lookup = True
        engine.generation_delay = .08
        config = Config("fixture", latency_lookups=30, lookup_interval_ms=10, load_timeout=4)
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), config)
            runner.latency()
        report = runner.finish()
        load = report["load"]
        self.assertEqual(report["failures"], [])
        self.assertTrue(all(load["completed_generations_with_tokens"]))
        attempts = load["cadence"]["attempted_slots"]
        self.assertEqual(attempts + load["cadence"]["missed_slots"], 30)
        self.assertEqual(load["availability_all_samples"]["busy_503"], attempts)
        self.assertEqual(load["availability_all_samples"]["busy_rate"], 1)
        self.assertEqual(load["availability_all_samples"]["http_status_counts"], {"503": attempts})
        self.assertEqual(load["availability_all_samples"]["error_code_counts"], {"cache_lookup_busy": attempts})
        self.assertEqual(load["latency_all_samples_ms"]["http_503_busy_wall"]["n"], attempts)
        self.assertEqual(load["latency_all_samples_ms"]["engine_lookup"]["n"], 0)
        self.assertIn("no_engine_lookup_ms_under_measured_load", report["uncertainties"])
        self.assertFalse(load["acceptance_target"]["observed_target_met"])
        self.assertTrue(all(row["status"] in ("busy", "missed") for row in report["cases"]))

    def test_external_slots_http_verifies_four_and_eight_busy_sessions_privately(self):
        for clients in (4, 8):
            engine = Engine()
            engine.slots_value = engine.slots_value[:clients]
            config = Config("fixture", clients=0, external_clients=clients, latency_lookups=30,
                            lookup_interval_ms=10, slots_interval_ms=20, load_timeout=3)
            with fake_http(engine) as url:
                report = Harness(Client(url, timeout=2), config).run("load")
            load = report["load"]
            self.assertGreater(load["external_slots"]["bracketed_lookup_samples"], 0)
            self.assertFalse(load["external_load_overlap_unverified"])
            self.assertNotIn("external_generation_load_not_verified", report["uncertainties"])
            self.assertEqual(load["acceptance_target"]["scope"], "all_samples_and_slots_brackets")
            self.assertEqual(load["external_slots"]["processing_counts"]["max"], clients)
            self.assertEqual(load["external_slots"]["error_codes"], {})
            self.assertEqual(report["failures"], [])
            self.assertEqual(report["summary"]["snapshot_metadata_load_samples"]["index_version"]["max"], 0)
            encoded = json.dumps(report)
            self.assertNotIn("PRIVATE", encoded)
            self.assertNotIn("987654321", encoded)
            self.assertNotIn(url, encoded)

    def test_external_slots_errors_remain_unqualified_and_slow_polls_are_counted(self):
        for error in (False, True):
            engine = Engine()
            engine.slots_error = error
            engine.slots_delay = .04
            config = Config("fixture", clients=0, external_clients=4, latency_lookups=20,
                            lookup_interval_ms=10, slots_interval_ms=10, load_timeout=.3)
            with fake_http(engine) as url:
                report = Harness(Client(url, timeout=2), config).run("load")
            load = report["load"]
            self.assertEqual(load["external_slots"]["bracketed_lookup_samples"], 0)
            self.assertGreater(load["external_slots"]["missed_poll_slots"], 0)
            self.assertTrue(load["external_load_overlap_unverified"])
            self.assertIn("external_slots_polling_incomplete", report["uncertainties"])
            self.assertIn("external_generation_load_not_verified", report["uncertainties"])
            if error:
                self.assertGreater(load["external_slots"]["error_codes"]["http_status"], 0)
            self.assertNotIn("PRIVATE", json.dumps(report))

    def test_own_unfinished_body_correlation_measures_underreport_not_tier_cause(self):
        class WarmBusyHarness(Harness):
            def pair(self, label, body, strict=True):
                # Establish the fixture's warm-cache precondition BEFORE pair
                # records the HTTP generation interval. Handler-side warmup is
                # too late: an in-flight lookup can legitimately still see cold.
                delay = engine.generation_delay
                engine.generation_delay = 0
                try:
                    engine.post("/v1/chat/completions", body)
                finally:
                    engine.generation_delay = delay
                return super().pair(label, body, strict)

        # One generation thread avoids mutable fake timing races; the harness's
        # 4/8 production configuration remains covered by the existing tests.
        engine = Engine()
        engine.generation_delay = .12
        engine.lookup_delta = 2
        config = Config("fixture", clients=1, latency_lookups=40,
                        lookup_interval_ms=10, system_lines=1, load_timeout=3)
        with fake_http(engine) as url:
            runner = WarmBusyHarness(Client(url, timeout=2), config)
            report = runner.run("load")
        stats = report["load"]["own_unfinished_conversation_observations"]
        self.assertGreater(stats["fully_in_flight_samples"], 0)
        self.assertGreater(stats["under_report_count"], 0)
        self.assertEqual(stats["under_report_rate"], 1)
        self.assertFalse(stats["suppression_cause_verified"])
        self.assertIsNone(stats["busy_live_frontier_suppression_count"])
        self.assertEqual(report["failures"], [])
        for row in runner.rows:
            if "own_generation_cache_delta_tokens" in row:
                self.assertTrue(row["own_generation_covers_lookup"])
                self.assertEqual(row["own_generation_cache_delta_tokens"], -2)
        self.assertNotIn("PRIVATE", json.dumps(report))

    def test_own_busy_denominator_excludes_uncovered_uncompared_and_general_underreports(self):
        rows = [{"own_generation_case": "fixture", "own_generation_covers_lookup": covered,
                 **({"own_generation_cache_delta_tokens": delta} if delta is not None else {})}
                for covered, delta in ((True, -2), (True, 0), (True, 3), (True, None), (False, None))]
        rows.append({"cache_comparison": "under_report"})
        stats = Harness.own_busy_observations(rows)
        self.assertEqual(stats["targeted_samples"], 5)
        self.assertEqual(stats["fully_in_flight_samples"], 4)
        self.assertEqual(stats["compared_samples"], 3)
        self.assertEqual(stats["uncompared_in_flight_samples"], 1)
        self.assertEqual(stats["under_report_count"], 1)
        self.assertEqual(stats["under_report_rate"], 1 / 3)
        self.assertEqual(stats["over_report_count"], 1)
        self.assertIsNone(Harness.own_busy_observations([])["under_report_rate"])

    def test_cli_reserves_new_private_output_and_refuses_overwrite(self):
        engine = Engine()
        with tempfile.TemporaryDirectory() as directory, fake_http(engine) as url:
            output = Path(directory) / "summary.json"
            args = ["--base-url", url, "--model", "fixture", "--exclusive", "--phase", "exactness",
                    "--rounds", "1", "--pressure-lineages", "1", "--output", str(output)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(args)
                before = output.read_bytes()
                with self.assertRaises(SystemExit):
                    main(args)
            self.assertEqual(code, 2)
            self.assertEqual(before, output.read_bytes())
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            report = json.loads(before)
            self.assertEqual(report["status"], "unqualified")
            self.assertNotIn(url, before.decode())

    def test_cli_load_only_external_and_eight_client_flags(self):
        self.assertEqual(Config("fixture").lookup_interval_ms, 50)
        engine = Engine()
        engine.generation_delay = .06
        with tempfile.TemporaryDirectory() as directory, fake_http(engine) as url:
            for clients in (0, 8):
                output = Path(directory) / f"load_{clients}.json"
                args = ["--base-url", url, "--model", "PRIVATE_MODEL", "--output", str(output),
                        "--load-only", "--clients", str(clients), "--lookup-workers", "3",
                        "--latency-lookups", "5", "--lookup-interval-ms", "20", "--load-timeout", "3"]
                with redirect_stdout(io.StringIO()):
                    code = main(args)
                self.assertEqual(code, 2)
                report = json.loads(output.read_text())
                self.assertEqual(report["config"]["clients"], clients)
                self.assertEqual(report["config"]["lookup_workers"], 3)
                self.assertEqual(report["config"]["phase"], "load")
                self.assertFalse(report["config"]["exclusive_acknowledged"])
                self.assertEqual(report["summary"]["sequential_pairs"], 0)
                self.assertNotIn("fewer_than_200_sequential_comparisons", report["uncertainties"])
                self.assertNotIn("sequential_exactness_not_run", report["uncertainties"])
                self.assertNotIn("PRIVATE", output.read_text())
            base = ["--base-url", url, "--model", "fixture", "--output", str(Path(directory) / "invalid.json")]
            for extra in (["--load-only", "--clients", "2"], ["--load-only", "--lookup-workers", "0"],
                          ["--load-only", "--phase", "latency"], ["--load-only", "--lookup-interval-ms", "nan"],
                          ["--load-only", "--external-clients", "4"],
                          ["--load-only", "--clients", "0", "--slots-interval-ms", "0"],
                          ["--exclusive", "--clients", "0", "--external-clients", "4"]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(base + extra)
            self.assertFalse((Path(directory) / "invalid.json").exists())

    def test_cli_external_slots_flags_enable_private_monitoring(self):
        with tempfile.TemporaryDirectory() as directory, fake_http(Engine()) as url:
            output = Path(directory) / "external.json"
            args = ["--base-url", url, "--model", "PRIVATE_MODEL", "--output", str(output),
                    "--load-only", "--clients", "0", "--external-clients", "8",
                    "--slots-interval-ms", "20", "--latency-lookups", "20",
                    "--lookup-interval-ms", "10", "--request-timeout", "2"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["schema"], "gufo-cache-lookup-probe-v3")
            self.assertEqual(report["config"]["external_clients"], 8)
            self.assertEqual(report["config"]["slots_interval_ms"], 20)
            self.assertGreater(report["load"]["external_slots"]["bracketed_lookup_samples"], 0)
            self.assertNotIn("PRIVATE", output.read_text())

    def test_cli_exactness_requires_exclusive_and_rejects_rendered_pin_text(self):
        with tempfile.TemporaryDirectory() as directory:
            pin = Path(directory) / "pin.txt"
            pin.write_text("PRIVATE_RENDERED_PIN")
            base = ["--base-url", "http://127.0.0.1:1", "--model", "fixture",
                    "--output", str(Path(directory) / "unused.json")]
            with redirect_stderr(io.StringIO()):
                for extra in ([], ["--exclusive", "--pin-prompt-file", str(pin)]):
                    with self.assertRaises(SystemExit):
                        main(base + extra)
            self.assertFalse((Path(directory) / "unused.json").exists())


if __name__ == "__main__":
    unittest.main()
