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
                          generation_metrics, lookup_metrics, main, png_url)


class Engine:
    """Independent exact-key cache oracle; not a model/cache implementation test."""

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.epoch = 0
        self.calls = []
        self.lock = threading.Lock()
        self.generation_delay = 0
        self.cache_delta = 0
        self.total_delta = 0
        self.fail_lookup = False
        self.busy_lookup = False
        self.fail_generation = False
        self.invalid_lookup = None

    def post(self, endpoint, body):
        body = deepcopy(body)
        if endpoint == "/v1/chat/completions":
            time.sleep(self.generation_delay)
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
                result = {"total_tokens": total + self.total_delta,
                          "matched_tokens": cached, "tier": "ram" if cached else "none",
                          "epoch": "fixture-opaque-epoch-" + str(self.epoch), "lookup_ms": .125,
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

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
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
        runner = Harness(engine, Config("fixture"))
        runner.sequential()
        report = runner.finish()
        self.assertEqual(report["summary"]["sequential_pairs"], 320 + 140 + 9)
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

    def test_strict_mismatches_fail_but_concurrent_cache_mismatch_is_uncertainty(self):
        engine = Engine()
        runner = self.runner(engine)
        body = runner.body("fixture")
        runner.pair("seed", body)
        engine.cache_delta = 1
        _, strict = runner.pair("strict", body)
        _, concurrent = runner.pair("concurrent", body, strict=False)
        self.assertIn("cached_tokens_mismatch", strict["failures"])
        self.assertEqual(concurrent["status"], "passed")
        self.assertEqual(concurrent["uncertainty"], "concurrent_mutation_not_a_reservation")
        concurrent.pop("_generation_interval")
        report = runner.finish()
        self.assertEqual(report["summary"]["concurrent_cache_mismatches_not_failures"], 1)
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

    def test_four_agent_http_load_separates_latency_overlap_from_exactness(self):
        engine = Engine()
        engine.generation_delay = .08
        config = Config("fixture", rounds=1, pressure_lineages=1, latency_lookups=40,
                        lookup_interval_ms=2, load_timeout=4, system_lines=1)
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), config)
            runner.latency()
        report = runner.finish()
        load = report["load"]
        self.assertEqual(load["agents"], 4)
        self.assertTrue(all(load["completed_generations_with_tokens"]))
        self.assertEqual(load["latency_all_samples_ms"]["http_wall"]["n"], 40)
        self.assertGreater(load["samples_fully_covered_by_four_generations"], 0)
        self.assertEqual(report["failures"], [])
        self.assertTrue(all("_generation_interval" not in row for row in report["cases"]))
        self.assertTrue(all(not row.get("failures") for row in report["cases"]
                            if row.get("cache_equal") is False))

    def test_four_agent_busy_http_load_retains_503_latency_and_does_not_fail_generation(self):
        engine = Engine()
        engine.busy_lookup = True
        engine.generation_delay = .08
        config = Config("fixture", latency_lookups=30, lookup_interval_ms=2, load_timeout=4)
        with fake_http(engine) as url:
            runner = Harness(Client(url, timeout=2), config)
            runner.latency()
        report = runner.finish()
        load = report["load"]
        self.assertEqual(report["failures"], [])
        self.assertTrue(all(load["completed_generations_with_tokens"]))
        self.assertEqual(load["availability_all_samples"]["busy_503"], 30)
        self.assertEqual(load["availability_all_samples"]["busy_rate"], 1)
        self.assertEqual(load["latency_all_samples_ms"]["http_503_busy_wall"]["n"], 30)
        self.assertEqual(load["latency_all_samples_ms"]["engine_lookup"]["n"], 0)
        self.assertIn("no_engine_lookup_ms_under_full_four_agent_load", report["uncertainties"])
        self.assertTrue(all(row["status"] == "busy" for row in report["cases"]))

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
