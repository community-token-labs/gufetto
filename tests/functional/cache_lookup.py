#!/usr/bin/env python3
r"""URL-only cache lookup exactness and concurrent HTTP load probe (stdlib).

The operator must enable Gufo's --cache-lookup-api, allocate the server, complete
its profiling/operation-lock gates, and exclude other clients for exactness.
This script never starts/stops services, clears caches, or loads local models.

Example (run only after the operator's GPU gate/lock):
  python3 /absolute/gufo/tests/functional/cache_lookup.py \
    --base-url http://127.0.0.1:8080 --model MODEL --exclusive \
    --output /tmp/cache-lookup.json

Default: 469 sequential pairs (320 history, 140 pressure, 9 anchor probes);
then 400 fixed-cadence lookup samples beside --clients 4 (or 8) generation clients.
Every sequential lookup is immediately followed by the SAME buffered Chat request.
Each history contrasts preserve_thinking true/false on the same appended
messages with synthetic historical reasoning; generation reasoning stays off.
Concurrent pairs FAIL if matched_tokens exceeds the following identical real
request's cached_tokens; under-reports are counted separately. These are observed
outcomes, not a reservation guarantee: intervening/queued work may change reuse.
HTTP 503 cache_lookup_busy is counted as unavailable, never retried or hidden.
Successful-only HTTP (http_200_wall) and engine p50/p95/p99 are reported separately
from errors; http_wall additionally retains timings for all outcomes.

--load-only skips idle exactness; --clients 0 sends no generation traffic, for
use alongside an independently launched docs bench/bench.py workload. External
load: add --external-clients 4 (or 8) to poll /slots at 20Hz. Only lookups
bracketed by adjacent polls with the same required number of processing task IDs
qualify; this is sampled server-state evidence, not continuous GPU execution or
proof that a processing request belongs to bench rather than another client.
Run bench/bench.py separately with its own 4/8-session options, then run:
  python3 tests/functional/cache_lookup.py --base-url URL --model MODEL \
    --load-only --clients 0 --external-clients 4 --output /tmp/external-4.json
Use --external-clients 8 for the 8-session run. Without this option external
overlap remains unverified. No benchmark process is launched or changed here.
Default lookup cadence is 20Hz
(--lookup-interval-ms 50), independent of response latency, with bounded parallel
--lookup-workers. Saturated client capacity produces explicit missed slots,
not queued requests or silent rate reduction. In-flight HTTP is not GPU overlap.
The report checks observed availability=100% and successful HTTP p99<5ms;
missing overlap/samples or unmet targets remain visibly unqualified.

--pin-prompt-file takes a private Chat JSON object with the original matching
messages/tools/template settings, NOT an already-rendered --pin-prefix-file.
It is sent only to the supplied URL and never copied/hashed into the report.
--pin-min-matched-tokens can assert the operator's known registered depth; this
alone cannot distinguish a durable pin from ordinary warmed state.
--images adds 8 tiny generated PNG cases (477 sequential pairs with defaults);
the server must already have vision loaded.

JSON retains only allowlisted counts/timings, synthetic labels and error codes;
no prompts, outputs, credentials, URLs, model names or server error messages.
No retries: all failures survive. HTTP wall includes response-body receipt but
not client JSON parsing; lookup_ms is engine-reported, not inferred GPU time.
index_version (uint64) and snapshot_age_ms (finite nonnegative double) are
required (report schema v3); old responses without these fields fail schema validation. Historical
restore/prefill estimates remain optional/nullable for compatibility. Snapshot
age and version stats are reported; concurrent versions need not arrive in order.
Own unfinished-conversation probes correlate the identical in-flight body with
that generation's eventual accounting. Under-report frequency is observational,
not proof of busy-live-frontier suppression; tier=none alone proves no cause.
No busy_live_frontiers field is assumed until its server definition is stable.
Percentiles use nearest rank. Exit 0: full probe passed, 1: failure,
2: incomplete/unqualified (including too few samples or unobserved eviction).
This is not a matched-revision speed regression gate or a restart/disk test.
"""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from http.client import HTTPException
import json
import math
import os
import re
from pathlib import Path
import struct
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import zlib


TOOLS = [{"type": "function", "function": {
    "name": "read_fixture", "description": "Read a synthetic record; never executed.",
    "parameters": {"type": "object", "properties": {"index": {"type": "integer"}},
                   "required": ["index"], "additionalProperties": False}}}]
MIN_SAMPLES = 200
PRESSURE_LINEAGES = 140


class ProbeError(Exception):
    """Only static codes/numeric data may escape into the report."""

    def __init__(self, code, **metrics):
        super().__init__(code)
        self.code, self.metrics = code, metrics


def integer(value):
    return type(value) is int and value >= 0


def number(value):
    if type(value) is int:
        return value >= 0
    return type(value) is float and math.isfinite(value) and value >= 0


def token_count(value):
    # Some timings serializers emit integer-valued JSON floats (e.g. 123.0).
    return integer(value) or (type(value) is float and number(value) and value.is_integer())


def receipt_request_id(value):
    candidate = value.get("_receipt_request_id")
    return candidate if isinstance(candidate, str) and re.fullmatch(r"r[0-9]{1,20}", candidate) else None


def lookup_metrics(value):
    if not isinstance(value, dict):
        raise ProbeError("lookup_not_object")
    for key in ("total_tokens", "matched_tokens"):
        if not integer(value.get(key)):
            raise ProbeError("lookup_invalid_" + key)
    if not isinstance(value.get("epoch"), str) or not value["epoch"]:
        raise ProbeError("lookup_invalid_epoch")
    if value.get("tier") not in ("live", "ram", "disk", "none"):
        raise ProbeError("lookup_invalid_tier")
    if value["matched_tokens"] > value["total_tokens"]:
        raise ProbeError("lookup_match_exceeds_total")
    if (value["tier"] == "none") != (value["matched_tokens"] == 0):
        raise ProbeError("lookup_tier_count_disagree")
    if not number(value.get("lookup_ms")):
        raise ProbeError("lookup_invalid_lookup_ms")
    if not integer(value.get("index_version")) or value["index_version"] > 2**64 - 1:
        raise ProbeError("lookup_invalid_index_version")
    if (not number(value.get("snapshot_age_ms")) or
            value["snapshot_age_ms"] > float.fromhex("0x1.fffffffffffffp+1023")):
        raise ProbeError("lookup_invalid_snapshot_age_ms")
    result = {key: value[key] for key in (
        "total_tokens", "matched_tokens", "tier", "epoch", "lookup_ms",
        "index_version", "snapshot_age_ms")}
    for key in ("restore_estimate_ms", "prefill_estimate_ms"):
        if key in value:
            if value[key] is not None and not number(value[key]):
                raise ProbeError("lookup_invalid_" + key)
            result[key] = value[key]
    if request_id := receipt_request_id(value):
        result["request_id"] = request_id
    return result


def generation_metrics(value):
    try:
        usage, timing = value["usage"], value["timings"]
        result = {"total_tokens": usage["prompt_tokens"],
                  "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"],
                  "cache_n": timing["cache_n"], "prompt_n": timing["prompt_n"],
                  "completion_tokens": usage["completion_tokens"]}
    except (KeyError, TypeError):
        raise ProbeError("generation_missing_accounting") from None
    if not all(token_count(v) for v in result.values()):
        raise ProbeError("generation_invalid_accounting")
    result = {key: int(value) for key, value in result.items()}
    if result["cached_tokens"] != result["cache_n"]:
        raise ProbeError("generation_cache_counters_disagree")
    if result["cache_n"] + result["prompt_n"] != result["total_tokens"]:
        raise ProbeError("generation_prompt_accounting_disagrees")
    if ("total_tokens" in usage and
            (not integer(usage["total_tokens"]) or
             usage["total_tokens"] != result["total_tokens"] + result["completion_tokens"])):
        raise ProbeError("generation_usage_total_disagrees")
    for key in ("draft_rounds", "draft_n", "draft_n_accepted"):
        if key in timing:
            if not integer(timing[key]):
                raise ProbeError("generation_invalid_" + key)
            result[key] = timing[key]
    for key in ("prompt_ms", "predicted_ms", "cache_restore_ms", "cache_snapshot_ms", "cache_disk_enqueue_ms"):
        if key in timing:
            if not number(timing[key]):
                raise ProbeError("generation_invalid_" + key)
            result[key] = timing[key]
    # Keep only numeric/static diagnostic metrics, never the rest of the reply.
    component = usage.get("gufo", {}).get("component_cache", {}) if isinstance(usage.get("gufo", {}), dict) else {}
    if isinstance(component, dict) and "restore_attempts" in component:
        metrics = {}
        for key in ("acquisition_id", "admission_index_version", "selected_entry_id",
                    "restore_attempts", "restore_failures"):
            if not integer(component.get(key)) or component[key] > 2**64 - 1:
                raise ProbeError("generation_invalid_component_" + key)
            metrics[key] = component[key]
        reason = component.get("restore_failure_reason")
        if reason not in ("none", "ram_begin_restore", "ram_pin", "ram_transfer", "ram_validate",
                          "ram_admission", "disk_streaming_unavailable", "disk_restore_exception",
                          "disk_restore_rejected"):
            raise ProbeError("generation_invalid_restore_failure_reason")
        metrics["restore_failure_reason"] = reason
        if metrics["restore_failures"] > metrics["restore_attempts"]:
            raise ProbeError("generation_invalid_restore_counters")
        result["component_cache"] = metrics
    if request_id := receipt_request_id(value):
        result["request_id"] = request_id
    return result


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, base_url, timeout=120, api_key=None):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc or
                parsed.username is not None or parsed.password is not None or
                parsed.query or parsed.fragment):
            raise ProbeError("invalid_base_url")
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.timeout, self.api_key = timeout, api_key
        self.local = threading.local()

    def post(self, endpoint, body):
        return self.request(endpoint, body)

    def get(self, endpoint):
        return self.request(endpoint)

    def request(self, endpoint, body=None):
        # No ambient proxy or redirect can forward a private pin request elsewhere.
        if not hasattr(self.local, "opener"):
            self.local.opener = build_opener(ProxyHandler({}), NoRedirect())
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(self.root + endpoint,
                          data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                          headers=headers)
        start = time.perf_counter()
        status = None
        try:
            with self.local.opener.open(request, timeout=self.timeout) as response:
                status = response.status
                request_id = response.headers.get("X-Request-ID")
                raw = response.read(8 * 1024 * 1024 + 1)
            wall = (time.perf_counter() - start) * 1000
            if len(raw) > 8 * 1024 * 1024:
                raise ProbeError("response_too_large", http_status=status, wall_ms=wall)
            if status != 200:
                raise ProbeError("http_status", http_status=status, wall_ms=wall)
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError):
                raise ProbeError("invalid_response_json", http_status=status, wall_ms=wall) from None
            if isinstance(value, dict):
                # Replace, never trust a body field masquerading as a header.
                value.pop("_receipt_request_id", None)
                if isinstance(request_id, str) and re.fullmatch(r"r[0-9]{1,20}", request_id):
                    value["_receipt_request_id"] = request_id
            return value, wall
        except HTTPError as error:
            code = "http_status"
            wall = None
            try:
                # Read the error body for this single recognized code only;
                # never retain arbitrary server messages, codes, or echoed text.
                raw = error.read(8 * 1024 * 1024 + 1)
                wall = (time.perf_counter() - start) * 1000
                if (endpoint == "/v1/cache/lookup" and error.code == 503 and
                        len(raw) <= 8 * 1024 * 1024):
                    value = json.loads(raw)
                    if (isinstance(value, dict) and isinstance(value.get("error"), dict) and
                            value["error"].get("code") == "cache_lookup_busy"):
                        code = "cache_lookup_busy"
            except (OSError, ValueError, UnicodeError):
                pass
            finally:
                error.close()
            raise ProbeError(code, http_status=error.code,
                             wall_ms=wall if wall is not None else
                             (time.perf_counter() - start) * 1000) from None
        except (OSError, URLError, HTTPException):
            raise ProbeError("transport_error", http_status=status,
                             wall_ms=(time.perf_counter() - start) * 1000) from None


def processing_tasks(value):
    """Allowlist metadata only; task identities stay in memory for correlation."""
    if not isinstance(value, list):
        raise ProbeError("slots_invalid_metadata")
    tasks, ids = set(), set()
    for slot in value:
        if (not isinstance(slot, dict) or not integer(slot.get("id")) or
                slot["id"] in ids or type(slot.get("is_processing")) is not bool):
            raise ProbeError("slots_invalid_metadata")
        ids.add(slot["id"])
        if slot["is_processing"]:
            if not integer(slot.get("id_task")):
                raise ProbeError("slots_invalid_metadata")
            tasks.add((slot["id"], slot["id_task"]))
    return frozenset(tasks)


def bracketed_tasks(start, end, polls, period):
    # Adjacent successful polls only: errors, task churn, slow/missed polling,
    # or a lookup spanning a poll cannot be silently bridged as continuous load.
    for left, right in zip(polls, polls[1:]):
        a, b, before, _ = left
        c, d, after, _ = right
        if (before is not None and after is not None and b <= start <= end <= c
                and d - a <= 3 * period):
            return len(before & after)
    return 0


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    result = {"n": len(values), "max": values[-1]}
    for label, p in (("p50", .50), ("p95", .95), ("p99", .99)):
        result[label] = values[max(0, math.ceil(p * len(values)) - 1)]
    return result


@dataclass
class Config:
    model: str
    rounds: int = 32
    pressure_lineages: int = PRESSURE_LINEAGES
    system_lines: int = 32
    pressure_lines: int = 64
    max_tokens: int = 8
    latency_lookups: int = 400
    load_max_tokens: int = 64
    lookup_interval_ms: float = 50
    clients: int = 4
    lookup_workers: int = 16
    external_clients: int = 0
    slots_interval_ms: float = 50
    load_timeout: float = 180
    images: bool = False
    pin_min_matched_tokens: int = 0
    required_tiers: tuple = ()


def png_url(rgb, metadata=False):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data +
                struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    raw = (b"\0" + bytes(rgb) * 64) * 64
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
    if metadata:
        data += chunk(b"tEXt", b"fixture\0same pixels, different encoded bytes")
    data += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


class Harness:
    def __init__(self, client, config, pin_body=None):
        self.client, self.config, self.pin_body = client, config, pin_body
        self.rows, self.failures, self.uncertainties = [], [], []
        self.targets = []  # Memory only: never serialized.
        self.lock = threading.Lock()
        self.report = {"schema": "gufo-cache-lookup-probe-v3", "status": "running",
                       "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "config": {key: value for key, value in vars(config).items() if key != "model"},
                       "cases": self.rows, "failures": self.failures,
                       "uncertainties": self.uncertainties,
                       "scope": {"sequential_requires_exclusive_server": True,
                                 "concurrent_lookup_is_reservation": False,
                                 "latency_regression_qualified": False,
                                 "service_manipulation": False}}

    def fail(self, row, code, **metrics):
        item = {"case": row["case"], "phase": row["phase"], "code": code, **metrics}
        with self.lock:
            self.failures.append(item)
            row.setdefault("failures", []).append(code)

    def uncertain(self, code):
        if code not in self.uncertainties:
            self.uncertainties.append(code)

    def body(self, lineage, text="Reply with OK.", tools=False, lines=None):
        lines = self.config.system_lines if lines is None else lines
        system = (f"Synthetic lookup lineage {lineage}. Follow the final user instruction.\n" +
                  "A record has an integer key, a status, and a revision.\n" * lines)
        body = {"model": self.config.model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": text}],
            "temperature": 0, "seed": 31337, "reasoning_effort": "none",
            "max_completion_tokens": self.config.max_tokens, "stream": False}
        if tools:
            body.update(tools=deepcopy(TOOLS), tool_choice="none")
        return body

    def lookup(self, row, body, allow_busy=False):
        try:
            value, wall = self.client.post("/v1/cache/lookup", body)
            row.update(lookup_wall_ms=wall, lookup_http_status=200)
            row["lookup"] = lookup_metrics(value)
            return row["lookup"]
        except ProbeError as error:
            if "wall_ms" in error.metrics:
                row["lookup_wall_ms"] = error.metrics["wall_ms"]
            row["lookup_http_status"] = error.metrics.get("http_status", row.get("lookup_http_status"))
            row["lookup_error_code"] = error.code
            if allow_busy and error.code == "cache_lookup_busy" and error.metrics.get("http_status") == 503:
                row["lookup_busy"] = True
            else:
                self.fail(row, error.code, **error.metrics)
            return None

    def pair(self, label, body, strict=True):
        row = {"case": label, "phase": "sequential" if strict else "load_diagnostic"}
        with self.lock:
            self.rows.append(row)
        actual, message = None, None
        lookup = self.lookup(row, body, allow_busy=not strict)
        # A failed/busy lookup must not hide generation failure or stop later cases.
        try:
            start = time.perf_counter()
            try:
                value, wall = self.client.post("/v1/chat/completions", body)
            finally:
                if not strict:
                    row["_generation_interval"] = (start, time.perf_counter())
            row["generation_wall_ms"] = wall
            actual = row["actual"] = generation_metrics(value)
            choices = value.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ProbeError("generation_missing_message")
            if not isinstance(choices[0], dict):
                raise ProbeError("generation_invalid_choice")
            message = choices[0].get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ProbeError("generation_invalid_message")
            message = deepcopy(message)
            if body.get("cache_prompt") is False and actual["cached_tokens"] != 0:
                self.fail(row, "cache_prompt_false_reused", actual=actual["cached_tokens"])
        except ProbeError as error:
            self.fail(row, error.code, **error.metrics)
        if lookup is not None and actual is not None:
            if lookup["total_tokens"] != actual["total_tokens"]:
                self.fail(row, "total_tokens_mismatch", predicted=lookup["total_tokens"],
                          actual=actual["total_tokens"])
            matched = lookup["matched_tokens"] == actual["cached_tokens"]
            row["cache_equal"] = matched
            if strict and not matched:
                self.fail(row, "cached_tokens_mismatch", predicted=lookup["matched_tokens"],
                          actual=actual["cached_tokens"])
            if not strict:
                delta = lookup["matched_tokens"] - actual["cached_tokens"]
                row["cache_delta_tokens"] = delta
                row["cache_comparison"] = "over_report" if delta > 0 else "under_report" if delta < 0 else "equal"
                if delta > 0:
                    self.fail(row, "concurrent_cached_tokens_over_report",
                              predicted=lookup["matched_tokens"], actual=actual["cached_tokens"])
            if body.get("cache_prompt") is False and lookup["matched_tokens"] != 0:
                self.fail(row, "lookup_cache_prompt_false_reused")
        row["status"] = self.row_status(row)
        return message, row

    def append(self, body, message, text):
        result = deepcopy(body)
        result["messages"] += [deepcopy(message), {"role": "user", "content": text}]
        return result

    def sequential(self):
        for index in range(self.config.rounds):
            label = f"history_{index:03d}"
            seed = self.body(label, tools=bool(index % 2), lines=self.config.system_lines + index % 4)
            message, _ = self.pair(label + "_seed", seed)
            self.pair(label + "_regenerate", seed)
            if message is not None:
                appended = self.append(seed, message, f"Turn {index}: reply with OK again.")
                next_message, _ = self.pair(label + "_append", appended)
                # Exercise rendering of retained historical reasoning without
                # enabling reasoning generation or mutating the ordinary append.
                historical = deepcopy(appended)
                historical["messages"][-2]["reasoning_content"] = (
                    "Synthetic historical reasoning: inspect the record's status before answering OK.")
                for preserve in (True, False):
                    replay = deepcopy(historical)
                    replay.setdefault("chat_template_kwargs", {})["preserve_thinking"] = preserve
                    self.pair(label + "_preserve_thinking_" + str(preserve).lower(), replay)
                if next_message is not None:
                    tool_history = self.append(appended, next_message, "Reply with OK after this record.")
                    tool_history["messages"][-1:-1] = [
                        {"role": "assistant", "content": None, "tool_calls": [{
                            "id": "fixture-call", "type": "function", "function": {
                                "name": "read_fixture", "arguments": '{"index": 7}'}}]},
                        {"role": "tool", "tool_call_id": "fixture-call", "content": "status: ready"}]
                    tool_history.update(tools=deepcopy(TOOLS), tool_choice="none")
                    self.pair(label + "_tool_append", tool_history)
                else:
                    self.uncertain("tool_append_dependency_failed")
            else:
                self.uncertain("append_dependency_failed")
            # Rewind drops later turns; editing the old user turn forks history.
            rewind = deepcopy(seed)
            rewind["messages"][-1]["content"] = f"Rewritten turn {index}: reply with YES."
            self.pair(label + "_rewind", rewind)
            self.pair(label + "_rewind_regenerate", rewind)
            newchat = deepcopy(seed)
            newchat["messages"][-1]["content"] = f"New chat {index} with shared system/tools: reply with NO."
            self.pair(label + "_newchat_shared", newchat)
            cold = deepcopy(seed)
            cold["cache_prompt"] = False
            self.pair(label + "_cache_prompt_false", cold)
            self.targets.append(seed)
        self.pressure()
        if self.pin_body is not None:
            for index in range(4):
                _, row = self.pair(f"pin_{index}", self.pin_body)
                if "lookup" in row and row["lookup"]["matched_tokens"] < self.config.pin_min_matched_tokens:
                    self.fail(row, "pin_depth_not_reached")
                    row["status"] = "failed"
            self.uncertain("pin_provenance_not_verifiable_from_lookup_api")
        if self.config.images:
            self.image_cases()

    def pressure(self):
        anchors = [self.body(f"pressure_anchor_{i}", tools=True) for i in range(3)]
        before = []
        for index, body in enumerate(anchors):
            self.pair(f"anchor_{index}_seed", body)
            _, row = self.pair(f"anchor_{index}_before", body)
            before.append(row.get("lookup", {}).get("matched_tokens"))
        for index in range(self.config.pressure_lineages):
            self.pair(f"unrelated_{index:03d}", self.body(
                f"unrelated_{index:03d}", lines=self.config.pressure_lines))
        drops = 0
        for index, body in enumerate(anchors):
            _, row = self.pair(f"anchor_{index}_after", body)
            after = row.get("lookup", {}).get("matched_tokens")
            if before[index] is not None and after is not None and after < before[index]:
                drops += 1
        self.targets.extend(anchors)
        self.report["pressure"] = {"unrelated_lineages": self.config.pressure_lineages,
                                   "anchors_with_retention_drop": drops,
                                   "entry_eviction_directly_observable": False}
        if not drops:
            self.uncertain("eviction_not_observed_no_anchor_retention_drop")
        if self.config.pressure_lineages < PRESSURE_LINEAGES:
            self.uncertain("too_few_pressure_lineages")

    def image_cases(self):
        base = self.body("image_identity")
        text = base["messages"][-1]["content"]

        def image_body(rgb, metadata=False):
            body = deepcopy(base)
            body["messages"][-1]["content"] = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": png_url(rgb, metadata)}}]
            return body

        red = image_body((255, 0, 0))
        self.pair("image_seed", red)
        _, repeat = self.pair("image_same_pixels", red)
        _, reencoded = self.pair("image_same_pixels_new_encoding", image_body((255, 0, 0), True))
        _, changed = self.pair("image_changed_pixels", image_body((0, 0, 255)))
        self.pair("image_removed", base)
        if "lookup" in repeat and "lookup" in reencoded:
            if repeat["lookup"]["total_tokens"] != reencoded["lookup"]["total_tokens"]:
                self.fail(reencoded, "image_reencoding_changed_token_count")
        if "lookup" in repeat and "lookup" in changed:
            if changed["lookup"]["matched_tokens"] == changed["lookup"]["total_tokens"]:
                self.fail(changed, "image_changed_pixels_reused_full_prompt")
            elif repeat["lookup"]["matched_tokens"] <= changed["lookup"]["matched_tokens"]:
                self.uncertain("image_identity_no_deep_warm_vs_changed_contrast")
        message, _ = self.pair("image_text_stem", base)
        if message is not None:
            appended = self.append(base, message, "Reply with OK.")
            appended["messages"][-1]["content"] = red["messages"][-1]["content"]
            self.pair("image_appended", appended)
            self.pair("image_appended_regenerate", appended)

    def latency(self):
        clients = self.config.clients
        stop = threading.Event()
        ready = threading.Barrier(clients + 1)
        active, agent_bodies = set(), {}
        condition = threading.Condition()
        intervals, samples, slots = [], [], []
        polls, poll_misses = [], []
        deadline = time.perf_counter() + self.config.load_timeout
        monitor_stop = threading.Event()
        external = self.config.external_clients
        poll_period = self.config.slots_interval_ms / 1000

        def poll_slots():
            start = time.perf_counter()
            try:
                value, _ = self.client.get("/slots")
                tasks = processing_tasks(value)
                polls.append((start, time.perf_counter(), tasks, None))
            except ProbeError as error:
                # Error text, slot prompts, model names and task IDs never escape.
                polls.append((start, time.perf_counter(), None, error.code))
            except Exception:
                polls.append((start, time.perf_counter(), None, "slots_worker_exception"))

        def monitor():
            origin, index = time.perf_counter(), 0
            while not monitor_stop.is_set() and time.perf_counter() < deadline:
                scheduled = origin + index * poll_period
                if monitor_stop.wait(max(0, scheduled - time.perf_counter())):
                    break
                now = time.perf_counter()
                if now - scheduled >= poll_period:
                    skipped = int((now - scheduled) / poll_period)
                    poll_misses.append(skipped)
                    index += skipped
                poll_slots()
                index += 1

        capacity = threading.BoundedSemaphore(self.config.lookup_workers)

        def worker(agent):
            turn, body = 0, None
            ready.wait()
            while not stop.is_set() and time.perf_counter() < deadline:
                if turn % 4 == 0 or body is None:
                    body = self.body(f"load_{agent}_{turn // 4}", tools=True,
                                     text="Write numbered lines from 1 to 100, each explaining a simple integer.")
                    body["max_completion_tokens"] = self.config.load_max_tokens
                with condition:
                    active.add(agent)
                    agent_bodies[agent] = (body, f"load_agent_{agent}_turn_{turn:04d}")
                    condition.notify_all()
                try:
                    message, row = self.pair(f"load_agent_{agent}_turn_{turn:04d}", body, strict=False)
                finally:
                    with condition:
                        active.discard(agent)
                intervals.append((agent, row))
                if message is not None:
                    body = self.append(body, message, "Continue the numbered list for another 100 lines.")
                turn += 1
                if row.get("failures"):
                    stop.wait(.05)  # Bound error storms; never erase failures.

        def sample(row, body, scheduled):
            start = time.perf_counter()
            row["dispatch_lag_ms"] = max(0, (start - scheduled) * 1000)
            row["lookup_attempted"] = True
            try:
                self.lookup(row, body, allow_busy=True)
            except Exception:
                row["lookup_error_code"] = "worker_exception"
                self.fail(row, "worker_exception")
            finally:
                samples.append((start, time.perf_counter(), row))
                row["status"] = self.row_status(row)
                capacity.release()

        monitor_thread = threading.Thread(target=monitor, daemon=True) if external else None
        if monitor_thread:
            monitor_thread.start()
            # Startup is bounded; insufficient external load remains unqualified.
            until = min(deadline, time.perf_counter() + 5)
            while time.perf_counter() < until and not any(
                    tasks is not None and len(tasks) >= external for _, _, tasks, _ in polls):
                time.sleep(min(.01, poll_period))

        # Separate pools: slow lookups must not consume generation workers or
        # serialize the launch clock. No unbounded queue of stale lookup work.
        with ThreadPoolExecutor(max_workers=max(1, clients)) as pool, \
                ThreadPoolExecutor(max_workers=self.config.lookup_workers) as lookup_pool:
            futures = [pool.submit(worker, agent) for agent in range(clients)]
            sample_futures = []
            ready.wait()
            with condition:
                condition.wait_for(lambda: len(active) == clients,
                                   timeout=min(5, self.config.load_timeout))
            origin = time.perf_counter()
            period = self.config.lookup_interval_ms / 1000
            try:
                for index in range(self.config.latency_lookups):
                    scheduled = origin + index * period
                    if scheduled >= deadline:
                        self.uncertain("load_deadline_before_all_samples")
                        break
                    stop.wait(max(0, scheduled - time.perf_counter()))
                    now = time.perf_counter()
                    if now >= deadline:
                        self.uncertain("load_deadline_before_all_samples")
                        break
                    row = {"case": f"latency_{index:04d}", "phase": "load_lookup",
                           "scheduled_offset_ms": index * self.config.lookup_interval_ms,
                           "lookup_attempted": False}
                    self.rows.append(row)
                    slots.append(row)
                    # Do not burst-replay slots missed by the launch thread.
                    if period and now - scheduled >= period:
                        row["lookup_error_code"] = "client_cadence_missed"
                        row["status"] = "missed"
                        continue
                    if not capacity.acquire(blocking=False):
                        row["lookup_error_code"] = "client_capacity_missed"
                        row["status"] = "missed"
                        continue
                    with condition:
                        current = agent_bodies.get((index // 2) % clients) if clients and index % 2 == 0 else None
                    body = current[0] if current else (self.targets[index % len(self.targets)]
                                                      if self.targets else self.body("latency"))
                    row["target_kind"] = "current_agent_history" if current else "prior_history"
                    if current:
                        row["own_generation_case"] = current[1]
                    sample_futures.append(lookup_pool.submit(sample, row, body, scheduled))
                # Keep load active until the final sample finishes, not merely
                # until its launch. Each HTTP request has its own timeout.
                for future in sample_futures:
                    try:
                        future.result()
                    except Exception:
                        self.fail({"case": "lookup_worker", "phase": "load_lookup"}, "worker_exception")
            finally:
                stop.set()
                monitor_stop.set()
            for future in futures:
                try:
                    future.result()
                except Exception:
                    self.fail({"case": "load_worker", "phase": "load_diagnostic"}, "worker_exception")
        if monitor_thread:
            # A final poll brackets even the last completed lookup. Do not race
            # poll_slots with the monitor's HTTP request.
            monitor_stop.set()
            monitor_thread.join()
            poll_slots()
        completed = Counter()
        own_generations = {row["case"]: row for _, row in intervals}
        for start, end, row in samples:
            generation = own_generations.get(row.get("own_generation_case"))
            if generation is not None:
                a, b = generation["_generation_interval"]
                row["own_generation_covers_lookup"] = a <= start and b >= end
                if row["own_generation_covers_lookup"] and "lookup" in row and "actual" in generation:
                    actual = generation["actual"]
                    row["own_generation_total_agrees"] = row["lookup"]["total_tokens"] == actual["total_tokens"]
                    if row["own_generation_total_agrees"]:
                        # This is an in-flight observation, not the pre-request
                        # pair contract: no causal inference from this delta.
                        row["own_generation_cache_delta_tokens"] = (
                            row["lookup"]["matched_tokens"] - actual["cached_tokens"])
        windows = {agent: [] for agent in range(clients)}
        for agent, row in intervals:
            start, end = row.pop("_generation_interval")
            windows[agent].append((start, end))
            if row.get("actual", {}).get("completion_tokens", 0) > 0:
                completed[agent] += 1
        for start, end, row in samples:
            row["generation_agents_overlapping"] = sum(
                any(a < end and b > start for a, b in windows[agent]) for agent in windows)
            row["generation_agents_covering_entire_lookup"] = sum(
                any(a <= start and b >= end for a, b in windows[agent]) for agent in windows)
        covered = [row for _, _, row in samples if clients and
                   row["generation_agents_covering_entire_lookup"] == clients]
        attempted = [row for row in slots if row["lookup_attempted"]]
        missed = len(slots) - len(attempted)
        external_covered = []
        if external:
            for start, end, row in samples:
                row["external_processing_tasks_bracketed"] = bracketed_tasks(
                    start, end, polls, poll_period)
                if row["external_processing_tasks_bracketed"] >= external:
                    external_covered.append(row)
        measured = covered if clients else external_covered if external else attempted
        available = self.availability(measured)["available_rate"]
        p99 = self.latencies(measured)["http_200_wall"]["p99"]
        all_available = self.availability(attempted)["available_rate"]
        all_p99 = self.latencies(attempted)["http_200_wall"]["p99"]
        target_met = (available == 1 and p99 is not None and p99 < 5 and
                      all_available == 1 and all_p99 is not None and all_p99 < 5)
        starts = sorted(start for start, _, _ in samples)
        self.report["load"] = {
            "agents": clients,
            "external_load_overlap_unverified": clients == 0 and not external_covered,
            "external_slots": {
                "required_processing_tasks": external,
                "poll_interval_ms": self.config.slots_interval_ms,
                "polls": len(polls), "missed_poll_slots": sum(poll_misses),
                "includes_final_tail_bracket_poll": bool(external),
                "poll_http_wall_ms": distribution([(end - start) * 1000
                                                   for start, end, _, _ in polls]),
                "observed_poll_intervals_ms": distribution([
                    (right[0] - left[0]) * 1000 for left, right in zip(polls, polls[1:])]),
                "successful_polls": sum(tasks is not None for _, _, tasks, _ in polls),
                "processing_counts": distribution([len(tasks) for _, _, tasks, _ in polls
                                                   if tasks is not None]),
                "error_codes": dict(Counter(error for _, _, _, error in polls if error)),
                "bracketed_lookup_samples": len(external_covered),
                "latency_bracketed_samples_ms": self.latencies(external_covered),
                "availability_bracketed_samples": self.availability(external_covered),
                "snapshot_metadata_bracketed_samples": self.snapshot_stats(external_covered),
                "workload_attribution_verified": False,
                "evidence": "adjacent_polls_same_processing_task_ids_not_continuous_gpu_proof"},
            "own_unfinished_conversation_observations": self.own_busy_observations(attempted),
            "completed_generations_with_tokens": [completed[i] for i in range(clients)],
            "samples_fully_covered_by_all_generations": len(covered),
            "latency_all_samples_ms": self.latencies(attempted),
            "latency_full_generation_overlap_ms": self.latencies(covered),
            "snapshot_metadata_full_generation_overlap": self.snapshot_stats(covered),
            "availability_all_samples": self.availability(attempted),
            "availability_full_generation_overlap": self.availability(covered),
            "diagnostic_pair_lookup_availability": self.availability([row for _, row in intervals]),
            "concurrent_pair_comparisons": self.comparisons([row for _, row in intervals]),
            "cadence": {"interval_ms": self.config.lookup_interval_ms,
                        "target_hz": 1000 / self.config.lookup_interval_ms if period else None,
                        "requested_slots": self.config.latency_lookups, "scheduled_slots": len(slots),
                        "attempted_slots": len(attempted), "missed_slots": missed,
                        "unlaunched_slots": self.config.latency_lookups - len(slots),
                        "missed_codes": dict(Counter(row["lookup_error_code"] for row in slots
                                                     if not row["lookup_attempted"])),
                        "dispatch_lag_ms": distribution([row["dispatch_lag_ms"] for row in attempted]),
                        "observed_launch_intervals_ms": distribution([
                            (b - a) * 1000 for a, b in zip(starts, starts[1:])])},
            "acceptance_target": {"available_rate": 1, "successful_http_p99_ms_lt": 5,
                                  "observed_target_met": target_met,
                                  "observed_available_rate": available,
                                  "observed_successful_http_p99_ms": p99,
                                  "scope": ("all_samples_and_full_http_overlap" if clients else
                                            "all_samples_and_slots_brackets" if external else
                                            "external_load_unverified")},
            "overlap_is_http_in_flight_not_gpu_execution": True}
        if clients and not all(completed[i] for i in range(clients)):
            self.uncertain("not_all_agents_completed_generation")
        if len(measured) < MIN_SAMPLES:
            self.uncertain("fewer_than_200_load_samples_at_required_overlap")
        if not any("lookup" in row for row in measured):
            self.uncertain("no_engine_lookup_ms_under_measured_load")
        if missed:
            self.uncertain("lookup_launch_cadence_missed")
        if not target_met:
            self.uncertain("load_availability_or_http_p99_target_not_met")
        if not clients and not external_covered:
            self.uncertain("external_generation_load_not_verified")
        if external and (poll_misses or any(error for _, _, _, error in polls)):
            self.uncertain("external_slots_polling_incomplete")

    @staticmethod
    def own_busy_observations(rows):
        targeted = [row for row in rows if "own_generation_case" in row]
        covered = [row for row in targeted if row.get("own_generation_covers_lookup")]
        compared = [row for row in covered if "own_generation_cache_delta_tokens" in row]
        under = sum(row["own_generation_cache_delta_tokens"] < 0 for row in compared)
        return {"targeted_samples": len(targeted), "fully_in_flight_samples": len(covered),
                "compared_samples": len(compared), "uncompared_in_flight_samples": len(covered) - len(compared),
                "under_report_count": under,
                "under_report_rate": under / len(compared) if compared else None,
                "equal_count": sum(row["own_generation_cache_delta_tokens"] == 0 for row in compared),
                "over_report_count": sum(row["own_generation_cache_delta_tokens"] > 0 for row in compared),
                "suppression_cause_verified": False,
                "basis": "same_body_own_in_flight_generation_eventual_accounting_not_tier",
                "busy_live_frontier_suppression_count": None}

    @staticmethod
    def snapshot_stats(rows):
        values = [row["lookup"] for row in rows if "lookup" in row]
        versions = [value["index_version"] for value in values]
        return {"snapshot_age_ms": distribution([value["snapshot_age_ms"] for value in values]),
                "index_version": {"n": len(versions), "min": min(versions) if versions else None,
                                  "max": max(versions) if versions else None,
                                  "distinct": len(set(versions))},
                "version_order_checked": False}

    @staticmethod
    def row_status(row):
        return ("failed" if row.get("failures") else "missed" if row.get("lookup_attempted") is False
                else "busy" if row.get("lookup_busy") else "passed")

    @staticmethod
    def availability(rows):
        available = sum("lookup" in row for row in rows)
        busy = sum(row.get("lookup_busy", False) for row in rows)
        return {"attempts": len(rows), "available_200": available, "busy_503": busy,
                "other_failures": len(rows) - available - busy,
                "available_rate": available / len(rows) if rows else None,
                "busy_rate": busy / len(rows) if rows else None,
                "http_status_counts": dict(Counter(str(row.get("lookup_http_status")) for row in rows)),
                "error_code_counts": dict(Counter(row["lookup_error_code"] for row in rows
                                                  if "lookup_error_code" in row))}

    @staticmethod
    def comparisons(rows):
        compared = [row for row in rows if "cache_comparison" in row]
        over = sum(row["cache_comparison"] == "over_report" for row in compared)
        under = sum(row["cache_comparison"] == "under_report" for row in compared)
        return {"pairs": len(rows), "compared_pairs": len(compared),
                "uncompared_pairs": len(rows) - len(compared),
                "over_report_count": over, "under_report_count": under,
                "over_report_rate": over / len(compared) if compared else None,
                "under_report_rate": under / len(compared) if compared else None,
                "equal_count": len(compared) - over - under,
                "observations_not_reservation_guarantees": True}

    @staticmethod
    def latencies(rows):
        return {"http_wall": distribution([row["lookup_wall_ms"] for row in rows if "lookup_wall_ms" in row]),
                "http_200_wall": distribution([row["lookup_wall_ms"] for row in rows
                                                if "lookup" in row]),
                "http_503_busy_wall": distribution([row["lookup_wall_ms"] for row in rows
                                                     if row.get("lookup_busy")]),
                "engine_lookup": distribution([row["lookup"]["lookup_ms"] for row in rows if "lookup" in row])}

    def finish(self):
        for row in self.rows:
            row.pop("_generation_interval", None)
            row["status"] = self.row_status(row)
        sequential = [r for r in self.rows if r["phase"] == "sequential"]
        diagnostic = [r for r in self.rows if r["phase"] == "load_diagnostic"]
        tiers = Counter(r["lookup"]["tier"] for r in self.rows if "lookup" in r)
        for tier in self.config.required_tiers:
            if not tiers[tier]:
                self.uncertain("required_tier_not_observed_" + tier)
        if self.report.get("scope", {}).get("sequential_requires_exclusive_server") and len(
                [r for r in sequential if "cache_equal" in r]) < MIN_SAMPLES:
            self.uncertain("fewer_than_200_sequential_comparisons")
        self.report["summary"] = {
            "sequential_pairs": len(sequential),
            "sequential_cache_mismatches": sum(r.get("cache_equal") is False for r in sequential),
            "concurrent_pair_comparisons": self.comparisons(diagnostic),
            "tier_counts": dict(tiers), "failure_count": len(self.failures),
            "sequential_latency_ms": self.latencies(sequential),
            "snapshot_metadata_all": self.snapshot_stats(self.rows),
            "snapshot_metadata_load_samples": self.snapshot_stats([
                row for row in self.rows if row["phase"] == "load_lookup"])}
        self.report["status"] = "failed" if self.failures else "unqualified" if self.uncertainties else "passed"
        return self.report

    def run(self, phase="all"):
        try:
            if phase in ("all", "exactness"):
                self.sequential()
            elif phase == "load":
                self.report["scope"]["sequential_requires_exclusive_server"] = False
                self.report["scope"]["sequential_exactness_tested"] = False
            else:
                self.uncertain("sequential_exactness_not_run")
            if phase in ("all", "latency", "load"):
                self.latency()
            else:
                self.uncertain("concurrent_latency_not_run")
        except KeyboardInterrupt:
            self.uncertain("interrupted")
        except Exception:
            # Do not serialize exceptions: servers may echo private prompts in them.
            self.fail({"case": "harness", "phase": "control"}, "harness_exception")
        return self.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path, help="new JSON file, never overwritten")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--phase", choices=("all", "exactness", "latency"), default="all")
    phase.add_argument("--load-only", action="store_const", dest="phase", const="load",
                       help="measure load only; no idle suite or exclusive-server acknowledgement")
    parser.add_argument("--exclusive", action="store_true", help="acknowledge no other server clients during exactness")
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--pressure-lineages", type=int, default=PRESSURE_LINEAGES)
    parser.add_argument("--system-lines", type=int, default=32)
    parser.add_argument("--pressure-lines", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--latency-lookups", type=int, default=400)
    parser.add_argument("--load-max-tokens", type=int, default=64)
    parser.add_argument("--lookup-interval-ms", type=float, default=50,
                        help="fixed launch period, not a post-response sleep (default: 20Hz)")
    parser.add_argument("--clients", type=int, choices=(0, 4, 8), default=4,
                        help="generation sessions; 0 for separately launched external bench load")
    parser.add_argument("--lookup-workers", type=int, default=16,
                        help="maximum simultaneous lookup HTTP requests; missed slots are reported")
    parser.add_argument("--external-clients", type=int, choices=(0, 4, 8), default=0,
                        help="with --clients 0, verify external load via metadata-only /slots polling")
    parser.add_argument("--slots-interval-ms", type=float, default=50,
                        help="fixed /slots poll period; missed polls never burst-replay")
    parser.add_argument("--load-timeout", type=float, default=180,
                        help="launch budget in seconds; in-flight pairs may take 2*request-timeout to drain")
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--pin-prompt-file", type=Path, help="private matching Chat JSON, not rendered prefix text")
    parser.add_argument("--pin-min-matched-tokens", type=int, default=0)
    parser.add_argument("--require-tier", action="append", choices=("live", "ram", "disk", "none"), default=[])
    args = parser.parse_args(argv)
    if args.phase in ("all", "exactness") and not args.exclusive:
        parser.error("exactness requires --exclusive and an allocated server with no other clients")
    if args.external_clients and (args.clients != 0 or args.phase not in ("load", "latency")):
        parser.error("--external-clients requires --clients 0 and --load-only or --phase latency")
    if (any(getattr(args, key) < 1 for key in ("rounds", "max_tokens", "latency_lookups",
                                             "load_max_tokens", "lookup_workers")) or
            any(getattr(args, key) < 0 for key in ("pressure_lineages", "system_lines", "pressure_lines",
                                                 "pin_min_matched_tokens")) or
            not number(args.lookup_interval_ms) or
            any(not number(v) or v == 0 for v in (
                args.request_timeout, args.load_timeout, args.slots_interval_ms))):
        parser.error("counts/deadlines must be finite and positive; line counts/interval/depth may be zero")
    if args.pin_min_matched_tokens and args.pin_prompt_file is None:
        parser.error("--pin-min-matched-tokens requires --pin-prompt-file")
    config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__ if hasattr(args, key)},
                    required_tiers=tuple(args.require_tier))
    try:
        client = Client(args.base_url, args.request_timeout, os.environ.get("OPENAI_API_KEY"))
        pin = None
        if args.pin_prompt_file:
            pin = json.loads(args.pin_prompt_file.read_text())
            if not isinstance(pin, dict) or not isinstance(pin.get("messages"), list) or not pin["messages"]:
                raise ProbeError("pin_file_requires_chat_json")
            pin.update(model=args.model, stream=False, max_completion_tokens=args.max_tokens)
            pin.pop("max_tokens", None)
    except (OSError, ValueError, ProbeError):
        parser.error("invalid URL or pin Chat JSON input (contents deliberately omitted)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the result before any network work; refuse to destroy earlier evidence.
    try:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        parser.error("output already exists; use a new file for every run")
    with os.fdopen(fd, "w") as output:
        harness = Harness(client, config, pin)
        report = harness.run(args.phase)
        report["config"]["request_timeout_s"] = args.request_timeout
        report["config"]["pin_prompt_supplied"] = pin is not None
        report["config"]["phase"] = args.phase
        report["config"]["exclusive_acknowledged"] = args.exclusive
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "load": report.get("load"), "uncertainties": report["uncertainties"]}, allow_nan=False))
    return {"passed": 0, "failed": 1, "unqualified": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
