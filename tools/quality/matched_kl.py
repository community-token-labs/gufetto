#!/usr/bin/env python3
"""Offline teacher-token scoring of pinned, full-vocabulary logit captures.

Adapted from jabata's matched-history comparison pattern in PR #697,
c1ab796010ba2a72278e692c6d5e17d363235373 (MIT; see repository LICENSE).
No server, model executor, sampling, sparse top-K imputation or network client.
See README.md beside this file for the capture contract and independence limits.
"""

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

from scoring_metrics import compare_logits, summarize
from scoring_contract import evaluate_strata, validate_contract

HISTORY_SCHEMA = "gufo-matched-history-v1"
CAPTURE_SCHEMA = "gufo-full-logits-v1"
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_INDEX_LINE_BYTES = 64 * 1024
MAX_VOCAB = 1_000_000
DTYPES = {"f32le": 4, "f16le": 2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name}: invalid integer")
    return value


def text(value, name):
    require(isinstance(value, str) and value.strip(), f"{name}: empty/non-string")
    return value


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            f"{name}: expected lowercase SHA-256")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), f"nonfinite JSON number: {value}")
        return number
    result = json.loads(raw, object_pairs_hook=unique_object,
                        parse_constant=invalid_constant, parse_float=finite_float)
    require(isinstance(result, dict), "expected a JSON object")
    return result


def load_history(path, max_positions):
    integer(max_positions, "max_positions", 1)
    with path.open("rb") as source:
        raw = source.read(MAX_HISTORY_BYTES + 1)
    require(len(raw) <= MAX_HISTORY_BYTES, "history exceeds 64 MiB limit")
    data = parse_json(raw)
    require(data.get("schema") == HISTORY_SCHEMA, "unsupported history schema")
    vocab = integer(data.get("vocab_size"), "vocab_size", 1)
    require(vocab <= MAX_VOCAB, "vocabulary exceeds supported budget")
    digest(data.get("tokenizer_sha256"), "tokenizer_sha256")
    require(isinstance(data.get("provenance"), dict) and data["provenance"],
            "history provenance required")
    sequences = data.get("sequences")
    require(isinstance(sequences, list) and sequences, "empty history set")
    expected = []
    seen = set()
    stratum_phases = {}
    for sequence in sequences:
        require(isinstance(sequence, dict), "invalid sequence")
        ident = text(sequence.get("id"), "sequence id")
        require(ident not in seen, f"duplicate sequence: {ident}")
        seen.add(ident)
        stratum = text(sequence.get("stratum"), "stratum")
        prompt_id = text(sequence.get("prompt_id"), "prompt_id")
        phase = text(sequence.get("phase"), "phase")
        conditioning = sequence.get("conditioning")
        require(isinstance(conditioning, dict) and conditioning,
                "explicit conditioning identity required (text or multimodal)")
        conditioning_hash = hashlib.sha256(json.dumps(
            conditioning, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
        require(stratum_phases.setdefault(stratum, phase) == phase,
                "a stratum cannot silently pool different phases")
        tokens, positions = sequence.get("tokens"), sequence.get("positions")
        require(isinstance(tokens, list) and len(tokens) >= 2, "history too short")
        require(isinstance(positions, list) and positions, "no scored positions")
        require(len(expected) + len(positions) <= max_positions,
                "position budget exceeded (no truncation or sampling allowed)")
        for token in tokens:
            require(integer(token, "token") < vocab, "token outside vocabulary")
        previous = 0
        for position in positions:
            integer(position, "position", 1)
            require(previous < position < len(tokens),
                    "positions must be strictly increasing, in [1,len(tokens))")
            previous = position
        # Hash the exact prefix tokens[0:position] as uint32 little-endian.
        # Advance once through the history, including gaps between scored rows.
        prefix = hashlib.sha256()
        cursor = 0
        for position in positions:
            while cursor < position:
                prefix.update(struct.pack("<I", tokens[cursor]))
                cursor += 1
            expected.append({"sequence_id": ident, "position": position,
                             "token": tokens[position], "stratum": stratum,
                             "prompt_id": prompt_id, "phase": phase,
                             "conditioning_sha256": conditioning_hash,
                             "prefix_sha256": prefix.hexdigest()})
    return data, hashlib.sha256(raw).hexdigest(), expected


class Capture:
    """A streamed JSONL index; only one full-vocabulary row is read at a time."""

    def __init__(self, path, history, history_hash, count):
        self.path = path
        self.source = path.open("rb")
        self.hasher = hashlib.sha256()
        try:
            self.header = self.record()
            require(self.header is not None, "empty capture")
            for key, expected in (("schema", CAPTURE_SCHEMA),
                                  ("history_sha256", history_hash),
                                  ("tokenizer_sha256", history["tokenizer_sha256"]),
                                  ("vocab_size", history["vocab_size"]),
                                  ("positions", count)):
                require(self.header.get(key) == expected,
                        f"capture {key} mismatch")
            integer(self.header["vocab_size"], "capture vocab_size", 1)
            integer(self.header["positions"], "capture positions", 1)
            self.dtype = self.header.get("dtype")
            require(isinstance(self.dtype, str) and self.dtype in DTYPES,
                    "only full-vocabulary f32le/f16le captures are supported")
            self.row_bytes = history["vocab_size"] * DTYPES[self.dtype]
            producer = self.header.get("producer")
            require(isinstance(producer, dict), "producer provenance required")
            for key in ("implementation", "revision", "arithmetic", "toolchain"):
                text(producer.get(key), f"producer.{key}")
            digest(producer.get("model_sha256"), "producer.model_sha256")
        except Exception:
            self.close()
            raise

    def close(self):
        self.source.close()

    def record(self):
        line = self.source.readline(MAX_INDEX_LINE_BYTES + 1)
        if not line:
            return None
        require(len(line) <= MAX_INDEX_LINE_BYTES, "capture index line too large")
        self.hasher.update(line)
        return parse_json(line)

    def row(self, expected):
        record = self.record()
        require(record is not None, "missing capture position")
        for key in ("sequence_id", "position", "token", "prefix_sha256", "phase",
                    "conditioning_sha256"):
            require(record.get(key) == expected[key], f"capture {key} mismatch")
        integer(record["position"], "capture position", 1)
        integer(record["token"], "capture token")
        filename = text(record.get("file"), "row file")
        relative = Path(filename)
        require(not relative.is_absolute(), "row file must be relative to capture")
        root = self.path.parent.resolve()
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file(),
                "row file must be a regular file inside the capture directory")
        offset = integer(record.get("offset"), "row offset")
        require(offset % DTYPES[self.dtype] == 0, "unaligned row offset")
        expected_hash = digest(record.get("sha256"), "row sha256")
        with path.open("rb") as source:
            source.seek(offset)
            raw = source.read(self.row_bytes)
        require(len(raw) == self.row_bytes, "truncated full-vocabulary row")
        require(hashlib.sha256(raw).hexdigest() == expected_hash,
                "row SHA-256 mismatch")
        if self.dtype == "f32le":
            values = array.array("f")
            require(values.itemsize == 4, "unsupported host float size")
            values.frombytes(raw)
            if sys.byteorder != "little":
                values.byteswap()
        else:
            values = array.array("d", (v[0] for v in struct.iter_unpack("<e", raw)))
        return values, expected_hash

    def finish(self):
        require(self.record() is None, "extra capture positions")
        return self.hasher.hexdigest()


def write_json(path, data):
    # Serialize first, then publish atomically: an exception/disk-full must not
    # leave a truncated success report containing an apparently passing status.
    serialized = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8") as out:
        out.write(serialized)
    partial.rename(path)


def compare(history_path, reference_path, candidate_path, out_dir, contract_path,
            top_k=20, max_positions=4096):
    history, history_hash, expected = load_history(history_path, max_positions)
    require(type(top_k) is int and 1 <= top_k <= history["vocab_size"],
            "top_k outside vocabulary")
    reference = Capture(reference_path, history, history_hash, len(expected))
    candidate = None
    try:
        candidate = Capture(candidate_path, history, history_hash, len(expected))
        ref_producer = reference.header["producer"]
        cand_producer = candidate.header["producer"]
        require(ref_producer["implementation"] != cand_producer["implementation"],
                "reference must use an independent implementation, not the candidate path")
        text(ref_producer.get("independence"), "reference producer.independence")
        with contract_path.open("rb") as source:
            contract_raw = source.read(MAX_HISTORY_BYTES + 1)
        require(len(contract_raw) <= MAX_HISTORY_BYTES, "contract exceeds 64 MiB limit")
        contract = parse_json(contract_raw)
        require(type(contract.get("top_k")) is int and contract["top_k"] == top_k,
                "contract top_k mismatch")
        validate_contract(contract, reference.header, candidate.header, history_hash)
        counts = {}
        for row in expected:
            group = counts.setdefault(row["stratum"], {"positions": 0, "prompt_ids": set()})
            group["positions"] += 1
            group["prompt_ids"].add(row["prompt_id"])
        counts = {key: {"positions": value["positions"], "prompts": len(value["prompt_ids"])}
                  for key, value in counts.items()}
        # Validate coverage/counts and gate syntax before reading binary rows.
        evaluate_strata(contract, {key: {} for key in counts}, counts)
        # Exclusive output directory: never mix a failed/resumed run with old success.
        out_dir.mkdir(parents=True, exist_ok=False)
        try:
            rows = []
            strata = {}
            byte_exact = 0
            partial = out_dir / "positions.partial.jsonl"
            with partial.open("x", encoding="utf-8") as out:
                for position in expected:
                    ref, ref_hash = reference.row(position)
                    cand, cand_hash = candidate.row(position)
                    metrics = compare_logits(ref, cand, position["token"], top_k)
                    exact = reference.dtype == candidate.dtype and ref_hash == cand_hash
                    byte_exact += int(exact)
                    result = dict(position, **metrics, reference_sha256=ref_hash,
                                  candidate_sha256=cand_hash, logit_bytes_equal=exact,
                                  reference_nll=-metrics["reference_logprob"],
                                  candidate_nll=-metrics["candidate_logprob"])
                    out.write(json.dumps(result, allow_nan=False) + "\n")
                    # Drop vocabulary-sized arrays and per-row top-K lists. Only
                    # bounded scalar metrics/identities remain for exact quantiles.
                    del ref, cand
                    compact = {k: v for k, v in metrics.items()
                               if k not in ("reference_topk", "candidate_topk")}
                    compact.update(position)
                    rows.append(compact)
                    strata.setdefault(position["stratum"], []).append(compact)
            ref_index_hash, cand_index_hash = reference.finish(), candidate.finish()
            worst = max(rows, key=lambda row: row["kl"])
            summaries = {key: dict(summarize(value), **counts[key])
                         for key, value in sorted(strata.items())}
            gates = evaluate_strata(contract, summaries, counts)
            scorer_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("matched_kl.py", "scoring_metrics.py", "scoring_contract.py")}
            report = {
                "schema": "gufo-matched-quality-report-v1",
                "status": gates["status"],
                "qualification": "NOT_QUALIFIED",
                "scope": "offline_logit_gates_only",
                "gates": gates,
                "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
                "scorer_sha256": scorer_hashes,
                "reference_independence": "declared_by_producer_not_proven_by_metrics",
                "history_sha256": history_hash,
                "history_provenance": history["provenance"],
                "tokenizer_sha256": history["tokenizer_sha256"],
                "vocab_size": history["vocab_size"], "top_k": top_k,
                "max_positions": max_positions,
                "reference": dict(reference.header, index_sha256=ref_index_hash,
                                  bytes_per_position=reference.row_bytes),
                "candidate": dict(candidate.header, index_sha256=cand_index_hash,
                                  bytes_per_position=candidate.row_bytes),
                "overall": dict(summarize(rows), prompts=len({r["prompt_id"] for r in rows})),
                "strata": summaries,
                "logit_byte_equal_positions": byte_exact,
                "worst_kl": {key: worst[key] for key in
                             ("sequence_id", "prompt_id", "phase", "position", "stratum", "token", "kl")},
                "worst_kl_by_stratum": {
                    key: {field: max(value, key=lambda r: r["kl"])[field]
                          for field in ("prompt_id", "sequence_id", "position", "phase", "kl")}
                    for key, value in sorted(strata.items())},
                "limits": ["Full-distribution KL in float64, not symbolic exact arithmetic.",
                           "Only explicitly registered logit gates; no model-quality qualification.",
                           "Logit equality does not prove state/transport exactness.",
                           "Capture producer and reference independence require external audit."],
            }
            partial.rename(out_dir / "positions.jsonl")
            write_json(out_dir / "report.json", report)
            return report
        except Exception as error:
            write_json(out_dir / "failure.json", {"status": "INVALID", "error": str(error)})
            raise
    finally:
        reference.close()
        if candidate is not None:
            candidate.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-positions", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        report = compare(args.history, args.reference, args.candidate, args.out_dir,
                         args.contract, args.top_k, args.max_positions)
    except (ValueError, OSError, OverflowError) as error:
        print(f"matched_kl: INVALID: {error}", file=sys.stderr)
        return 2
    print(f"matched_kl: {report['overall']['positions']} positions; "
          f"{report['status']} (offline logit gates only; model NOT_QUALIFIED)")
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
