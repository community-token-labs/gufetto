#!/usr/bin/env python3
"""Compare emitted IDs, never retokenized text. No automatic coherence PASS.

Exit 0: exact fixed-length numeric match (human review still required);
1: demonstrated non-near-tie first divergence; 2: invalid/blocked capability;
3: unknown/incomplete or near-tie requiring human review. See token_parity.md.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

LLAMA_REVISION = "6d5ac3c8d5aae1831edf17c2d5cda11abe025bb3"
SAMPLING = {"temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0,
            "repeat_penalty": 1, "frequency_penalty": 0, "presence_penalty": 0,
            "seed": 0}


def ids(value, label, nonempty=False):
    if (not isinstance(value, list) or (nonempty and not value) or
            any(type(item) is not int or not 0 <= item <= 2147483647 for item in value)):
        raise ValueError(f"{label}: expected {'nonempty ' if nonempty else ''}token-ID array")
    return value


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def reject_constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def validate_config(config, engine):
    for key in ("revision", "flags", "stack", "model_sha256", "tokenizer_sha256", "mode"):
        if key not in config or config[key] in (None, "", []):
            raise ValueError(f"{engine}: missing provenance config {key}")
    for key in ("model_sha256", "tokenizer_sha256"):
        if not isinstance(config[key], str) or not re.fullmatch(r"[0-9a-f]{64}", config[key]):
            raise ValueError(f"{engine}: {key} must be a SHA256 digest")
    if config["mode"] != "ar":
        raise ValueError(f"{engine}: only declared AR mode is supported")
    if not isinstance(config["flags"], list) or not all(isinstance(s, str) for s in config["flags"]):
        raise ValueError(f"{engine}: flags must be an argv array")
    if engine == "llama" and config["revision"] != LLAMA_REVISION:
        raise ValueError("llama: expected pinned revision " + LLAMA_REVISION)


def load_recordings(path, prompts, count, engine):
    document = read_json(path)
    validate_config(document["config"], engine)
    rows = document["requests"]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {p["id"] for p in prompts}:
        raise ValueError(f"{engine}: duplicate/missing/extra request IDs")
    for prompt in prompts:
        row = by_id[prompt["id"]]
        if ids(row["prompt_token_ids"], "recorded prompt", True) != prompt["prompt_token_ids"]:
            raise ValueError(f"{engine}: prompt IDs differ; text substitution forbidden")
        ids(row["generated_token_ids"], "recorded output")
        if row["sampling"] != SAMPLING or row["max_tokens"] != count:
            raise ValueError(f"{engine}: recordings must use registered greedy settings/count")
        if not isinstance(row["text"], str):
            raise ValueError(f"{engine}: continuation text required for human review")
        for position in range(len(row.get("top_logprobs", []))):
            distribution(row, position)
    return document["config"], by_id


def native_live(url, config, prompts, count, top_k, timeout, engine):
    validate_config(config, engine)
    rows = {}
    endpoint = url.rstrip("/").removesuffix("/v1") + "/completion"
    for prompt in prompts:
        payload = dict(SAMPLING, prompt=prompt["prompt_token_ids"], n_predict=count,
                       return_tokens=True, stream=False, cache_prompt=False,
                       ignore_eos=True, n_probs=top_k, post_sampling_probs=False)
        request = Request(endpoint, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        # No retries, text-tokenization endpoints or endpoint fallback.
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response, parse_constant=reject_constant)
        except HTTPError as error:
            raise ValueError(f"BLOCKED: {engine} native token-ID /completion request rejected "
                             f"(HTTP {error.code}); no text fallback") from error
        if "tokens" not in result:
            raise ValueError(f"BLOCKED: {engine} response has no emitted token IDs")
        generated = ids(result["tokens"], engine + " response tokens")
        if result.get("truncated"):
            raise ValueError("llama: prompt/context truncated")
        probabilities = result.get("completion_probabilities", [])
        if probabilities and (len(probabilities) != len(generated) or any(
                entry["id"] != token for entry, token in zip(probabilities, generated))):
            raise ValueError("llama: probability positions do not match emitted IDs")
        rows[prompt["id"]] = dict(
            id=prompt["id"], prompt_token_ids=prompt["prompt_token_ids"],
            generated_token_ids=generated, text=result["content"],
            sampling=SAMPLING, max_tokens=count,
            top_logprobs=[entry.get("top_logprobs", []) for entry in probabilities],
            request=payload, generation_settings=result.get("generation_settings"),
            response=result)
    return config, rows


def llama_live(url, config, prompts, count, top_k, timeout):
    return native_live(url, config, prompts, count, top_k, timeout, "llama")


def gufo_live(url, config, prompts, count, top_k, timeout):
    # One native protocol for both engines; missing distributions stay UNKNOWN.
    # A future full-logit transport can implement this same config + rows contract.
    return native_live(url, config, prompts, count, top_k, timeout, "gufo")


def distribution(row, position):
    entries = row.get("top_logprobs", [])
    if position >= len(entries):
        return None
    result = entries[position]
    if not isinstance(result, list):
        raise ValueError("top_logprobs position must be a list")
    seen = set()
    for entry in result:
        token = entry["id"]
        ids([token], "top-logprob token")
        value = entry["logprob"]
        if type(value) not in (int, float) or not math.isfinite(value) or value > 0:
            raise ValueError("top-logprobs must be finite nonpositive natural log probabilities")
        if token in seen:
            raise ValueError("duplicate top-logprob token ID")
        seen.add(token)
    return sorted(result, key=lambda item: (-item["logprob"], item["id"]))


def near_tie(top, emitted, other, margin):
    if not top or len(top) < 2:
        return {"status": "UNKNOWN", "reason": "per-token top-k unavailable or fewer than two entries"}
    values = {entry["id"]: entry["logprob"] for entry in top}
    gap = top[0]["logprob"] - top[1]["logprob"]
    if emitted not in values or values[emitted] != top[0]["logprob"]:
        return {"status": "UNKNOWN", "reason": "emitted token is not a documented greedy maximum",
                "top1_top2_margin": gap}
    if other not in values:
        return {"status": "NO" if gap > margin else "UNKNOWN",
                "reason": "other engine winner outside captured top-k",
                "top1_top2_margin": gap}
    cross_gap = top[0]["logprob"] - values[other]
    return {"status": "YES" if cross_gap <= margin else "NO",
            "top1_top2_margin": gap, "other_winner_margin": cross_gap}


def compare_prompt(prompt, left, right, count, horizon, margin):
    a, b = left["generated_token_ids"], right["generated_token_ids"]
    common = min(len(a), len(b))
    first = next((i for i in range(common) if a[i] != b[i]),
                 common if len(a) != len(b) else None)
    matches = sum(i < common and a[i] == b[i] for i in range(horizon))
    row = {"id": prompt["id"], "prompt_token_ids": prompt["prompt_token_ids"],
           "lengths": {"gufo": len(a), "llama": len(b)},
           "first_divergence": first, "horizon": horizon,
           "agreement_count": matches, "agreement_rate": matches / horizon,
           "agreement_is_same_index_not_same_history_after_divergence": True,
           "gufo": left, "llama": right, "human_coherence": "NOT_REVIEWED",
           "near_tie": "NOT_APPLICABLE", "status": "NUMERIC_MATCH_REVIEW_REQUIRED"}
    if first is not None:
        row["context"] = {"shared_prefix_tail": (prompt["prompt_token_ids"] + a[:first])[-16:],
                          "gufo_ids": a[max(0, first - 4):first + 5],
                          "llama_ids": b[max(0, first - 4):first + 5]}
        top_a, top_b = distribution(left, first), distribution(right, first)
        row["first_divergence_top_k"] = {"gufo": top_a, "llama": top_b}
        if first < common:
            diagnostics = {"gufo": near_tie(top_a, a[first], b[first], margin),
                           "llama": near_tie(top_b, b[first], a[first], margin)}
            states = {item["status"] for item in diagnostics.values()}
            tie = "NO" if "NO" in states else "YES" if states == {"YES"} else "UNKNOWN"
            row.update(near_tie=tie, near_tie_diagnostics=diagnostics,
                       status={"NO": "NON_NEAR_TIE", "YES": "NEAR_TIE_REVIEW_REQUIRED",
                               "UNKNOWN": "UNKNOWN"}[tie])
        else:
            row.update(near_tie="UNKNOWN", status="INCOMPLETE")
    if len(a) != count or len(b) != count:
        row["fixed_count_complete"] = False
        if row["status"] != "NON_NEAR_TIE":
            row["status"] = "INCOMPLETE"
    else:
        row["fixed_count_complete"] = True
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path)
    for engine in ("gufo", "llama"):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(f"--{engine}-recorded", type=Path)
        group.add_argument(f"--{engine}-url")
        parser.add_argument(f"--{engine}-config", type=Path, help="required with live URL")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--margin", type=float, required=True,
                        help="pre-registered maximum logprob gap (natural logs), not tuned after seeing outputs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args(argv)
    report = {"schema": 1, "registered_margin": args.margin if math.isfinite(args.margin) else None,
              "margin_units": "natural_logprob_gap",
              "tokens": args.tokens, "horizon": args.horizon, "requested_top_k": args.top_k,
              "human_coherence": "NOT_REVIEWED", "automatic_quality_pass": False}
    try:
        if (not 1 <= args.horizon <= args.tokens <= 4096 or not 2 <= args.top_k <= 100 or
                not math.isfinite(args.margin) or args.margin < 0 or
                not math.isfinite(args.timeout) or args.timeout <= 0):
            raise ValueError("invalid count/horizon/top-k/margin/timeout")
        prompts = read_json(args.prompts)
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts must be a nonempty request array")
        names = [p["id"] for p in prompts]
        if not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
            raise ValueError("prompt IDs must be distinct nonempty strings")
        for prompt in prompts:
            ids(prompt["prompt_token_ids"], "prompt", True)
        report["prompts_sha256"] = sha(prompts)
        if args.gufo_url:
            if not args.gufo_config:
                raise ValueError("--gufo-config is required with --gufo-url")
            gufo = gufo_live(args.gufo_url, read_json(args.gufo_config), prompts,
                             args.tokens, args.top_k, args.timeout)
        else:
            gufo = load_recordings(args.gufo_recorded, prompts, args.tokens, "gufo")
        if args.llama_url:
            if not args.llama_config:
                raise ValueError("--llama-config is required with --llama-url")
            llama_config = read_json(args.llama_config)
            validate_config(llama_config, "llama")
            for key in ("model_sha256", "tokenizer_sha256"):
                if gufo[0][key] != llama_config[key]:
                    raise ValueError(f"mismatched {key}; refusing live request")
            llama = llama_live(args.llama_url, llama_config, prompts,
                               args.tokens, args.top_k, args.timeout)
        else:
            llama = load_recordings(args.llama_recorded, prompts, args.tokens, "llama")
        report["configs"] = {"gufo": gufo[0], "llama": llama[0]}
        for key in ("model_sha256", "tokenizer_sha256"):
            if gufo[0][key] != llama[0][key]:
                raise ValueError(f"mismatched {key}; this is not a matched-weight/token-ID comparison")
        report["requests"] = [compare_prompt(p, gufo[1][p["id"]], llama[1][p["id"]],
                                             args.tokens, args.horizon, args.margin) for p in prompts]
        states = {row["status"] for row in report["requests"]}
        code = 1 if "NON_NEAR_TIE" in states else 0 if states == {"NUMERIC_MATCH_REVIEW_REQUIRED"} else 3
        report["status"] = {0: "NUMERIC_MATCH_REVIEW_REQUIRED", 1: "NON_NEAR_TIE",
                            3: "INCONCLUSIVE_OR_HUMAN_REVIEW_REQUIRED"}[code]
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.update(status="BLOCKED_OR_INVALID", error=str(error))
        code = 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
