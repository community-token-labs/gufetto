"""CPU-only, full-vocabulary metrics for pinned teacher-token histories.

Donor pattern: jabata #697, commit c1ab796010ba2a72278e692c6d5e17d363235373
(MIT). This independent implementation imports no served-inference source.

All arithmetic is Python float (IEEE binary64), with natural logarithms and
math.fsum reductions. KL is reference || candidate over the entire vocabulary,
not a top-k approximation; there is no epsilon or missing-tail imputation.
Ordinary binary64 exponential underflow is possible. Only negative KL within
32 * ulp(1) * max(1, sum(abs(KL terms))) is clamped to zero as roundoff;
more negative results fail. This is not a scientific acceptance tolerance.
Finite inputs whose intermediate/results cannot be represented finitely raise
ValueError, rather than emitting nonstandard JSON NaN/Infinity.

Ranks are one-based; token IDs are zero-based. Equal logits favor the lower
ID, including at top-k boundaries. Each comparison uses O(V) memory and
O(V log(k + 1)) time, and retains no caller-owned rows or global state.
"""

from collections.abc import Sequence
import heapq
import math
from numbers import Real


_FLOAT_METRICS = (
    "kl", "reference_logprob", "candidate_logprob", "nll_delta",
    "topk_overlap", "max_abs_logit_delta",
)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as a finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _logsoftmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    shifted = [_finite(value - maximum, "shifted logit") for value in logits]
    log_normalizer = math.log(math.fsum(math.exp(value) for value in shifted))
    # Do not add maximum back: that loses the normalizer at large offsets.
    return [_finite(value - log_normalizer, "log probability") for value in shifted]


def _index(value: int, name: str, lower: int, upper: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def compare_logits(
    reference: Sequence[float], candidate: Sequence[float], token: int, top_k: int
) -> dict:
    """Compare two complete logit vectors at one pinned teacher-token position.

    Reject empty/mismatched vocabularies, invalid token/top_k, nonfinite logits,
    and unrepresentable arithmetic with ValueError. max_abs_logit_delta uses
    raw (not centered) logits, so unlike distribution metrics it is not shift
    invariant. topk_overlap is intersection size / top_k.
    """
    size = len(reference)
    if size == 0 or len(candidate) != size:
        raise ValueError("reference and candidate must have the same nonempty vocabulary")
    _index(token, "token", 0, size - 1)
    _index(top_k, "top_k", 1, size)
    ref = [_finite(value, "reference logit") for value in reference]
    cand = [_finite(value, "candidate logit") for value in candidate]
    ref_logp = _logsoftmax(ref)
    cand_logp = _logsoftmax(cand)
    terms = [
        math.exp(r) * _finite(r - c, "log probability difference")
        for r, c in zip(ref_logp, cand_logp)
    ]
    try:
        kl = _finite(math.fsum(terms), "KL")
        roundoff_bound = 32 * math.ulp(1.0) * max(1.0, math.fsum(abs(t) for t in terms))
    except OverflowError as exc:
        raise ValueError("KL reduction overflow") from exc
    if kl < 0:
        if kl < -roundoff_bound:
            raise ValueError("negative KL exceeds binary64 roundoff bound")
        kl = 0.0

    def topk(logits: list[float]) -> list[int]:
        return heapq.nsmallest(top_k, range(size), key=lambda i: (-logits[i], i))

    def rank(logits: list[float]) -> int:
        return 1 + sum(
            value > logits[token] or (value == logits[token] and i < token)
            for i, value in enumerate(logits)
        )

    ref_topk = topk(ref)
    cand_topk = topk(cand)
    return {
        "kl": kl,
        "reference_logprob": ref_logp[token],
        "candidate_logprob": cand_logp[token],
        "nll_delta": _finite(ref_logp[token] - cand_logp[token], "NLL delta"),
        "reference_rank": rank(ref),
        "candidate_rank": rank(cand),
        "reference_top1": ref_topk[0],
        "reference_top1_margin": (_finite(heapq.nlargest(2, ref)[0] - heapq.nlargest(2, ref)[1],
                                          "reference top1 margin") if size > 1 else None),
        "candidate_top1": cand_topk[0],
        "top1_agreement": ref_topk[0] == cand_topk[0],
        "topk_overlap": len(set(ref_topk).intersection(cand_topk)) / top_k,
        "reference_topk": ref_topk,
        "candidate_topk": cand_topk,
        "max_abs_logit_delta": max(
            _finite(abs(r - c), "absolute logit delta") for r, c in zip(ref, cand)
        ),
    }


def summarize(rows: list[dict]) -> dict:
    """Summarize nonempty comparison rows, each position weighted equally.

    KL quantiles use sorted values and linear interpolation at (n - 1) * q
    (including the endpoints). NLL fields are mean teacher-token NLLs; delta
    is mean candidate NLL minus reference NLL. Perplexity ratio is exp(mean
    delta), with null and perplexity_ratio_overflow=True on overflow (ordinary
    underflow returns zero). No pass/fail gate is selected. The caller owns
    strata, provenance, and annotation of the worst-KL position.
    """
    if not rows:
        raise ValueError("cannot summarize zero positions")
    for row in rows:
        for key in _FLOAT_METRICS:
            _finite(row[key], key)
        if not isinstance(row["top1_agreement"], bool):
            raise ValueError("top1_agreement must be bool")

    count = len(rows)

    def mean(key: str, sign: float = 1.0) -> float:
        try:
            total = math.fsum(float(row[key]) for row in rows)
        except OverflowError:
            # Scale only when necessary, preserving subnormal means otherwise.
            try:
                return _finite(sign * math.fsum(float(row[key]) / count for row in rows), key)
            except OverflowError as exc:
                raise ValueError(f"{key} mean overflow") from exc
        return _finite(sign * (total / count), key)

    ordered_kl = sorted(float(row["kl"]) for row in rows)

    def quantile(q: float) -> float:
        index = (count - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        fraction = index - lower
        low, high = ordered_kl[lower], ordered_kl[upper]
        difference = high - low
        if math.isfinite(difference):
            return _finite(low + fraction * difference, "KL quantile")
        return _finite(math.fsum(((1.0 - fraction) * low, fraction * high)), "KL quantile")

    delta = mean("nll_delta")
    try:
        ratio = math.exp(delta)
    except OverflowError:
        ratio = None
    return {
        "positions": count,
        "kl_mean": mean("kl"),
        "kl_p50": quantile(0.50),
        "kl_p95": quantile(0.95),
        "kl_p99": quantile(0.99),
        "kl_max": ordered_kl[-1],
        "top1_agreement": sum(row["top1_agreement"] for row in rows) / count,
        "topk_overlap": mean("topk_overlap"),
        "reference_nll": mean("reference_logprob", -1.0),
        "candidate_nll": mean("candidate_logprob", -1.0),
        "nll_delta": delta,
        "perplexity_ratio": ratio,
        "perplexity_ratio_overflow": ratio is None,
    }
