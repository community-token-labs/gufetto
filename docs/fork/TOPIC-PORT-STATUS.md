# Feature documentation after the offline topic split

The archived nightly feature notes are retained by concern rather than by the
retired nightly/main/stable channel model. The current branch roster, pinned
source identities, CPU checks and integration recipe are in the private
coordination repository's `docs/topics.md` and `docs/topic-split-20261028.md`.
No public/upstream publication or model execution is part of this port.

- [Persistent registered prefixes](PINNED-PREFIX.md)
- [Optional serving trace](PERF-TRACE.md)
- [Trusted-LAN checkpoint transfer](../checkpoint-transfer.md)
- [Prefill correctness](../models/qwen3.8-flash-next/PREFILL.md)
- [PLE format comparison](../models/qwen3.8-flash-next/PLE.md)
- [Shared-Q4 MTP gates](../models/qwen3.8-flash-next/MTP-Q4KM.md)
- [Historical MTP measured setup](../models/qwen3.8-flash-next/MTP-MEASURED-SETUP.md)

The last two comparison documents describe their original dated sources,
outputs, costs and rejection reasons. They do not qualify the newly composed
c92-based topics. The PLE recommendation remains IQ4; the BF16 comparison
profile is retained alongside its original applicability limits. CPU-only
checks cannot establish HIP kernel compilation, NPU lifecycle safety, numerical
model quality, serving parity or timing. All original failed public gates stay
failed; internal-success guidance is not release qualification.

The existing `topic/fork-docs` q8 documentation is unchanged and remains the
canonical cleaned q8 guide. Superseded channel policies and accumulated-product
qualification indexes are available at `archive/nightly-20261028`; they are not
current deployment instructions. The legacy sync utility is separately parked.
