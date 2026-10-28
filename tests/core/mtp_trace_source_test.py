"""CPU-only source gates; these do not execute or qualify HIP dispatch."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src/models/qwen38_flash_next"


class MtpTraceSourceTest(unittest.TestCase):
    def test_scalar_counts_execution_not_capture_or_kv_only(self):
        source = (MODEL / "kernels/rocm/executor.cpp").read_text()
        forward = source.split("bool Executor::MtpForward(", 1)[1].split(
            "bool Executor::MtpBody(", 1
        )[0]
        self.assertLess(forward.index("if (!Run("), forward.index("TraceMtpHead("))
        self.assertIn("if (!output.kv_only &&", forward)
        self.assertIn("(output.token != nullptr || output.candidates != nullptr)", forward)
        # Both scalar callers are instrumented outside MtpHead, never its
        # graph-recordable body. A new caller needs its own execution hook.
        self.assertEqual(source.replace("TraceMtpHead(", "").count("MtpHead("), 2)
        head = source.split("bool Executor::MtpHead(", 1)[1].split(
            "bool Executor::Run(", 1
        )[0]
        self.assertNotIn("TraceMtpHead", head)

    def test_batch_counts_after_projections_and_scratch_decision(self):
        source = (MODEL / "kernels/rocm/batch.cpp").read_text()
        head = source.split("bool Executor::MtpHeads(", 1)[1].split(
            "bool Executor::QuantizeBatch(", 1
        )[0]
        self.assertEqual(head.count("TraceMtpHead("), 2)  # singleton + batch
        self.assertLess(
            head.index("const auto logical_subset_rows = subset_rows;"),
            head.index("subset_rows = 0;", head.index("// The prefix projection")),
        )
        self.assertIn("TraceMtpHead(n, logical_subset_rows, subset_rows", head)
        self.assertLess(
            head.rindex("DenseBatch(full_head"), head.rindex("TraceMtpHead(")
        )
        self.assertIn("Dense(DraftTable(session)", head)
        self.assertIn("if (mixed)", head)
        self.assertIn("GatherDraftLogits(", head)

    def test_cycles_follow_commit_and_keep_anchor_out_of_drafts(self):
        source = (MODEL / "engine.cpp").read_text()
        finish = source.split("bool Session::FinishDecode(", 1)[1].split(
            "bool Session::DecodeStep(", 1
        )[0]
        self.assertEqual(finish.count("TraceMtpCycle("), 2)
        self.assertLess(
            finish.index("result->tokens.push_back(anchor);"),
            finish.index("TraceMtpCycle(base, 0, 0, 0, 1)"),
        )
        self.assertLess(
            finish.index("result->tokens.assign("),
            finish.index("TraceMtpCycle(base, k - 1, keep - 1, pending.lookup, keep)"),
        )
        self.assertIn("mtp_drafts = k - 1 - pending.lookup", finish)
        self.assertIn("first + draft_vocab_.rare().size()", source)
        self.assertIn("kMtpDraftExtraRows", source)


if __name__ == "__main__":
    unittest.main()
