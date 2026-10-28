#!/usr/bin/env python3
"""Model-free SOURCE invariants, not HIP/XRT execution or TSan coverage.

Run directly with Python; no build, packages, hardware or model required.
Optional --revision REF checks the same contracts against a Git source revision.
These deliberately narrow checks guard synchronization/ownership shape; they do
not prove DMA completion, callback ordering, numerical correctness or RAM usage.
"""

import argparse
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
CPP = "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.cpp"
HPP = "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"


def body(source, signature):
    """Extract one balanced function body (these bodies have no brace strings)."""
    start = source.index("{", source.index(signature))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1:end - 1]


class RuntimeSourceInvariants(unittest.TestCase):
    def test_shared_statistics_are_atomic(self):
        for name in ("submit_us", "copy_a_us", "launch_us", "run_wait_us",
                     "copy_c_us", "wait_us", "worker_trace_us",
                     "worker_trace_calls", "processes", "waits"):
            self.assertRegex(SOURCE, rf"std::atomic<[^>]+>\s+{name}\{{\s*0\s*\}}")
            self.assertNotRegex(SOURCE, rf"\b{name}\s*\+=")

    def test_ticket_wait_does_not_take_engine_lock(self):
        wait = body(SOURCE, "void WaitNow(")
        self.assertIn("completed >= ticket", wait)
        self.assertIn("lock(queue_mutex)", wait)
        self.assertNotRegex(wait, r"lock\(mutex\)|api_mutex")
        self.assertIn("waits.fetch_add(1, std::memory_order_relaxed)", wait)
        self.assertIn("NOT ", wait)
        self.assertIn("model layers", wait)
        self.assertNotIn(" / waits", wait)
        self.assertIn("processcount", wait)
        self.assertIn("worker_trace_us", wait)
        self.assertIn("NOT device compute time", wait)

    def test_trace_off_returns_before_clock_or_write(self):
        trace = body(SOURCE, "void Trace(")
        guard = trace.index("if (fd < 0)")
        returned = trace.index("return;", guard)
        clock = trace.index("std::chrono::steady_clock::now()")
        write = trace.index("write(fd,")
        self.assertLess(returned, clock)
        self.assertLess(clock, write)
        self.assertLess(write, trace.index("++trace_write_calls"))
        self.assertIn("O_APPEND | O_DSYNC", trace)
        self.assertIn("thread_local double trace_write_us", SOURCE)
        self.assertIn("thread_local std::uint64_t trace_write_calls", SOURCE)

    def test_worker_publishes_trace_delta_before_ticket_completion(self):
        worker = body(SOURCE, "void WorkerLoop()")
        before = worker.index("const double trace_before = trace_write_us")
        process = worker.index("Process(call)")
        publish = worker.index("worker_trace_us.fetch_add(trace_write_us - trace_before")
        count = worker.index("processes.fetch_add(1")
        complete = worker.index("++completed")
        self.assertLess(before, process)
        self.assertLess(process, publish)
        self.assertLess(publish, count)
        self.assertLess(count, complete)
        self.assertIn("trace_write_calls - calls_before", worker)
        self.assertIn("submit_us.fetch_add(Since(t0)", worker)

    def test_release_unregisters_before_fallible_sync(self):
        release = body(SOURCE, "bool NpuGemm::ReleaseStream(")
        unregister = release.index("std::erase(impl_->streams, stream)")
        sync = release.index("hipStreamSynchronize(stream)")
        poison = release.index("process_failed = true", sync)
        failure_return = release.index("return false", sync)
        self.assertLess(unregister, sync)
        self.assertLess(poison, failure_return)
        self.assertEqual(release.count("hipStreamSynchronize("), 1)
        self.assertNotIn("hipStreamDestroy", release)
        self.assertNotIn("hipHostFree", release)
        self.assertNotIn("streams.push_back", release)
        self.assertIn("return Healthy();", release[:sync])

    def test_failed_resources_are_quarantined_not_freed(self):
        destructor = body(SOURCE, "NpuGemm::~NpuGemm()")
        failed = destructor[destructor.index("if (impl_->failed)"):]
        self.assertIn("process_failed = true", failed)
        self.assertIn("impl_.release()", failed)
        self.assertNotIn("hipHostFree", failed)
        self.assertNotIn("hipStreamDestroy", destructor)

    def test_memory_ledger_counts_import_as_alias_not_extra_backing(self):
        ledger = body(SOURCE, "NpuGemm::MemoryFootprint NpuGemm::MemoryFootprintBytes()")
        self.assertIn("lock(impl_->mutex)", ledger)
        self.assertEqual(ledger.count("pinned_host_bytes += buffer.bytes"), 1)
        self.assertIn("logical_buffer_bytes += buffer.bytes", ledger)
        self.assertRegex(ledger, r"if \(impl_->isolate\)\s*\{\s*"
                         r"result.private_bo_bytes \+= buffer.bytes;\s*"
                         r"\} else \{\s*result.imported_alias_bytes \+= buffer.bytes;")
        total = re.sub(r"\s+", " ", ledger[ledger.index("result.total_backing_bytes ="):])
        self.assertIn("result.pinned_host_bytes + result.private_bo_bytes", total)
        self.assertIn("result.instruction_bo_bytes", total)
        self.assertNotIn("imported_alias_bytes", total)
        self.assertIn("impl_->extra_programs", ledger)
        self.assertIn("NOT empirically observed physical RAM", HEADER)

    def test_npu_qkv_does_not_select_decode_batch_index_keys(self):
        executor = read_source("src/models/qwen38_flash_next/kernels/rocm/executor.cpp")
        declaration = read_source("src/models/qwen38_flash_next/kernels/rocm/executor.hpp")
        attention = body(executor, "bool Executor::Attention(")
        self.assertIn("projections_ready && !qkv_only_precomputed", attention)
        self.assertIn("DenseBatch(l.indexer_k", attention)
        self.assertIn("Dense(l.indexer_k", attention)
        self.assertIn("bool qkv_only_precomputed = false", declaration)

    def test_transport_flags_and_private_default_unchanged(self):
        self.assertIn('value == nullptr || std::strcmp(value, "0") != 0', SOURCE)
        new_buffer = body(SOURCE, "int NpuGemm::NewBuffer(")
        self.assertIn("hipHostMallocMapped | (i.isolate ? hipHostMallocNonCoherent : 0U)",
                      new_buffer)
        self.assertNotIn("hipHostMallocCoherent", new_buffer)
        self.assertEqual(new_buffer.count("hipHostMalloc(&host, bytes, flags)"), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="read source from this local Git ref")
    args, unittest_args = parser.parse_known_args()

    def read_source(path):
        if args.revision:
            return subprocess.check_output(
                ["git", "show", f"{args.revision}:{path}"], cwd=ROOT, text=True)
        return (ROOT / path).read_text()

    SOURCE = read_source(CPP)
    HEADER = read_source(HPP)
    unittest.main(argv=[__file__, *unittest_args])
