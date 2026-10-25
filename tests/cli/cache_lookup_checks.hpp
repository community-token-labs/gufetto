// Included after the component cache test's CPU Runner and request helpers.
#ifndef GUFO_TEST_CACHE_LOOKUP_CHECKS_HPP_
#define GUFO_TEST_CACHE_LOOKUP_CHECKS_HPP_

void LookupMeasuredRatesAndEpoch() {
  CacheLookupTimings timings;
  TextGenerationBackend::CacheLookupResult unknown;
  unknown.total_tokens = 100;
  unknown.matched_tokens = 40;
  unknown.tier = "ram";
  timings.Estimate(unknown, 400);
  assert(!unknown.restore_estimate_ms && !unknown.prefill_estimate_ms);
  timings.Prefill(100, 10);
  timings.Restore("ram", 100, 1000, 20);
  timings.Restore("disk", 100, 1000, 200);
  timings.Estimate(unknown, 400);
  assert(*unknown.prefill_estimate_ms == 6);
  assert(*unknown.restore_estimate_ms == 8);
  auto disk = unknown;
  disk.restore_estimate_ms.reset();
  disk.tier = "disk";
  timings.Estimate(disk, 400);
  assert(*disk.restore_estimate_ms == 80);
  timings.Restore("ram", 200, 4000, 40);
  auto ram_ewma = unknown;
  ram_ewma.restore_estimate_ms.reset();
  timings.Estimate(ram_ewma, 400);
  assert(*ram_ewma.restore_estimate_ms == 6.4);
  disk.restore_estimate_ms.reset();
  timings.Estimate(disk, 400);
  assert(*disk.restore_estimate_ms == 80);  // independent disk EWMA unchanged
  auto tokens_only = unknown;
  tokens_only.restore_estimate_ms.reset();
  timings.Estimate(tokens_only);
  assert(*tokens_only.restore_estimate_ms == 8);
  timings.Prefill(200, 10);  // EWMA 0.75*10 + 0.25*20 = 12.5 token/ms
  auto interpolated = unknown;
  interpolated.prefill_estimate_ms.reset();
  timings.Estimate(interpolated, 400);
  assert(*interpolated.prefill_estimate_ms == 4.8);
  auto exact = unknown;
  exact.total_tokens = 300;
  exact.matched_tokens = 100;
  exact.restore_estimate_ms.reset();
  exact.prefill_estimate_ms.reset();
  timings.Estimate(exact, 1000);
  assert(*exact.restore_estimate_ms == 20 && *exact.prefill_estimate_ms == 10);
  auto no_work = unknown;
  no_work.total_tokens = no_work.matched_tokens;
  no_work.tier = "live";
  no_work.restore_estimate_ms.reset();
  no_work.prefill_estimate_ms.reset();
  timings.Estimate(no_work);
  assert(*no_work.restore_estimate_ms == 0 &&
         *no_work.prefill_estimate_ms == 0);
  auto none = unknown;
  none.total_tokens = 0;
  none.matched_tokens = 0;
  none.tier = "none";
  timings.Estimate(none);
  assert(*none.restore_estimate_ms == 0 && *none.prefill_estimate_ms == 0);

  ReasoningOptions reasoning;
  const auto epoch =
      CacheLookupConfiguredEpoch("base", reasoning, 512, "a", "rev");
  assert(epoch.size() == 64);
  assert(epoch ==
         CacheLookupConfiguredEpoch("base", reasoning, 512, "a", "rev"));
  reasoning.preserve_thinking = true;
  assert(epoch !=
         CacheLookupConfiguredEpoch("base", reasoning, 512, "a", "rev"));
  reasoning = {};
  assert(epoch !=
         CacheLookupConfiguredEpoch("base", reasoning, 256, "a", "rev"));
  assert(epoch !=
         CacheLookupConfiguredEpoch("base", reasoning, 512, "b", "rev"));
  assert(epoch !=
         CacheLookupConfiguredEpoch("base", reasoning, 512, "a", "new"));
  TextRunnerDescriptor descriptor{
      .model_id = "model",
      .state_abi = "abi",
      .max_context = 4096,
      .persistence = TextRunnerPersistenceDescriptor{{1, 2, 3}, 20}};
  const auto base = CacheLookupEpoch(descriptor, {}, 1, {}, {});
  descriptor.persistence->payload_version = 21;
  assert(base != CacheLookupEpoch(descriptor, {}, 1, {}, {}));
  descriptor.persistence->payload_version = 20;
  descriptor.persistence->compatibility_identity.push_back(4);
  assert(base != CacheLookupEpoch(descriptor, {}, 1, {}, {}));
  descriptor.persistence->compatibility_identity.pop_back();
  assert(base != CacheLookupEpoch(descriptor, {}, 2, {}, {}));
  assert(base != CacheLookupEpoch(descriptor, {}, 1, {}, {}, "projector"));
}
void LookupIsReadOnlyAndMatchesAcquire() {
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 1);
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17};
  const auto resets = runner->adapter->Invalidations();
  const auto cold = pool.Lookup(prompt);
  assert(cold.total_tokens == prompt.size() && !cold.matched_tokens);
  assert(cold.tier == "none" && cold.epoch.size() == 64);
  assert(cold.lookup_ms >= 0 && !cold.prefill_estimate_ms);
  assert(runner->adapter->Invalidations() == resets && !runner->captures);
  auto first = pool.Acquire(prompt);
  Prefill(first);
  (void)first.Commit();
  auto longer = prompt;
  longer.push_back(99);
  const auto captured = runner->captures;
  const auto invalidations = runner->adapter->Invalidations();
  double sum_ms = 0;
  const auto published = pool.Lookup(longer);
  assert(published.index_version > cold.index_version);
  for (int i = 0; i < 100; ++i) {
    const auto live = pool.Lookup(longer);
    assert(live.matched_tokens == prompt.size() && live.tier == "live");
    assert(live.epoch == cold.epoch);
    assert(live.index_version == published.index_version);
    assert(live.snapshot_age_ms >= published.snapshot_age_ms);
    sum_ms += live.lookup_ms;
  }
  std::cout << "CPU metadata lookup mean_ms=" << sum_ms / 100 << '\n';
  assert(runner->captures == captured);
  assert(runner->adapter->Invalidations() == invalidations);
  const auto disabled = pool.Lookup(longer, nullptr, false);
  assert(disabled.matched_tokens == 0 && disabled.tier == "none");
  auto actual = pool.Acquire(longer);
  assert(actual.cached_prompt_tokens() == prompt.size());
  actual.Invalidate();  // removes live, retains the immutable prompt
  const auto ram = pool.Lookup(longer);
  assert(ram.matched_tokens == prompt.size() && ram.tier == "ram");
  auto restored = pool.Acquire(longer);
  assert(restored.cached_prompt_tokens() == ram.matched_tokens);
  assert(restored.cache_restore_bytes() > 0);
  restored.Invalidate();
  assert(pool.Lookup(longer, nullptr, true, 4).matched_tokens == 0);
  TextPromptContext image;
  image.cache_identity = {1, 2, 3};
  assert(pool.Lookup(longer, &image).matched_tokens == 0);
  image.cache_prefixes = {{prompt.size(), {}}};
  assert(pool.Lookup(longer, &image).matched_tokens == prompt.size());
}
void LookupDoesNotWaitForAdmission() {
  class BlockingRunner final : public Runner {
  public:
    mutable std::binary_semaphore entered{0}, released{0};
    void SetPromptContext(
        TextRunnerState&,
        std::shared_ptr<const TextPromptContext>) const override {
      entered.release();
      released.acquire();
    }
  };
  auto runner = std::make_shared<BlockingRunner>();
  TextRunnerPool pool(runner, 1);
  const std::vector<TextRunnerToken> prompt{10, 11};
  std::thread generation([&] { auto request = pool.Acquire(prompt); });
  runner->entered.acquire();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) {
    const auto observed = pool.Lookup(prompt);
    assert(observed.total_tokens == prompt.size());
    assert(observed.matched_tokens == 0 && observed.tier == "none");
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  runner->released.release();
  // Acquire reinstalls prompt context after restoration too.
  runner->entered.acquire();
  runner->released.release();
  generation.join();
  assert(elapsed < std::chrono::milliseconds(100));
}

void LookupWhileAllSlotsPrefillAndDecode() {
  class BlockingWorkRunner final : public Runner {
  public:
    mutable std::binary_semaphore entered{0}, released{0};
    std::atomic<bool> block{false};
    void Wait() const {
      if (block.load()) {
        entered.release();
        released.acquire();
      }
    }
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      Wait();
      return Runner::Prefill(state, prompt, offset, budget);
    }
    void Advance(TextRunnerState& state, TextRunnerToken token) const override {
      Wait();
      Runner::Advance(state, token);
    }
  };
  auto runner = std::make_shared<BlockingWorkRunner>();
  TextRunnerPool pool(runner, 1);
  const std::vector<TextRunnerToken> root{10, 11, 12, 13, 14, 15, 16, 17};
  auto warm = pool.Acquire(root);
  Prefill(warm);
  (void)warm.Commit();
  auto prompt = root;
  prompt.push_back(99);
  auto request = pool.Acquire(prompt);
  assert(request.cached_prompt_tokens() == root.size());
  runner->block = true;
  std::thread prefill([&] { Prefill(request); });
  runner->entered.acquire();
  for (int i = 0; i < 100; ++i) {
    const auto observed = pool.Lookup(prompt);
    assert(observed.matched_tokens == root.size() && observed.tier == "ram");
  }
  runner->released.release();
  prefill.join();
  auto longer = prompt;
  longer.push_back(100);
  std::thread decode([&] {
    (void)request.SelectNext();
    request.Advance();
  });
  runner->entered.acquire();
  for (int i = 0; i < 100; ++i) {
    const auto observed = pool.Lookup(longer);
    assert(observed.matched_tokens <= prompt.size());
    assert(observed.matched_tokens >= root.size());
  }
  runner->released.release();
  decode.join();
  runner->block = false;
  (void)request.Commit();
  const auto idle = pool.Lookup(longer);
  auto actual = pool.Acquire(longer);
  assert(idle.matched_tokens == actual.cached_prompt_tokens());
}

void LookupInvalidatesBeforeLiveRetirement() {
  class BlockingPublicationRunner final : public Runner {
  public:
    mutable std::binary_semaphore entered{0}, released{0};
    mutable std::atomic<bool> block_next{false};
    bool LookupFullCheckpointReady() const override {
      if (block_next.exchange(false)) {
        entered.release();
        released.acquire();
      }
      return true;
    }
  };
  auto runner = std::make_shared<BlockingPublicationRunner>();
  TextRunnerPool pool(runner, 1);
  const std::vector<TextRunnerToken> root{10, 11, 12, 13, 14, 15, 16, 17};
  auto warm = pool.Acquire(root);
  Prefill(warm);
  const auto next = warm.SelectNext();
  warm.Advance();
  (void)warm.Commit();
  auto target = root;
  target.push_back(next.token);
  target.push_back(99);
  const auto before = pool.Lookup(target);
  assert(before.matched_tokens == root.size() + 1 && before.tier == "live");
  runner->block_next = true;
  std::thread mutation([&] {
    auto other = pool.Acquire({1000, 1001, 1002});
    other.Invalidate();
  });
  // Publication is paused with the owner mutex held, after retiring the live
  // frontier. A reader must not wait or reuse the previous published frontier.
  runner->entered.acquire();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) {
    const auto observed = pool.Lookup(target);
    assert(observed.matched_tokens == 0 && observed.tier == "none");
    assert(observed.index_version > before.index_version);
  }
  assert(std::chrono::steady_clock::now() - start <
         std::chrono::milliseconds(100));
  runner->released.release();
  mutation.join();
  const auto after = pool.Lookup(target);
  assert(after.matched_tokens == root.size() && after.tier == "ram");
  auto actual = pool.Acquire(target);
  assert(after.matched_tokens == actual.cached_prompt_tokens());
}

void LookupPublicationDisabled() {
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 1, {}, {.publish_lookup = false});
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17};
  auto request = pool.Acquire(prompt);
  Prefill(request);
  (void)request.Commit();
  const auto disabled = pool.Lookup(prompt);
  assert(disabled.matched_tokens == 0 && disabled.index_version == 0);
  auto actual = pool.Acquire(prompt);
  assert(actual.cached_prompt_tokens() == prompt.size());
}

void LookupPublicationStress() {
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 2);
  const std::vector<TextRunnerToken> root{10, 11, 12, 13, 14, 15, 16, 17};
  auto warm = pool.Acquire(root);
  Prefill(warm);
  (void)warm.Commit();
  auto prompt = root;
  prompt.push_back(99);
  std::atomic<bool> done{false};
  std::atomic<std::size_t> reads{0}, under_reports{0}, ready_reads{0};
  std::barrier ready(9);
  std::vector<std::thread> readers;
  for (int i = 0; i < 8; ++i)
    readers.emplace_back([&] {
      ready.arrive_and_wait();
      bool first = true;
      while (!done.load()) {
        const auto observed = pool.Lookup(prompt);
        assert(observed.total_tokens == prompt.size());
        assert(observed.matched_tokens <= prompt.size());
        if (observed.matched_tokens < root.size())
          ++under_reports;
        ++reads;
        if (first) {
          ++ready_reads;
          first = false;
        }
      }
    });
  ready.arrive_and_wait();
  while (ready_reads.load() != 8)
    std::this_thread::yield();
  const auto before_updates = reads.load();
  for (int i = 0; i < 200; ++i) {
    auto request = pool.Acquire(prompt);
    Prefill(request);
    if (i % 3)
      (void)request.Commit();
    else
      request.Invalidate();
    if (i == 0)
      while (reads.load() == before_updates)
        std::this_thread::yield();
  }
  done = true;
  for (auto& reader : readers)
    reader.join();
  assert(reads > 0);
  std::cout << "CPU concurrent lookup reads=" << reads
            << " conservative_misses=" << under_reports << '\n';
  const auto idle = pool.Lookup(prompt);
  auto actual = pool.Acquire(prompt);
  assert(idle.matched_tokens == actual.cached_prompt_tokens());
}
#endif
