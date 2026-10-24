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
  for (int i = 0; i < 100; ++i) {
    const auto live = pool.Lookup(longer);
    assert(live.matched_tokens == prompt.size() && live.tier == "live");
    assert(live.epoch == cold.epoch);
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
void LookupReturnsBusyWithoutWaitingForAdmission() {
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
  bool busy = false;
  try {
    (void)pool.Lookup(prompt);
  } catch (const TextGenerationError& error) {
    busy = error.code() == TextGenerationErrorCode::kCacheLookupBusy &&
           error.http_status() == 503 && error.retryable();
  }
  runner->released.release();
  // Acquire reinstalls prompt context after restoration too.
  runner->entered.acquire();
  runner->released.release();
  generation.join();
  assert(busy);
}
#endif
