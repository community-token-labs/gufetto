// Included inside the component fake-runner test's anonymous namespace.
// All barriers are semaphore handshakes, never sleeps or probabilistic races.
void LookupR3BarrierFixtures() {
  const auto old_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  for (const std::string scenario :
       {"control", "retirement", "restore_failure", "in_flight_mutation",
        "completed_mutation_overlap"}) {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 2, 1 << 22, {});
    std::vector<TextRunnerToken> prompt(620);
    std::iota(prompt.begin(), prompt.end(), 10);
    const std::vector<TextRunnerToken> root(prompt.begin(),
                                            prompt.begin() + 531);
    auto source = cache.Acquire(root, {}, {}, false, 0, true);
    auto& state = dynamic_cast<State&>(source->state());
    runner->adapter->Append(state.slot, root, root);
    assert(source->TryReserveSnapshot(source->CaptureBytes(), root.size(),
                                      false, SnapshotPurpose::kHistory, {}));
    assert(source->PublishSnapshot(root, source->Capture(), false));
    // Finish preservation before arming failure injection. Otherwise an idle
    // spill could consume FailNextTransfer instead of the selected restore.
    runner->adapter->GuardRows(state.slot, cache::testing::kTarget, 0, 531);
    runner->adapter->GuardRows(state.slot, cache::testing::kDraft, 0, 531);
    // No live frontier: the only advertised match is the 531-token RAM entry.
    source->Invalidate();
    Logger::Info("test",
                 "schema=cache-lookup-fixture-v1 event=begin case=" + scenario);
    std::binary_semaphore entered{0}, released{0};
    std::atomic<bool> armed{true};
    std::unique_ptr<ComponentTextCache::Lease> actual;
    std::optional<ContinuationReuse> lookup;
    const auto pause = [&](std::string_view wanted) {
      runner->barrier = [&, wanted](std::string_view point) {
        if (point == wanted && armed.exchange(false)) {
          entered.release();
          released.acquire();
        }
      };
    };
    if (scenario == "in_flight_mutation") {
      pause("mutation");
      std::thread writer([&] {
        auto other = cache.Acquire({&prompt.back(), 1}, {}, {}, false, 0, true);
        other->Invalidate();
      });
      entered.acquire();
      lookup = cache.Lookup(prompt, nullptr, true, 0);
      assert(lookup->matched_tokens == 0 && (lookup->index_version & 1));
      released.release();
      writer.join();
      actual = cache.Acquire(prompt, {}, {}, true, 0, true);
      assert(actual->cached_tokens() == 531);
    } else if (scenario == "completed_mutation_overlap") {
      pause("lookup_observed");
      std::thread reader(
          [&] { lookup = cache.Lookup(prompt, nullptr, true, 0); });
      entered.acquire();
      auto other = cache.Acquire({&prompt.back(), 1}, {}, {}, false, 0, true);
      other->Invalidate();
      released.release();
      reader.join();
      assert(lookup->matched_tokens == 0 && !(lookup->index_version & 1));
      actual = cache.Acquire(prompt, {}, {}, true, 0, true);
      assert(actual->cached_tokens() == 531);
    } else {
      lookup = cache.Lookup(prompt, nullptr, true, 0);
      assert(lookup->matched_tokens == 531 && lookup->tier == "ram");
      pause(scenario == "restore_failure" ? "selected" : "admission");
      std::thread admission(
          [&] { actual = cache.Acquire(prompt, {}, {}, true, 0, true); });
      entered.acquire();
      if (scenario == "retirement") {
        // Real bounded retention, not a mock erase: higher-priority unrelated
        // continuations retire the older history checkpoint at record pressure.
        for (std::size_t i = 0; i < cache::RetentionPolicy::kRecordLimit + 1;
             ++i) {
          std::vector<TextRunnerToken> pressure(8, 10000 + i);
          auto peer = cache.Acquire(pressure, {}, {}, false, 0, true);
          auto& s = dynamic_cast<State&>(peer->state());
          runner->adapter->Append(s.slot, pressure, pressure);
          assert(peer->TryReserveSnapshot(peer->CaptureBytes(), pressure.size(),
                                          false, SnapshotPurpose::kContinuation,
                                          {}));
          assert(peer->PublishSnapshot(pressure, peer->Capture(), false));
          peer->Invalidate();
        }
        assert(cache.Lookup(prompt, nullptr, true, 0)->matched_tokens == 0);
      } else if (scenario == "restore_failure") {
        runner->adapter->FailNextTransfer();
      }
      released.release();
      admission.join();
      const auto metrics = actual->Metrics();
      if (scenario == "control") {
        assert(actual->cached_tokens() == 531 && metrics.restore_attempts == 1);
        assert(metrics.restore_failures == 0);
        // Admission's own invalidation must not masquerade as advisory
        // staleness.
        assert(metrics.admission_index_version == lookup->index_version);
      } else {
        assert(actual->cached_tokens() == 0 && actual->restore_ms() == 0.0);
        assert(metrics.restore_attempts ==
               (scenario == "restore_failure" ? 1 : 0));
        assert(metrics.restore_failures ==
               (scenario == "restore_failure" ? 1 : 0));
        if (scenario == "restore_failure")
          assert(cache.Lookup(prompt, nullptr, true, 0)->matched_tokens == 0);
      }
    }
    runner->barrier = {};
    const auto metrics = actual->Metrics();
    // This record identifies only the pair, not its causal classification.
    // The offline classifier uses the same production logs/counters as R2/R3.
    Logger::Info(
        "test",
        "schema=cache-lookup-fixture-v1 case=" + scenario +
            " lookup_id=" + std::to_string(lookup->observation_id) +
            " acquisition_id=" + std::to_string(metrics.acquisition_id) +
            " total_tokens=620 actual_tokens=" +
            std::to_string(actual->cached_tokens()));
    actual->Invalidate();
  }
  Logger::SetLevel(old_level);
}
