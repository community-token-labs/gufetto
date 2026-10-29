// Pattern only: SGLang #46975 (3355d668). No upstream implementation copied.
#ifndef GUFO_TEST_CACHE_COMPLETION_FANOUT_CHECKS_HPP_
#define GUFO_TEST_CACHE_COMPLETION_FANOUT_CHECKS_HPP_

namespace completion_fanout {
// One deterministic event per attention target/draft and recurrent participant.
// No sleeps or device work: the real capture waits on each wrapped completion.
struct Events {
  struct Participant {
    std::binary_semaphore entered{0}, ready{0};
    std::atomic<unsigned> submitted{}, waits{}, settled{};
    std::atomic<bool> delivered{}, failed{};
  };
  std::array<Participant, 3> participants;
  std::atomic<bool> timed_out{};
  static std::size_t Index(cache::ComponentId id) {
    if (id == kTarget)
      return 0;
    if (id == kDraft)
      return 1;
    assert(id == kRecurrent);
    return 2;
  }
  void Deliver(std::size_t index, bool fail = false) {
    auto& p = participants[index];
    p.failed = fail;
    p.delivered = !fail;
    p.ready.release();
  }
};
class Signal final : public cache::CompletionSignal {
public:
  Signal(Events& events, std::size_t index, cache::Completion inner)
      : events_(events), index_(index), inner_(std::move(inner)) {}
  bool Ready() const noexcept override {
    return events_.participants[index_].delivered && inner_.Ready();
  }
  cache::TransferResult Wait() noexcept override {
    auto& p = events_.participants[index_];
    ++p.waits;
    p.entered.release();
    const bool released = p.ready.try_acquire_for(std::chrono::seconds(5));
    if (!released)
      events_.timed_out = true;
    // Even an absent/failed event must drain the underlying read before its
    // destination can unwind. A later notification cannot write into it.
    const auto result = inner_.Wait();
    ++p.settled;
    return released && p.delivered && !p.failed
               ? result
               : cache::TransferResult::kFailed;
  }

private:
  Events& events_;
  const std::size_t index_;
  cache::Completion inner_;
};

void CaptureWaitsForEveryParticipant(std::size_t last, bool missing) {
  Events events;
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  const std::vector<TextRunnerToken> prompt{71, 72, 73};
  auto source = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(source->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  const auto expected_hash = runner->adapter->RecurrentHash(state.slot);
  assert(source->TryReserveSnapshot(source->CaptureBytes(), prompt.size(),
                                    false, SnapshotPurpose::kContinuation, {}));
  runner->adapter->capture_completion = [&](auto id, auto completion) {
    const auto index = Events::Index(id);
    assert(!completion.Ready());
    assert(++events.participants[index].submitted == 1);
    return cache::Completion(
        std::make_unique<Signal>(events, index, std::move(completion)));
  };
  // Three tokens are tails, not borrowable complete chunks: the actual cache
  // capture must submit one transfer for each of the three adapter components.
  std::atomic<bool> published{false};
  std::exception_ptr failure;
  std::jthread capturing([&] {
    try {
      auto snapshot = source->Capture();
      assert(snapshot);
      published =
          source->PublishSnapshot(prompt, std::move(snapshot), false) != 0;
    } catch (...) {
      failure = std::current_exception();
    }
  });
  assert(
      events.participants[0].entered.try_acquire_for(std::chrono::seconds(5)));
  for (const auto& p : events.participants) {
    assert(p.submitted == 1);  // including recurrent, not attention alone
    assert(!p.delivered);
  }
  assert(!published);
  assert(cache.Lookup(prompt, nullptr, true, 0)->matched_tokens == 0);
  // Deliver out of wait order. Every participant takes a turn being last;
  // releasing the other two must never publish an incomplete checkpoint.
  for (std::size_t i = 3; i-- > 0;)
    if (i != last)
      events.Deliver(i);
  if (last != 0)
    assert(events.participants[last].entered.try_acquire_for(
        std::chrono::seconds(5)));
  assert(!events.participants[last].delivered);
  assert(events.participants[last].settled == 0);
  assert(!published);
  assert(cache.Lookup(prompt, nullptr, true, 0)->matched_tokens == 0);
  events.Deliver(last, missing);
  capturing.join();
  runner->adapter->capture_completion = {};
  assert(!events.timed_out);
  for (const auto& p : events.participants)
    assert(p.submitted == 1 && p.waits == 1 && p.settled == 1);
  if (missing) {
    // A required participant received no success event. The actual capture
    // must reject it, not publish an attention-only or otherwise partial state.
    assert(failure && !published);
    bool transfer_failed = false;
    try {
      std::rethrow_exception(failure);
    } catch (const std::runtime_error& error) {
      transfer_failed =
          std::string_view(error.what()) == "component cache transfer failed";
    }
    assert(transfer_failed);
    assert(source->Metrics().captures == 0);
    assert(source->Metrics().captured_checkpoint_ids.empty());
    assert(cache.Lookup(prompt, nullptr, true, 0)->matched_tokens == 0);
    events.Deliver(last);  // late notification after rejection/drain
    assert(!published && events.participants[last].settled == 1);
    assert(cache.Lookup(prompt, nullptr, true, 0)->matched_tokens == 0);
  } else {
    if (failure)
      std::rethrow_exception(failure);
    assert(published && source->Metrics().captures == 1);
    assert(cache.Lookup(prompt, nullptr, true, 0)->matched_tokens ==
           prompt.size());
    // Independent consumer: the published hybrid checkpoint restores complete
    // positions AND recurrent bytes into another slot, while source stays busy.
    auto restored = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(restored->cached_tokens() == prompt.size());
    auto& restored_state = dynamic_cast<State&>(restored->state());
    assert(runner->adapter->Positions(restored_state.slot) ==
           runner->adapter->Positions(state.slot));
    assert(runner->adapter->RecurrentHash(restored_state.slot) ==
           expected_hash);
  }
}
void Check() {
  for (std::size_t last = 0; last < 3; ++last) {
    CaptureWaitsForEveryParticipant(last, false);
    CaptureWaitsForEveryParticipant(last, true);
  }
}
}  // namespace completion_fanout
#endif  // GUFO_TEST_CACHE_COMPLETION_FANOUT_CHECKS_HPP_
