#include "src/core/startup_admission.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using namespace gufo::core;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

struct FakeClock {
  Clock::time_point current{};

  Clock::time_point Now() const { return current; }
  void Advance(Clock::duration elapsed) { current += elapsed; }
};

void TestMeasuredRegression() {
  const ExecutionCapacityObservation before{50000000000ULL, 36795928576ULL,
                                            18397964288ULL};
  const ExecutionCapacityObservation after{41000000000ULL, 27725746176ULL,
                                           13862873088ULL};
  const auto debit = MeasuredExecutionDebit(before, after);
  assert(debit.host == 9000000000ULL);
  assert(debit.device == 9070182400ULL);
  const auto pending = AddMemoryBytes(797000000, 749248);
  assert(pending == 797749248);
  assert(!FitsMeasuredExecution(after, debit, 3, pending));
  assert(FitsMeasuredExecution(after, debit, 2, pending));
  // Omitting the hard pending obligations incorrectly admits three more slots.
  assert(FitsMeasuredExecution(after, debit, 3, 0));

  // The original logical precheck admitted this demand/capacity pair, unlike
  // the measured projection above. Payload is not the physical slot debit.
  assert(FitsMemoryBudget(37332799488ULL, 36833485904ULL, 0));

  const auto exact = AddMemoryBytes(3 * debit.device, pending);
  assert(exact == 28008296448ULL);
  assert(FitsMeasuredExecution({exact, exact, 0}, debit, 3, pending));
  assert(!FitsMeasuredExecution({exact, exact - 1, 0}, debit, 3, pending));

  // Preserve the exact post-slot observation for the separate cache guard.
  const auto protected_cache =
      ProtectCacheHeadroom(after.automatic, after.Capacity(), pending);
  assert(protected_cache.ceiling == after.device - pending);
  assert(protected_cache.automatic == after.automatic - pending / 2);
}

void TestIndependentLanesAndOverhead() {
  const ExecutionCapacityObservation before{1000, 1200, 500};
  const ExecutionCapacityObservation after{900, 700, 350};
  assert(before.Capacity() == 1000);
  assert(after.Capacity() == 700);
  const auto debit = MeasuredExecutionDebit(before, after);
  assert(debit.host == 100 && debit.device == 500);
  // A delta of the minima is 300: twice that would fit, unlike the device lane.
  assert(FitsMemoryBudget(after.Capacity(),
                          2 * (before.Capacity() - after.Capacity()), 0));
  assert(!FitsMeasuredExecution(after, debit, 2, 0));
  assert(FitsMeasuredExecution(after, debit, 1, 200));
  assert(!FitsMeasuredExecution(after, debit, 1, 201));

  const auto host_debit =
      MeasuredExecutionDebit({1200, 1000, 500}, {700, 900, 350});
  assert(host_debit.host == 500 && host_debit.device == 100);
  assert(!FitsMeasuredExecution({700, 900, 350}, host_debit, 2, 0));
  assert(FitsMeasuredExecution({700, 900, 0}, host_debit, 1, 200));

  // Equal logical payloads can have unequal physical overhead on either lane.
  const auto first = MeasuredExecutionDebit({1000, 1000, 0}, {900, 800, 0});
  const auto second = MeasuredExecutionDebit({900, 800, 0}, {600, 650, 0});
  const auto maximum = MaxExecutionDebit(first, second);
  assert(maximum.host == 300 && maximum.device == 200);
  assert(MaxExecutionDebit(second, first).host == maximum.host);
  assert(MaxExecutionDebit(second, first).device == maximum.device);
  assert(FitsMeasuredExecution({600, 650, 0}, first, 3, 0));
  assert(!FitsMeasuredExecution({600, 650, 0}, maximum, 3, 0));
  assert(FitsMeasuredExecution({600, 650, 0}, maximum, 2, 0));
}

void TestSaturationAndOverflow() {
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  constexpr auto zero = MeasuredExecutionDebit({0, 0, max}, {max, max, 0});
  static_assert(zero.host == 0 && zero.device == 0);
  constexpr auto unchanged = MeasuredExecutionDebit({5, 6, 0}, {5, 6, max});
  static_assert(unchanged.host == 0 && unchanged.device == 0);
  constexpr auto mixed = MeasuredExecutionDebit({8, 3, 0}, {2, 9, 0});
  static_assert(mixed.host == 6 && mixed.device == 0);
  constexpr auto loss = MeasuredExecutionDebit({max, max, 0}, {0, 0, max});
  static_assert(loss.host == max && loss.device == max);

  assert(FitsMeasuredExecution({max, max, 0}, {1, 1}, max, 0));
  assert(!FitsMeasuredExecution({max, max, 0}, {2, 1}, max, 0));
  assert(!FitsMeasuredExecution({max, max, 0}, {1, 2}, max, 0));
  assert(!FitsMeasuredExecution({max, max, 0}, {max, max}, 1, 1));
  assert(FitsMeasuredExecution({max, max, 0}, {max, max}, 0, max));
  assert(FitsMeasuredExecution({max, max, 0}, {0, 0}, max, max));
  assert(!FitsMeasuredExecution({max - 1, max, 0}, {0, 0}, 0, max));
  assert(!FitsMeasuredExecution({max, max - 1, 0}, {0, 0}, 0, max));
  assert(FitsMeasuredExecution({0, 0, max}, {0, 0}, max, 0));
  assert(!FitsMeasuredExecution({0, 0, max}, {0, 0}, 0, 1));
}

void TestRoundingAndEmptySearch() {
  FakeClock clock;
  auto now = [&] { return clock.Now(); };
  for (const std::size_t requested : {0, 1, 4095, 4096}) {
    const auto result = SearchVerifiedContext(
        requested,
        [](std::size_t, Clock::time_point) {
          assert(false);
          return true;
        },
        {}, now);
    assert(!result.verified_context && result.probes == 0);
    assert(!result.step_limit_hit && !result.time_limit_hit);
    assert(result.elapsed == 0s);
  }
  for (const std::size_t requested : {4097, 8192, 8193, 20000}) {
    const auto result = SearchVerifiedContext(
        requested,
        [&](std::size_t context, Clock::time_point deadline) {
          assert(context > 0 && context % 4096 == 0 && context < requested);
          assert(deadline == Clock::time_point{} + 30s);
          return true;
        },
        {}, now);
    assert(result.verified_context == ((requested - 1) / 4096) * 4096);
    assert(!result.step_limit_hit && !result.time_limit_hit);
  }
  // Exercise the public default steady-clock callback as well.
  const auto real_clock = SearchVerifiedContext(
      4097, [](std::size_t context, Clock::time_point deadline) {
        assert(context == 4096 && deadline > Clock::now());
        return true;
      });
  assert(real_clock.verified_context == 4096 && real_clock.probes == 1);
}

void TestProbeBoundAndFallback() {
  FakeClock clock;
  auto now = [&] { return clock.Now(); };
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  std::size_t last_verified = 0;
  std::size_t calls = 0;
  const auto result = SearchVerifiedContext(
      max,
      [&](std::size_t context, Clock::time_point) {
        assert(context > last_verified && context < max && context % 4096 == 0);
        last_verified = context;
        ++calls;
        return true;
      },
      {}, now);
  assert(calls == 8 && result.probes == 8);
  assert(result.step_limit_hit && !result.time_limit_hit);
  assert(result.verified_context == last_verified);

  // A successful low probe survives later failures and the step limit.
  std::array<std::size_t, 3> contexts{};
  calls = 0;
  const auto fallback = SearchVerifiedContext(
      17,
      [&](std::size_t context, Clock::time_point) {
        contexts[calls++] = context;
        return context <= 8;
      },
      {1, 3, 30s}, now);
  assert((contexts == std::array<std::size_t, 3>{8, 12, 10}));
  assert(fallback.verified_context == 8 && fallback.probes == 3);
  assert(fallback.step_limit_hit && !fallback.time_limit_hit);

  // Index arithmetic remains bounded even at SIZE_MAX with unit granularity.
  const auto unit = SearchVerifiedContext(
      max,
      [&](std::size_t context, Clock::time_point) {
        assert(context > 0 && context < max);
        return true;
      },
      {1, std::numeric_limits<std::size_t>::digits + 1, 30s}, now);
  assert(unit.verified_context == max - 1 && !unit.step_limit_hit);
}

void TestNoFitAndNonlinearSteps() {
  FakeClock clock;
  auto now = [&] { return clock.Now(); };
  const auto nofit = SearchVerifiedContext(
      100 * 4096, [](std::size_t, Clock::time_point) { return false; }, {},
      now);
  assert(!nofit.verified_context);
  assert(nofit.probes <= 8 && !nofit.step_limit_hit && !nofit.time_limit_hit);

  // Full verification sees allocation steps, not linear bytes/token scaling.
  const auto stepped = SearchVerifiedContext(
      65 * 4096,
      [](std::size_t context, Clock::time_point) {
        const auto units = context / 4096;
        const auto allocated = 100 + ((units + 6) / 7) * 90;
        return FitsMemoryBudget(400, allocated, 20);
      },
      {}, now);
  assert(stepped.verified_context == 21 * 4096);
  assert(!stepped.step_limit_hit && !stepped.time_limit_hit);
}

void TestTimeBounds() {
  FakeClock clock;
  auto now = [&] { return clock.Now(); };
  std::size_t last_on_time = 0;
  std::size_t calls = 0;
  const auto capped = SearchVerifiedContext(
      1000 * 4096,
      [&](std::size_t context, Clock::time_point deadline) {
        assert(clock.Now() < deadline);
        assert(deadline == Clock::time_point{} + 30s);
        clock.Advance(10s);
        if (clock.Now() < deadline)
          last_on_time = context;
        ++calls;
        return true;
      },
      {}, now);
  assert(calls == 3 && capped.probes == 3 && capped.elapsed == 30s);
  assert(capped.time_limit_hit && !capped.step_limit_hit);
  assert(capped.verified_context == last_on_time);

  clock = {};
  calls = 0;
  const auto late = SearchVerifiedContext(
      1000 * 4096,
      [&](std::size_t, Clock::time_point deadline) {
        assert(clock.Now() < deadline);
        ++calls;
        clock.Advance(31s);
        return true;
      },
      {}, now);
  assert(!late.verified_context && late.probes == 1 && calls == 1);
  assert(late.time_limit_hit && late.elapsed == 31s);

  clock = {};
  calls = 0;
  const auto before_probe = SearchVerifiedContext(
      8192,
      [](std::size_t, Clock::time_point) {
        assert(false);
        return true;
      },
      {},
      [&] {
        if (++calls == 2)
          clock.Advance(30s);
        return clock.Now();
      });
  assert(before_probe.probes == 0 && before_probe.time_limit_hit);
  assert(before_probe.elapsed == 30s && !before_probe.verified_context);

  clock.current = Clock::time_point::max() - 1s;
  const auto saturated_deadline = SearchVerifiedContext(
      4097,
      [](std::size_t, Clock::time_point deadline) {
        assert(deadline == Clock::time_point::max());
        return true;
      },
      {}, now);
  assert(saturated_deadline.verified_context == 4096);
}

void TestLimitsAndExceptions() {
  FakeClock clock;
  auto now = [&] { return clock.Now(); };
  auto no_probe = [](std::size_t, Clock::time_point) {
    assert(false);
    return true;
  };
  for (const auto budget : {0s, -1s}) {
    const auto result =
        SearchVerifiedContext(8192, no_probe, {4096, 8, budget}, now);
    assert(result.time_limit_hit && !result.step_limit_hit &&
           result.probes == 0);
  }
  const auto zero_steps =
      SearchVerifiedContext(8192, no_probe, {4096, 0, 30s}, now);
  assert(zero_steps.step_limit_hit && !zero_steps.time_limit_hit);
  assert(zero_steps.probes == 0 && !zero_steps.verified_context);

  bool caught = false;
  try {
    (void)SearchVerifiedContext(8192, no_probe, {0, 8, 30s}, now);
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  assert(caught);

  // Preserve provider exception identity; do not translate unknown failure to
  // no-fit or silently return a previously verified context.
  struct ProviderFailure {
  } failure;
  std::size_t calls = 0;
  caught = false;
  try {
    (void)SearchVerifiedContext(
        100 * 4096,
        [&](std::size_t, Clock::time_point) {
          if (++calls == 2)
            throw &failure;
          return true;
        },
        {}, now);
  } catch (ProviderFailure* error) {
    caught = error == &failure;
  }
  assert(caught && calls == 2);

  caught = false;
  try {
    (void)SearchVerifiedContext(8192, no_probe, {}, []() -> Clock::time_point {
      throw std::runtime_error("clock failure");
    });
  } catch (const std::runtime_error&) {
    caught = true;
  }
  assert(caught);
}

}  // namespace

int main() {
  TestMeasuredRegression();
  TestIndependentLanesAndOverhead();
  TestSaturationAndOverflow();
  TestRoundingAndEmptySearch();
  TestProbeBoundAndFallback();
  TestNoFitAndNonlinearSteps();
  TestTimeBounds();
  TestLimitsAndExceptions();
  std::cout << "Startup admission checks passed.\n";
}
