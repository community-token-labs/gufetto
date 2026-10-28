#include "src/cli/serve/prefill_prerequisites.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace gufo::server::detail;

int main() {
  // B1/B2: Gufo has no added checkpoint grid. Ragged offsets, every side of
  // a real boundary, final one/two-token tails and native capacity edges.
  for (const auto offset : {std::size_t{999}, std::size_t{1601}}) {
    for (const auto budget : {std::size_t{1}, std::size_t{98}, std::size_t{99},
                              std::size_t{100}, std::size_t{101}}) {
      const auto cut =
          ClipSinglePass(offset, offset + 200, budget, 4096, offset + 100);
      assert(cut == std::min(budget, std::size_t{100}));
    }
    assert(ClipSinglePass(offset, offset + 1, 4096, 4096, 0) == 1);
    assert(ClipSinglePass(offset, offset + 2, 1, 4096, 0) == 1);
    assert(ClipSinglePass(offset, offset + 2, 4096, 4096, 0) == 2);
  }
  assert(ClipSinglePass(0, 5000, 4097, 4096, 0) == 4096);
  assert(ClipSinglePass(0, 4224, 4224, 4224, 0) == 4224);
  assert(ClipSinglePass(4224, 4224, 1, 4224, 0) == 0);
  assert(ClipSinglePass(4225, 4224, 1, 4224, 0) == 0);

  PrefillPrerequisites config;
  assert(config.mode == PrefillPrerequisiteMode::kLegacy);
  assert(!config.Enabled());
  config.mode = PrefillPrerequisiteMode::kAdaptive;
  config.tokens = 1024;
  config.Validate(4);
  assert(SelectPrerequisiteBudget(config, 0, {}).prompt_tokens == 4096);
  assert(SelectPrerequisiteBudget(config, 1, {}).prompt_tokens == 1024);
  // SC is a TOTAL-row budget, not an AD prompt grant. Reserve against upper
  // bounds, never refund prompt rows after a peer drops out or lookup trims.
  config.mode = PrefillPrerequisiteMode::kSerialShadow;
  const std::array<std::size_t, 3> caps{8, 2, 8};
  const auto sc = SelectPrerequisiteBudget(config, 3, caps);
  assert(sc.verifier_rows_upper == 18 && sc.prompt_tokens == 1006);
  config.tokens = 64;
  const std::array<std::size_t, 7> full{8, 8, 8, 8, 8, 8, 8};
  const auto minimum = SelectPrerequisiteBudget(config, 7, full);
  assert(minimum.verifier_rows_upper == 56 && minimum.prompt_tokens == 8);
  for (const auto invalid :
       {std::size_t{0}, std::size_t{63}, std::size_t{4225}}) {
    config.tokens = invalid;
    bool rejected = false;
    try {
      config.Validate(8);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  config.tokens = 1024;
  bool rejected = false;
  try {
    config.Validate(9);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  // E1/B5: preserve the backend route and its sampler ownership. The same
  // final singleton count does NOT imply the same legacy route.
  assert(QfnBackendRoute(true, 1, 1) == DecodeRoute::kGenericScalar);
  assert(QfnBackendRoute(false, 1, 32) == DecodeRoute::kGenericScalar);
  assert(QfnBackendRoute(false, 3, 32) == DecodeRoute::kGenericSequential);
  assert(QfnBackendRoute(true, 1, 32) == DecodeRoute::kEngineScalar);
  const auto batch = QfnBackendRoute(true, 3, 1);
  assert(batch == DecodeRoute::kEngineBatch);
  // Native model filtering must retain this backend route; pd_profile_test
  // separately checks its representation. No mixed eligibility is implemented.
  assert(QfnBackendRoute(true, 0, 1) == DecodeRoute::kNone);

  // P1/P2: a frozen cohort drains once, independent of decode_due or later
  // newcomers. Removing a blocked/cancelled peer never stops another member.
  ReadyDecodeCohort cohort;
  const std::array<std::uint64_t, 3> ready{11, 22, 33};
  cohort.Freeze(ready);
  assert(cohort.Contains(11) && !cohort.Contains(44));
  cohort.ServedOrDeferred(22);
  assert(cohort.Contains(11) && cohort.Contains(33));
  cohort.ServedOrDeferred(11);
  cohort.ServedOrDeferred(11);
  assert(cohort.Contains(33) && !cohort.Empty());
  cohort.ServedOrDeferred(33);
  assert(cohort.Empty());
  bool duplicate_rejected = false;
  try {
    cohort.Freeze(std::array<std::uint64_t, 2>{44, 44});
  } catch (const std::invalid_argument&) {
    duplicate_rejected = true;
  }
  assert(duplicate_rejected && cohort.Empty());
  cohort.Freeze(std::array<std::uint64_t, 1>{44});
  assert(cohort.Contains(44) && !cohort.Contains(11));
  std::cout << "prefill prerequisites policy/route/cohort fixtures PASS\n";
}
