// Host serialization/capacity contracts, NOT native Model/factory/restore
// proof.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen38_flash_next/continuation_layout.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace identity_probe {
namespace cache = gufo::cache;
namespace core = gufo::core;
namespace models = gufo::models;
using qfn::ExpectedKvPayloadVersion;
using qfn::KvCacheCompatibilityIdentity;
struct ExecutorProxy {
  std::uint32_t capacity{};
  std::uint32_t max_speculative() const { return capacity; }
};
struct ModelProxy {
  qfn::KvCacheType type;
  struct {
    std::uint32_t max_context;
  } options_;
  bool mtp_weights_;
  std::uint32_t concurrency;
  ExecutorProxy executor;
  ExecutorProxy* executor_ = &executor;
  const ModelProxy* get() const { return this; }
  qfn::KvCacheType CacheType() const { return type; }
  std::uint32_t DecodeConcurrency() const { return concurrency; }
  std::uint32_t PrefillCapacity() const noexcept;
  std::uint32_t PrefillThroughCapacity() const noexcept;
};
}  // namespace identity_probe

#include "continuation_identity_body.inc"

namespace {
using gufo::cache::Identity;
using gufo::server::QwenFlashNextCompatibilityIdentity;
using qfn::KvCacheType;
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
std::string Text(const Identity& bytes) {
  return {bytes.begin(), bytes.end()};
}
void Field(const Identity& bytes, const std::string& name,
           const std::string& value) {
  const auto text = Text(bytes);
  const auto field = name + "=" + value + "\n";
  const auto at = text.find(field);
  Require(at != std::string::npos && (at == 0 || text[at - 1] == '\n') &&
              text.find(field, at + field.size()) == std::string::npos,
          "identity field missing, duplicated, or changed");
}
Identity Persistent(KvCacheType type, bool mtp, std::uint32_t context = 256,
                    std::uint32_t draft = 7, std::uint32_t concurrency = 2,
                    std::string_view position = "absolute-v1",
                    std::string model = std::string(64, 'a'),
                    std::string auxiliary = std::string(64, 'b')) {
  return QwenFlashNextCompatibilityIdentity(model, auxiliary, mtp, context,
                                            draft, concurrency, position, type);
}
void PersistentContracts() {
  for (auto type :
       {KvCacheType::kF16, KvCacheType::kQ8_0, KvCacheType::kQ8_0Rotated}) {
    for (bool mtp : {false, true}) {
      const auto id = Persistent(type, mtp);
      Require(id == Persistent(type, mtp),
              "persistent identity is not deterministic");
      Field(id, "schema", "gufo-text-continuation-v2");
      Field(id, "artifact_id", "gguf-sha256-v1:" + std::string(64, 'a'));
      Field(id, "model_kind", "qwen38-flash-next");
      Field(id, "tokenizer", "embedded-in-artifact");
      Field(id, "chat_template", "qwen38-reasoning-compiled-v3");
      Field(
          id, "chat_template_reference_sha256",
          std::string(
              gufo::tokenization::QwenChatTemplate::OfficialTemplateSha256()));
      Field(id, "state_abi", "qwen38-flash-next-rocm-session-v1");
      Field(id, "payload_layout",
            type == KvCacheType::kQ8_0Rotated
                ? "qfn-rocm-session-snapshot-v20"
                : "qfn-rocm-session-snapshot-v19");
      Field(id, "context_tokens", "256");
      Field(id, "position_policy", "absolute-v1");
      Field(id, "adapters", "none");
      const auto numeric = qfn::KvCacheCompatibilityIdentity(type);
      Require(Text(id).find(numeric) != std::string::npos,
              "persistent identity omitted the independently tested numerical "
              "contract");
      Require(id != Persistent(type, mtp, 257), "context not bound");
      Require(id != Persistent(type, mtp, 256, 7, 2, "static-yarn-v1"),
              "position policy not bound");
      Require(id != Persistent(type, mtp, 256, 7, 2, "absolute-v1",
                               std::string(64, 'c')),
              "model artifact not bound");
      for (auto other :
           {KvCacheType::kF16, KvCacheType::kQ8_0, KvCacheType::kQ8_0Rotated})
        if (other != type)
          Require(id != Persistent(other, mtp), "cross-type identities alias");
      Require(id != Persistent(type, !mtp), "MTP mode not bound");
      if (mtp) {
        Field(id, "draft_backend", "qfn-mtp-v1");
        Field(id, "draft_artifact_id",
              "gguf-sha256-v1:" + std::string(64, 'b'));
        Field(id, "draft_max_tokens", "7");
        Field(id, "draft_cost_concurrency", "2");
        Require(id != Persistent(type, true, 256, 8),
                "proposal budget not bound");
        Require(id != Persistent(type, true, 256, 7, 3),
                "decode concurrency not bound");
        Require(id != Persistent(type, true, 256, 7, 2, "absolute-v1",
                                 std::string(64, 'a'), std::string(64, 'c')),
                "draft artifact not bound");
      } else {
        Require(Text(id).find("draft_") == std::string::npos,
                "MTP-off emitted draft fields");
        Require(id == Persistent(type, false, 256, 99, 13, "absolute-v1",
                                 std::string(64, 'a'), "unused"),
                "MTP-off identity depends on irrelevant draft options");
      }
    }
  }
  const auto reject = [](auto&& action) {
    bool rejected = false;
    try {
      action();
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected, "invalid persistence input accepted");
  };
  for (const auto& bad :
       {std::string{}, std::string(63, 'a'), std::string(65, 'a'),
        std::string(64, 'A'), std::string(64, 'g')}) {
    reject([&] {
      (void)Persistent(KvCacheType::kF16, false, 256, 7, 2, "absolute-v1", bad);
    });
    reject([&] {
      (void)Persistent(KvCacheType::kQ8_0, true, 256, 7, 2, "absolute-v1",
                       std::string(64, 'a'), bad);
    });
  }
  reject([&] { (void)Persistent(static_cast<KvCacheType>(99), false); });
}
std::uint32_t U32(const Identity& bytes, std::size_t at) {
  Require(at + 4 <= bytes.size(), "short identity field");
  std::uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i)
    value |= std::uint32_t{bytes[at + i]} << (8 * i);
  return value;
}
void AdapterAndCapacityContracts() {
  using namespace identity_probe;
  struct Case {
    std::uint32_t context, proposals, prefill, through, verification;
  };
  // Independently named expected capacities: proposals + bonus row, clamped
  // by available prefill-through capacity; UINT32_MAX must not wrap to zero.
  constexpr std::array cases{Case{256, 7, 256, 256, 8}, Case{4, 7, 4, 4, 4},
                             Case{8192, 5000, 4096, 4224, 4224},
                             Case{256, 0, 256, 256, 1},
                             Case{8192, UINT32_MAX, 4096, 4224, 4224}};
  for (auto type :
       {KvCacheType::kF16, KvCacheType::kQ8_0, KvCacheType::kQ8_0Rotated}) {
    for (bool mtp : {false, true}) {
      for (const auto c : cases) {
        ModelProxy model{type, {c.context}, mtp, 2, {}};
        model.executor.capacity = Capacity(&model, c.proposals);
        Require(model.PrefillCapacity() == c.prefill &&
                    model.PrefillThroughCapacity() == c.through,
                "production prefill capacities changed");
        Require(
            model.executor.capacity == (mtp ? c.verification : 1),
            "verification capacity confused with proposal count or overflowed");
        auto local = Process(&model, c.context, mtp, c.proposals);
        Require(local == Process(&model, c.context, mtp, c.proposals),
                "local identity unstable");
        ModelProxy other{type, {c.context}, mtp, 2, {}};
        Require(local != Process(&other, c.context, mtp, c.proposals),
                "different model instances alias");
        Require(local != Process(&model, c.context + 1, mtp, c.proposals),
                "process context not bound");
        Require(local != Process(&model, c.context, !mtp, c.proposals),
                "process MTP mode not bound");
        Require(local != Process(&model, c.context, mtp, c.proposals ^ 1U),
                "process proposal budget not bound");
        ++model.concurrency;
        Require(local != Process(&model, c.context, mtp, c.proposals),
                "process concurrency not bound");
        --model.concurrency;
        model.type =
            type == KvCacheType::kF16 ? KvCacheType::kQ8_0 : KvCacheType::kF16;
        Require(local != Process(&model, c.context, mtp, c.proposals),
                "process storage/numerical type not bound");
        model.type = type;
        Field(local, "schema", "gufo-process-text-continuation-v1");
        Field(local, "draft_max_tokens", std::to_string(c.proposals));
        Field(local, "draft_cost_concurrency", "2");
        const auto persistent = Persistent(type, mtp, c.context, c.proposals);
        if (mtp) {
          // A loaded MTP executor can also serve an autoregressive session.
          // Identity binds its actual verification capacity, not mode * budget.
          const auto base = Persistent(type, false, c.context, c.proposals);
          const auto autoregressive = Adapter(
              &model, core::SessionMode::kAutoregressive, c.context, base);
          Require(U32(autoregressive, base.size() + 12) == 0 &&
                      U32(autoregressive, base.size() + 16) == c.verification,
                  "MTP-loaded autoregressive adapter lost executor capacity");
        }
        Require(local != persistent,
                "process identity aliases durable identity");
        for (const auto& base : {local, persistent}) {
          const auto encoded = Adapter(&model,
                                       mtp ? core::SessionMode::kSpeculative
                                           : core::SessionMode::kAutoregressive,
                                       c.context, base);
          const auto numeric = qfn::KvCacheCompatibilityIdentity(type);
          Require(encoded.size() == base.size() + 32 + numeric.size() &&
                      std::equal(base.begin(), base.end(), encoded.begin()),
                  "adapter identity base/shape changed");
          const std::array expected{
              3U,
              type == KvCacheType::kQ8_0Rotated ? 20U : 19U,
              c.context,
              mtp ? 1U : 0U,
              mtp ? c.verification : 1U,
              c.prefill,
              2U,
              static_cast<std::uint32_t>(numeric.size())};
          for (std::size_t i = 0; i < expected.size(); ++i)
            Require(U32(encoded, base.size() + i * 4) == expected[i],
                    "adapter named LE field mismatch");
          Require(std::equal(numeric.begin(), numeric.end(),
                             encoded.begin() + base.size() + 32),
                  "adapter numerical suffix changed");
        }
      }
    }
  }
}
}  // namespace
int main() {
  try {
    PersistentContracts();
    AdapterAndCapacityContracts();
    std::cout << "continuation_identity: current production host bodies PASS; "
                 "model/executor proxies, native NOT RUN\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
