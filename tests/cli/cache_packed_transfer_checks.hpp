// Ported from archive/nightly-20261028 packed ingress CPU fixtures.
// Real transfer/pin admission only; no native model execution.
namespace packed_transfer_cpu {
namespace qfn = models::qwen38_flash_next;
constexpr std::uint32_t kContext = 256, kElements = 512;
constexpr std::size_t kRam = 4ULL << 20;
constexpr auto kPlain = qfn::KvCacheType::kQ8_0;
constexpr auto kConfiguration = "fixture=qfn-packed-cpu-admission";
qfn::Config Geometry() {
  qfn::Config c;
  c.num_layers = 48;
  c.full_attention_interval = 4;
  c.context_length = kContext;
  c.hidden_size = 8;
  c.hc_count = 1;
  c.vocab_size = 32;
  c.num_kv_heads = 2;
  c.num_heads = 24;
  c.head_dim = 256;
  c.rotary_dim = 64;
  c.indexer_head_dim = 8;
  c.indexer_top_k = 16;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 2;
  c.ssm_head_dim = 4;
  c.ssm_num_k_heads = c.ssm_num_v_heads = 1;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_conv_kernel = 4;
  return c;
}
bool Packed(qfn::ContinuationPart part) {
  return part == qfn::ContinuationPart::kKey ||
         part == qfn::ContinuationPart::kValue ||
         part == qfn::ContinuationPart::kDraftKey ||
         part == qfn::ContinuationPart::kDraftValue;
}
cache::CompatibilityDigest Digest(std::span<const std::uint8_t> bytes) {
  crypto::Sha256Hasher hash;
  hash.Update(bytes);
  return hash.Finish();
}
cache::Identity Identity(qfn::KvCacheType type) {
  const auto text =
      "cpu-admission-only;" + qfn::KvCacheCompatibilityIdentity(type);
  return {text.begin(), text.end()};
}
// Only the registered descriptor inventory is needed for Import/startup. Any
// slot/transfer use is a test error, rather than a simulated native restore.
class LayoutAdapter final : public cache::Adapter {
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
  [[noreturn]] static void Unexpected() {
    throw std::logic_error(
        "CPU ingress fixture must not execute/restore a slot");
  }

public:
  explicit LayoutAdapter(qfn::KvCacheType type) : identity_(Identity(type)) {
    const qfn::ContinuationLayout layout(Geometry(), true, kContext, 1, type);
    for (const auto& c : layout.Components())
      descriptors_.push_back(c.descriptor);
  }
  cache::Capabilities GetCapabilities() const override { return {true, true}; }
  std::span<const cache::ComponentDescriptor> Components() const override {
    return descriptors_;
  }
  cache::Identity CompatibilityIdentity() const override { return identity_; }
  std::unique_ptr<cache::Slot> CreateSlot(cache::MutationGuard&) override {
    Unexpected();
  }
  std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override {
    Unexpected();
  }
  std::vector<cache::Rows> PlanPrefill(const cache::Slot&, cache::Rows,
                                       cache::Rows) const override {
    Unexpected();
  }
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override {
    Unexpected();
  }
  cache::Completion CapturePrivate(const cache::Slot&, cache::ComponentId,
                                   std::span<std::byte>,
                                   cache::Stream&) override {
    Unexpected();
  }
  cache::Completion CopyRowsOut(const cache::Slot&, cache::ComponentId,
                                cache::Rows, cache::Rows, std::span<std::byte>,
                                cache::Stream&) override {
    Unexpected();
  }
  cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId, cache::Rows,
                               cache::Rows, std::span<const std::byte>,
                               cache::Stream&) override {
    Unexpected();
  }
  cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                std::span<const std::byte>,
                                cache::Stream&) override {
    Unexpected();
  }
  bool Validate(cache::Slot&,
                std::span<const cache::ComponentPosition>) override {
    Unexpected();
  }
  bool Invalidate(cache::Slot&) noexcept override { std::terminate(); }
};
class AdmissionRunner final : public Runner {
public:
  qfn::KvCacheType type;
  bool fingerprint{true}, foreign_abi{};
  mutable cache::ResourceLedger* ledger{};
  mutable std::size_t allocations{}, finishes{};
  explicit AdmissionRunner(qfn::KvCacheType type) : type(type) {}
  std::vector<TextRunnerToken> TokenizeControl(
      std::string_view) const override {
    return {10, 11};
  }
  TextRunnerDescriptor Descriptor() const override {
    auto d = Runner::Descriptor();
    d.max_context = kContext;
    d.state_abi = foreign_abi ? "foreign-session-v1"
                              : "qwen38-flash-next-rocm-session-v1";
    if (fingerprint) {
      std::uint32_t version = qfn::kKvCacheNumericsEpoch;
#if 1  // Both packed modes are unconditional in the cleaned q8 topic.
      version = qfn::ExpectedKvPayloadVersion(type);
#endif
      d.persistence = TextRunnerPersistenceDescriptor{Identity(type), version};
    }
    return d;
  }
  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    this->ledger = &ledger;
    auto r = std::make_unique<ComponentCacheResources>();
    r->adapter = std::make_unique<LayoutAdapter>(type);
    r->owner = std::make_shared<int>(1);
    r->stream = [] { return std::make_unique<FakeStream>(); };
    r->staging = [](std::size_t size) {
      auto data = std::make_shared<std::vector<std::uint8_t>>(size);
      return cache::StagingAllocation{data, *data};
    };
    r->allocate = [this, &ledger](const cache::PayloadRequest& request) {
      ++allocations;
      assert(request.bytes && request.bytes < kRam);
      struct Backing {
        cache::ResourceCharge pool;
        std::vector<std::byte> bytes;
      };
      auto backing = std::make_shared<Backing>(Backing{
          ledger.Reserve(cache::ResourceCategory::kBackingFree, request.bytes)
              .Convert(),
          std::vector<std::byte>(request.bytes)});
      auto reservation = std::make_shared<cache::ResourceReservation>(
          backing->pool.ReserveBacking(request.category));
      auto owner = std::shared_ptr<const void>(backing, backing->bytes.data());
      return ComponentCaptureBuffer{backing->bytes,
                                    [this, reservation, owner] {
                                      ++finishes;
                                      return cache::Payload::Committed(
                                          reservation->Convert(), owner);
                                    },
                                    nullptr,
                                    {}};
    };
    detail::InstallQfnPackedKvIngressValidation(*r, Geometry(), true, kContext,
                                                type);
    return r;
  }
};
struct Artifact {
  CheckpointTransferHeader header;
  std::vector<std::vector<std::uint8_t>> images;
  std::array<std::size_t, 4> kv_images{};
};
Artifact Canonical(qfn::KvCacheType type) {
  Artifact a;
  auto& h = a.header;
  h.configuration = kConfiguration;
  h.identity = Identity(type);
  h.stable_tokens = 2;
  h.decode_ready = true;
  h.manifest.checkpoint = {1000};
  h.manifest.lineage = {1000};
  h.manifest.tokens = {10, 11};
  h.manifest.compatibility = Digest(h.identity);
  const qfn::ContinuationLayout layout(Geometry(), true, kContext, 1, type);
  const auto positions = layout.Positions(2, 2, 0, 0);
  std::array<bool, 4> found{};
  for (std::size_t i = 0; i < layout.Components().size(); ++i) {
    const auto& c = layout.Components()[i];
    const auto& d = c.descriptor;
    cache::DiskComponent component{.descriptor = d, .position = positions[i]};
    const auto bytes =
        d.row_bytes ? positions[i].valid_rows * d.row_bytes : d.state_bytes;
    if (bytes) {
      const auto image = a.images.size();
      a.images.emplace_back(bytes);
      if (Packed(c.part)) {
        std::vector<float> values(kElements);
        values[0] = 127;
        values[1] = 0.5F;
        values[2] = -0.5F;
        values[32] = 1e-7F;
        for (std::size_t off = 0; off < bytes; off += d.row_bytes)
          qfn::EncodeKvRow(values, type,
                           std::as_writable_bytes(std::span(a.images.back()))
                               .subspan(off, d.row_bytes));
        const auto part = c.part == qfn::ContinuationPart::kKey        ? 0U
                          : c.part == qfn::ContinuationPart::kValue    ? 1U
                          : c.part == qfn::ContinuationPart::kDraftKey ? 2U
                                                                       : 3U;
        if (!found[part]) {
          a.kv_images[part] = image;
          found[part] = true;
        }
      }
      cache::DiskFileId id{};
      id[0] = static_cast<std::uint8_t>(image + 1);
      const cache::DiskPayload payload{id, bytes,
                                       cache::DiskChecksum(a.images.back())};
      if (d.row_bytes)
        component.tail = payload;
      else
        component.private_state = payload;
    }
    h.manifest.components.push_back(std::move(component));
  }
  assert(std::ranges::all_of(found, [](bool b) { return b; }));
  return a;
}
void Poison(Artifact& a, unsigned part, unsigned fault) {
  auto& image = a.images[a.kv_images.at(part)];
  auto block = std::span(image).last(sizeof(qfn::KvBlockQ8_0));
  if (fault == 0) {
    block[offsetof(qfn::KvBlockQ8_0, values)] = 0x80;
  } else {
    const auto scale =
        std::array<std::uint16_t, 5>{0xbc00, 0x8000, 0x7c00, 0x7e00, 0x7bff}.at(
            fault - 1);
    block[0] = scale & 255;
    block[1] = scale >> 8;
    std::fill(block.begin() + 2, block.end(), 0);
    if (fault == 5)
      block[2] = 127;
  }
  // Both real codecs accept the metadata; the bytes' CRC is correct even
  // though their last block (in row two) is semantically inadmissible.
  std::size_t next = 0;
  for (auto& c : a.header.manifest.components) {
    if (c.tail)
      c.tail->checksum = cache::DiskChecksum(a.images[next++]);
    if (c.private_state)
      c.private_state->checksum = cache::DiskChecksum(a.images[next++]);
  }
  (void)cache::DecodeManifest(cache::EncodeManifest(a.header.manifest));
}
CheckpointTransferReader Reader(const std::vector<std::uint8_t>& wire) {
  return [&, offset = std::size_t{0}](std::span<std::uint8_t> out) mutable {
    const auto n =
        std::min({out.size(), wire.size() - offset, std::size_t{137}});
    std::memcpy(out.data(), wire.data() + offset, n);
    offset += n;
    return n;
  };
}
std::vector<std::uint8_t> Wire(const Artifact& a) {
  // Export recomputes the real per-payload CRC/SHA and envelope SHA; Import
  // must independently verify all of them before the serving semantic check.
  auto blob = CheckpointTransferBlob::Export(
      a.header, [&](auto i, const auto& consume) { consume(a.images.at(i)); });
  std::vector<std::uint8_t> wire;
  blob->Write([&](auto bytes) {
    wire.insert(wire.end(), bytes.begin(), bytes.end());
    return true;
  });
  auto verified = CheckpointTransferBlob::Import(
      Reader(wire), wire.size(), a.header.identity, kConfiguration);
  assert(verified->Header().manifest.compatibility ==
         Digest(a.header.identity));
  const auto payloads = CheckpointTransferPayloads(verified->Header().manifest);
  assert(payloads.size() == a.images.size());
  for (std::size_t i = 0; i < payloads.size(); ++i) {
    std::vector<std::uint8_t> bytes(a.images[i].size());
    verified->Load(i, bytes);
    assert(bytes == a.images[i]);
    assert(payloads[i]->checksum == cache::DiskChecksum(bytes));
  }
  return wire;
}
void Publish(const std::filesystem::path& root, const Artifact& a,
             std::uint8_t publication = 0) {
  cache::ResourceLedger ledger({kRam, kRam, 0, 0});
  cache::DiskStore disk(ledger, root, kRam);
  const auto payloads = CheckpointTransferPayloads(a.header.manifest);
  std::vector<cache::DiskWriteBuffer> buffers;
  std::size_t next = 0;
  for (const auto& c : a.header.manifest.components) {
    if (c.tail) {
      buffers.push_back({true, c.tail->file, a.images[next]});
      ++next;
    }
    if (c.private_state) {
      buffers.push_back({true, c.private_state->file, a.images[next]});
      ++next;
    }
  }
  assert(next == payloads.size());
  cache::DiskFileId manifest{};
  manifest[0] = 255;
  manifest[1] = publication;
  disk.Publish(manifest, a.header.manifest, buffers);
}
#if 1  // Both packed modes are unconditional in the cleaned q8 topic.
void RotatedWithoutFingerprintDisk() {
  const auto type = qfn::KvCacheType::kQ8_0Rotated;
  const auto valid = Canonical(type);
  auto bad = valid;
  Poison(bad, 0, 0);
  const TemporaryDirectory temp("gufo-rotated-process-local-disk");
  Publish(temp.path(), bad);
  auto runner = std::make_shared<AdmissionRunner>(type);
  runner->fingerprint = false;
  ComponentTextCache restarted(
      runner, 0, kRam,
      TextRunnerDiskCacheOptions{.directory = temp.path(),
                                 .capacity_bytes = kRam});
  if (restarted.Lookup(valid.header.manifest.tokens, nullptr, true, 0)
          ->matched_tokens != 0)
    throw std::runtime_error(
        "fingerprint-absent rotated disk published semantically malformed KV");
  assert(runner->allocations > 0);
}
void RotatedWithoutFingerprintImport() {
  const auto type = qfn::KvCacheType::kQ8_0Rotated;
  auto bad = Canonical(type);
  Poison(bad, 0, 0);
  const auto wire = Wire(bad);  // Real envelope/payload SHA and CRC all pass.
  auto runner = std::make_shared<AdmissionRunner>(type);
  runner->fingerprint = false;
  ComponentTextCache target(runner, 0, kRam, {});
  bool semantic = false;
  try {
    (void)target.Import(Reader(wire), wire.size(), kConfiguration);
  } catch (const std::invalid_argument& e) {
    semantic =
        std::string_view(e.what()) == "invalid Flash-Next KV reconstruction";
  } catch (const CheckpointTransferError& e) {
    throw std::runtime_error(
        "fingerprint-absent rotated semantic rejection misclassified as " +
        std::to_string(e.status) + ": " + e.code);
  }
  assert(semantic && runner->allocations > 0);
  assert(target.Lookup(bad.header.manifest.tokens, nullptr, true, 0)
             ->matched_tokens == 0);
}
#if defined(__linux__)
void RotatedExistingPinsBeforeReady() {
  const auto type = qfn::KvCacheType::kQ8_0Rotated;
  auto valid = Canonical(type);
  const auto suffix = platform::StrictRuntimeFingerprint(
      "gufo-registered-prefix-v1;prefill-chunk=4096");
  valid.header.identity.insert(valid.header.identity.end(), suffix.begin(),
                               suffix.end());
  valid.header.manifest.compatibility = Digest(valid.header.identity);
  for (bool lazy : {false, true}) {
    for (unsigned part = 0; part < 4; ++part) {
      for (unsigned fault : {0U, 1U}) {
        const TemporaryDirectory temp("gufo-rotated-packed-pin-admission");
        const auto file = temp.path() / "prefix.txt";
        {
          std::ofstream rendered(file);
          rendered << "CPU fixture rendered prefix";
        }
        TextRunnerPinnedPrefixOptions pins{.files = {file},
                                           .directory = temp.path() / "pins",
                                           .capacity_bytes = kRam,
                                           .lazy = lazy};
        auto bad = valid;
        Poison(bad, part, fault);
        Publish(pins.directory, bad);
        auto runner = std::make_shared<AdmissionRunner>(type);
        bool semantic = false;
        try {
          ComponentTextCache cache(runner, 0, kRam, {}, false, 0, pins);
        } catch (const std::invalid_argument& e) {
          semantic = std::string_view(e.what()) ==
                     (fault == 0 ? "invalid Flash-Next KV reconstruction"
                                 : "invalid Flash-Next KV scale");
        }
        // Zero slots and the layout-only adapter prohibit native restore.
        // This exact codec error must precede readiness for eager AND lazy
        // pins.
        assert(semantic && runner->allocations > runner->finishes &&
               runner->finishes > 0);
        if (part == 0 && fault == 0) {
          runner = std::make_shared<AdmissionRunner>(type);
          runner->fingerprint = false;
          bool identity_required = false;
          try {
            ComponentTextCache cache(runner, 0, kRam, {}, false, 0, pins);
          } catch (const std::invalid_argument& e) {
            identity_required =
                std::string_view(e.what()) ==
                "pinned prefix requires a durable model identity";
          }
          // Honest negative contract: the constructor rejects BEFORE opening
          // the pin store/preflight hook. This is NOT absent-fingerprint pin
          // predicate coverage and must not authorize fingerprint-free pins.
          assert(identity_required && runner->allocations == 0 &&
                 runner->finishes == 0);
        }
        {
          cache::ResourceLedger ledger({kRam, kRam, 0, 0});
          cache::DiskStore disk(ledger, pins.directory, kRam);
          assert(disk.Entries().size() == 1);
          assert(cache::EncodeManifest(disk.Entries().front().Manifest()) ==
                 cache::EncodeManifest(bad.header.manifest));
          // Bad registered roots are preserved, not silently retired/replaced.
          // Explicit fixture repair, then a valid lazy readiness retry.
          assert(disk.Retire(bad.header.manifest.checkpoint));
        }
        Publish(pins.directory, valid);
        pins.lazy = true;
        runner = std::make_shared<AdmissionRunner>(type);
        ComponentTextCache repaired(runner, 0, kRam, {}, false, 0, pins);
        assert(runner->allocations > 0 && runner->finishes > 0);
        assert(repaired.Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                   ->matched_tokens == 2);
      }
    }
  }
  std::cout
      << "CPU rotated pins: 16 eager/lazy poison refusals before ready, "
         "16 valid lazy repairs; 2 absent-fingerprint constructor refusals "
         "BEFORE pin ingress (hook not exercised); native NOT RUN\n";
}
#endif
#endif
void NotExportable(ComponentTextCache& target, const Artifact& source) {
  bool absent = false;
  try {
    (void)target.Export(source.header.manifest.tokens, nullptr, true, 0,
                        kConfiguration);
  } catch (const CheckpointTransferError& error) {
    absent = error.status == 404;
  }
  assert(absent && "rejected checkpoint must not be exportable");
}

void MalformedImportsAndDisk(qfn::KvCacheType selected_type,
                             bool selected_fingerprint) {
  const std::array types{
      kPlain,
#if 1  // Both packed modes are unconditional in the cleaned q8 topic.
      qfn::KvCacheType::kQ8_0Rotated,
#endif
  };
  for (auto type : types) {
    for (bool fingerprint : {true, false}) {
      // The four hosted suites partition the full dtype/fingerprint matrix.
      if (type != selected_type || fingerprint != selected_fingerprint)
        continue;
      const auto valid = Canonical(type);
      const auto valid_wire = Wire(valid);
      auto survivor = valid;
      survivor.header.manifest.tokens = {20, 21};
      const auto survivor_wire = Wire(survivor);
      {
        auto runner = std::make_shared<AdmissionRunner>(type);
        runner->fingerprint = fingerprint;
        ComponentTextCache target(runner, 0, kRam, {});
        auto corrupt = valid_wire;
        corrupt.back() ^= 1;
        bool integrity_rejected = false;
        try {
          (void)target.Import(Reader(corrupt), corrupt.size(), kConfiguration);
        } catch (const CheckpointTransferError& e) {
          integrity_rejected = e.status == 400;
        }
        assert(integrity_rejected && runner->allocations == 0);
      }
      for (unsigned part = 0; part < 4; ++part) {
        for (unsigned fault = 0; fault < 6; ++fault) {
          auto bad = valid;
          Poison(bad, part, fault);
          std::string codec_error;
          try {
            qfn::ValidateKvRow(
                std::as_bytes(std::span(bad.images[bad.kv_images[part]]))
                    .last(qfn::KvRowBytes(kElements, type)),
                kElements, type);
          } catch (const std::invalid_argument& error) {
            codec_error = error.what();
          }
          assert(!codec_error.empty());
          const auto wire = Wire(bad);
          auto runner = std::make_shared<AdmissionRunner>(type);
          runner->fingerprint = fingerprint;
          ComponentTextCache target(runner, 0, kRam, {});
          (void)target.Import(Reader(survivor_wire), survivor_wire.size(),
                              kConfiguration);
          const auto before = runner->ledger->Snapshot();
          bool semantic = false;
          try {
            (void)target.Import(Reader(wire), wire.size(), kConfiguration);
          } catch (const std::invalid_argument& error) {
            semantic = codec_error == error.what();
          }
          assert(semantic && "digest-valid malformed q8 import published");
          const auto after = runner->ledger->Snapshot();
          assert(before.bytes == after.bytes &&
                 before.reserved_bytes == after.reserved_bytes);
          assert(
              target.Lookup(survivor.header.manifest.tokens, nullptr, true, 0)
                  ->matched_tokens == 2);
          assert(target.Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                     ->matched_tokens == 0);
          NotExportable(target, valid);
          (void)target.Import(Reader(valid_wire), valid_wire.size(),
                              kConfiguration);
          assert(target.Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                     ->matched_tokens == 2);
          assert(runner->finishes > 0);

          const TemporaryDirectory temp("gufo-packed-semantic-disk");
          const TextRunnerDiskCacheOptions disk{.directory = temp.path(),
                                                .capacity_bytes = kRam};
          Publish(temp.path(), bad);
          {
            auto disk_runner = std::make_shared<AdmissionRunner>(type);
            disk_runner->fingerprint = fingerprint;
            ComponentTextCache restarted(disk_runner, 0, kRam, disk);
            assert(disk_runner->allocations > 0);
            assert(restarted
                       .Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                       ->matched_tokens == 0);
            NotExportable(restarted, valid);
          }
          Publish(temp.path(), valid);  // Same-root repair/retry.
          {
            auto disk_runner = std::make_shared<AdmissionRunner>(type);
            disk_runner->fingerprint = fingerprint;
            ComponentTextCache restarted(disk_runner, 0, kRam, disk);
            assert(disk_runner->finishes > 0);
            assert(restarted
                       .Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                       ->matched_tokens == 2);
          }
        }
      }
    }
  }
}
// Recovered old-main intent: a digest-valid envelope with wrong identity or
// component layout must fail before publication, not merely fail at restore.
// These are current-format mutations, NOT archived native-model captures.
void IdentityAndImportLayouts() {
  for (auto type : {kPlain, qfn::KvCacheType::kQ8_0Rotated}) {
    const auto valid = Canonical(type);
    const auto valid_wire = Wire(valid);
    auto survivor = valid;
    survivor.header.manifest.tokens = {20, 21};
    const auto survivor_wire = Wire(survivor);
    std::vector<Artifact> identities;
    for (auto other :
         {qfn::KvCacheType::kF16, kPlain, qfn::KvCacheType::kQ8_0Rotated}) {
      if (other == type)
        continue;
      auto bad = valid;
      bad.header.identity = Identity(other);
      identities.push_back(std::move(bad));
    }
    for (auto epoch : {22U, 23U, 24U, 25U, 26U}) {
      auto bad = valid;
      auto text =
          std::string(bad.header.identity.begin(), bad.header.identity.end());
      for (auto prefix : {"numerical_kernel_epoch=",
                          "numerical_build_lane=rocm-qfn-q8-wmma-v"}) {
        const auto current =
            std::string(prefix) +
            std::to_string(qfn::ExpectedKvPayloadVersion(type)) + "\n";
        const auto at = text.find(current);
        assert(at != std::string::npos &&
               epoch != qfn::ExpectedKvPayloadVersion(type));
        text.replace(at, current.size(),
                     std::string(prefix) + std::to_string(epoch) + "\n");
      }
      bad.header.identity.assign(text.begin(), text.end());
      identities.push_back(std::move(bad));
    }
    for (auto& bad : identities) {
      assert(bad.header.identity != valid.header.identity);
      bad.header.manifest.compatibility = Digest(bad.header.identity);
    }
    auto cases = identities;
    for (const auto image : valid.kv_images) {
      // Map each packed payload back to its manifest component, including
      // distinct ordinary/draft K/V. Test both wrong layout versions.
      const auto payloads = CheckpointTransferPayloads(valid.header.manifest);
      const auto file = payloads.at(image)->file;
      for (auto layout : {1U, type == kPlain ? 3U : 2U}) {
        auto bad = valid;
        bool found = false;
        for (auto& component : bad.header.manifest.components) {
          if (component.tail && component.tail->file == file) {
            assert(component.descriptor.layout_version != layout);
            component.descriptor.layout_version = layout;
            found = true;
          }
        }
        assert(found);
        cases.push_back(std::move(bad));
      }
    }
    for (const auto& bad : cases) {
      const auto wire =
          Wire(bad);  // Independently verifies transport with its own identity.
      auto runner = std::make_shared<AdmissionRunner>(type);
      ComponentTextCache target(runner, 0, kRam, {});
      (void)target.Import(Reader(survivor_wire), survivor_wire.size(),
                          kConfiguration);
      const auto before = runner->ledger->Snapshot();
      const auto allocations = runner->allocations, finishes = runner->finishes;
      bool rejected = false;
      try {
        (void)target.Import(Reader(wire), wire.size(), kConfiguration);
      } catch (const CheckpointTransferError& error) {
        rejected = error.status == 409;
      }
      const auto after = runner->ledger->Snapshot();
      assert(rejected && allocations == runner->allocations &&
             finishes == runner->finishes);
      assert(before.bytes == after.bytes &&
             before.reserved_bytes == after.reserved_bytes);
      assert(target.Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                 ->matched_tokens == 0);
      NotExportable(target, valid);
      assert(target.Lookup(survivor.header.manifest.tokens, nullptr, true, 0)
                 ->matched_tokens == 2);
      (void)target.Export(survivor.header.manifest.tokens, nullptr, true, 0,
                          kConfiguration);
      (void)target.Import(Reader(valid_wire), valid_wire.size(),
                          kConfiguration);
      assert(target.Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                 ->matched_tokens == 2);
      (void)target.Export(valid.header.manifest.tokens, nullptr, true, 0,
                          kConfiguration);
    }
    for (const auto& bad : identities) {
      const TemporaryDirectory temp("gufo-packed-identity-restart");
      const TextRunnerDiskCacheOptions disk{.directory = temp.path(),
                                            .capacity_bytes = kRam};
      std::cerr << "identity restart type=" << qfn::KvCacheTypeName(type)
                << " case=" << (&bad - identities.data()) << " publish bad\n";
      Publish(temp.path(), bad);
      {
        auto runner = std::make_shared<AdmissionRunner>(type);
        ComponentTextCache restarted(runner, 0, kRam, disk);
        assert(restarted.Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                   ->matched_tokens == 0);
        std::cerr << "identity restart check non-exportable\n";
        NotExportable(restarted, valid);
      }
      // Immutable disk publications cannot reuse the alien manifest/file IDs.
      // Like old-main's Rekey control, retain the bad entry and publish a
      // distinct valid checkpoint in the SAME root, then reopen both.
      auto recovery = valid;
      recovery.header.manifest.checkpoint = {2000};
      recovery.header.manifest.lineage = {2000};
      for (auto& component : recovery.header.manifest.components) {
        if (component.tail)
          component.tail->file[1] = 1;
        if (component.private_state)
          component.private_state->file[1] = 1;
      }
      Publish(temp.path(), recovery, 1);
      auto runner = std::make_shared<AdmissionRunner>(type);
      ComponentTextCache repaired(runner, 0, kRam, disk);
      assert(repaired.Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                 ->matched_tokens == 2);
      (void)repaired.Export(valid.header.manifest.tokens, nullptr, true, 0,
                            kConfiguration);
    }
  }
}

void IncompatibleDiskLayouts() {
  const std::array types{kPlain, qfn::KvCacheType::kQ8_0Rotated};
  for (auto type : types) {
    // Every registered append descriptor is compared, even zero-row pooled
    // components that the packed KV classifier correctly leaves unwrapped.
    const auto valid = Canonical(type);
    for (std::size_t i = 0; i < valid.header.manifest.components.size(); ++i) {
      if (valid.header.manifest.components[i].descriptor.kind !=
          cache::ComponentKind::kAppendRows)
        continue;
      auto mismatch = valid;
      ++mismatch.header.manifest.components[i].descriptor.layout_version;
      const TemporaryDirectory temp("gufo-packed-layout-mismatch");
      Publish(temp.path(), mismatch);
      auto runner = std::make_shared<AdmissionRunner>(type);
      runner->fingerprint = false;
      ComponentTextCache restarted(
          runner, 0, kRam,
          TextRunnerDiskCacheOptions{.directory = temp.path(),
                                     .capacity_bytes = kRam});
      assert(restarted.Lookup(valid.header.manifest.tokens, nullptr, true, 0)
                 ->matched_tokens == 0);
    }
    for (bool fingerprint : {true, false}) {
      // Equal numeric component layouts, even with a matching fingerprint
      // version, must not activate QFN admission for a foreign ABI.
      const TemporaryDirectory foreign("gufo-packed-foreign-abi");
      auto bad = valid;
      Poison(bad, 0, 0);
      Publish(foreign.path(), bad);
      auto runner = std::make_shared<AdmissionRunner>(type);
      runner->foreign_abi = true;
      runner->fingerprint = fingerprint;
      ComponentTextCache cache(
          runner, 0, kRam,
          TextRunnerDiskCacheOptions{.directory = foreign.path(),
                                     .capacity_bytes = kRam});
      assert(runner->allocations == 0);
      assert(cache.Lookup(bad.header.manifest.tokens, nullptr, true, 0)
                 ->matched_tokens == 2);
    }
  }
  // F16 layout1 remains on the original disk startup path, not packed q8
  // admission. No claim of executable/valid restored F16 state follows.
  for (bool fingerprint : {true, false}) {
    const TemporaryDirectory temp("gufo-f16-no-packed-admission");
    auto f16 = Canonical(qfn::KvCacheType::kF16);
    Poison(f16, 0, 3);  // Includes nonfinite F16 bytes; never restore them.
    Publish(temp.path(), f16);
    auto runner = std::make_shared<AdmissionRunner>(qfn::KvCacheType::kF16);
    runner->fingerprint = fingerprint;
    ComponentTextCache cache(
        runner, 0, kRam,
        TextRunnerDiskCacheOptions{.directory = temp.path(),
                                   .capacity_bytes = kRam});
    assert(runner->allocations == 0);
    assert(cache.Lookup(f16.header.manifest.tokens, nullptr, true, 0)
               ->matched_tokens == 2);
  }
}
}  // namespace packed_transfer_cpu
