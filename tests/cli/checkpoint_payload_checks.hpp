// Logical-identity fixture pattern inspired by LMCache #5944/#5952
// (58c2ace2, dce11e43); no upstream code or checksum conversion is copied.
#ifndef GUFO_TEST_CHECKPOINT_PAYLOAD_CHECKS_HPP_
#define GUFO_TEST_CHECKPOINT_PAYLOAD_CHECKS_HPP_

void CheckpointPayloadChecks() {
  // Rank is a fixture participant index encoded into a distinct component ID,
  // not a claim that the transfer API implements tensor-parallel ranks.
  const auto payload = [](std::uint32_t prefix, std::uint32_t component,
                          std::uint32_t rank, std::uint32_t first,
                          std::uint32_t bytes) {
    std::vector<std::uint8_t> result;
    for (auto offset = first; offset < first + bytes; offset += 16)
      for (const auto value : {prefix, component, rank, offset})
        for (unsigned byte = 0; byte < 4; ++byte)
          result.push_back(static_cast<std::uint8_t>(value >> (8 * byte)));
    return result;
  };
  for (const std::uint32_t prefix : {100U, 200U}) {
    CheckpointTransferHeader header;
    header.identity = {1, 20, 7};
    header.configuration = "deterministic-exact-byte-fixture-v1";
    crypto::Sha256Hasher identity_hash;
    identity_hash.Update(header.identity);
    header.manifest.compatibility = identity_hash.Finish();
    header.manifest.checkpoint = {17};
    header.manifest.lineage = {3};
    header.manifest.tokens = {prefix, prefix + 1, prefix + 2, prefix + 3,
                              prefix + 4};
    header.stable_tokens = 5;
    std::vector<std::vector<std::uint8_t>> expected;
    std::uint8_t file = 1;
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
      for (std::uint32_t component = 0; component < 2; ++component) {
        const cache::ComponentId id{1 + rank * 2 + component};
        header.manifest.components.push_back(
            {.descriptor = {id, 1, cache::ComponentKind::kAppendRows, 16, 2, 0},
             .position = {id, 5},
             .chunks = {{{file++}, 32, 0}, {{file++}, 32, 0}},
             .tail = cache::DiskPayload{{file++}, 16, 0}});
        expected.push_back(payload(prefix, component, rank, 0, 32));
        expected.push_back(payload(prefix, component, rank, 32, 32));
        expected.push_back(payload(prefix, component, rank, 64, 16));
      }
    }
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
      const cache::ComponentId id{5 + rank};
      header.manifest.components.push_back(
          {.descriptor = {id, 1, cache::ComponentKind::kPrivateState, 0, 0, 64},
           .position = {id, 5},
           .private_state = cache::DiskPayload{{file++}, 64, 0}});
      expected.push_back(payload(prefix, 2, rank, 0, 64));
    }
    const auto export_payloads = [&](auto h, const auto& data) {
      return CheckpointTransferBlob::Export(
          h, [&](auto i, const auto& consume) {
            // Deliberately split a logical record across producer writes.
            consume(std::span(data[i]).first(7));
            consume(std::span(data[i]).subspan(7));
          });
    };
    const auto matches = [&](const CheckpointTransferBlob& blob) {
      for (std::size_t i = 0; i < expected.size(); ++i) {
        std::vector<std::uint8_t> got(expected[i].size());
        blob.Load(i, got);
        if (got != expected[i] ||
            crypto::Sha256Hex(got) != crypto::Sha256Hex(expected[i]))
          return false;
        // Offset-addressed reads must preserve the exact bytes too.
        std::vector<std::uint8_t> suffix(got.size() - 7);
        blob.Read(i, 7, suffix);
        if (!std::ranges::equal(suffix, std::span(expected[i]).subspan(7)))
          return false;
      }
      return true;
    };
    auto source = export_payloads(header, expected);
    auto wire = Wire(*source);
    auto imported = CheckpointTransferBlob::Import(
        Reader(wire), wire.size(), header.identity, header.configuration);
    assert(matches(*imported));
    auto relocated = header;
    relocated.manifest.checkpoint = {99};
    relocated.manifest.lineage = {91};
    for (auto& component : relocated.manifest.components) {
      for (auto& chunk : component.chunks)
        chunk.file[0] += 100;
      if (component.tail)
        component.tail->file[0] += 100;
      if (component.private_state)
        component.private_state->file[0] += 100;
    }
    auto moved = export_payloads(relocated, expected);
    auto moved_wire = Wire(*moved);
    auto restored =
        CheckpointTransferBlob::Import(Reader(moved_wire), moved_wire.size(),
                                       header.identity, header.configuration);
    assert(matches(*restored));
    assert(restored->Header().epoch == imported->Header().epoch);
    assert(moved_wire !=
           wire);  // different physical handles, same logical bytes

    for (unsigned mutation = 0; mutation < 6; ++mutation) {
      auto wrong = expected;
      if (mutation == 0)
        std::swap(wrong[0], wrong[3]);  // component
      if (mutation == 1)
        std::swap(wrong[0], wrong[6]);  // rank
      if (mutation == 2)
        std::swap(wrong[0], wrong[1]);  // offset/chunk
      if (mutation == 3)
        wrong[0] = payload(prefix + 1, 0, 0, 0, 32);
      if (mutation == 4)
        std::swap(wrong[12], wrong[13]);  // private ranks
      if (mutation == 5)
        std::swap_ranges(wrong[12].begin(), wrong[12].begin() + 16,
                         wrong[12].begin() + 16);  // private offsets
      // A self-consistent but wrongly routed producer gets fresh checksums.
      // The independent logical oracle must still reject it after import.
      auto misrouted = export_payloads(header, wrong);
      auto wrong_wire = Wire(*misrouted);
      auto wrong_blob =
          CheckpointTransferBlob::Import(Reader(wrong_wire), wrong_wire.size(),
                                         header.identity, header.configuration);
      assert(!matches(*wrong_blob));
      // Without changing the original authenticated header, wire swaps must
      // be rejected by Gufo's own exact-byte SHA/CRC contract, before
      // admission.
      auto corrupt = wire;
      std::size_t offset = corrupt.size() - source->PayloadBytes();
      for (const auto& bytes : wrong) {
        std::copy(bytes.begin(), bytes.end(), corrupt.begin() + offset);
        offset += bytes.size();
      }
      Check(
          true,
          [&] {
            (void)CheckpointTransferBlob::Import(
                Reader(corrupt), corrupt.size(), header.identity,
                header.configuration);
          },
          400, "cache_transfer_invalid");
    }
  }
}

#endif  // GUFO_TEST_CHECKPOINT_PAYLOAD_CHECKS_HPP_
