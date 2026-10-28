// Included after the envelope test's Wire, Reader and Check helpers.
#ifndef GUFO_TEST_CHECKPOINT_IDENTITY_CHECKS_HPP_
#define GUFO_TEST_CHECKPOINT_IDENTITY_CHECKS_HPP_

void CheckpointIdentityChecks(
    CheckpointTransferHeader header,
    const std::vector<std::vector<std::uint8_t>>& payloads) {
  const TextRunnerDescriptor descriptor{
      .model_id = "fake-model",
      .state_abi = "fake-abi",
      .max_context = 4096,
      .persistence = TextRunnerPersistenceDescriptor{{1, 2, 3}, 21}};
  platform::CheckpointTransferRuntimeIdentity runtime{
      .executable_sha256 = "fake-executable-hash",
      .object_sha256 = {"fake-library-b-hash", "fake-library-a-hash"},
      .environment = {"GUFO_FLASH_NEXT_TARGET_PROBE=/private/source-probe"}};
  const std::vector<std::string> relevant{
      "GUFO_FUTURE_NUMERICS", "HIP_FUTURE_MATH",       "HSA_FUTURE_MATH",
      "ROCR_FUTURE_MATH",     "ROCBLAS_FUTURE_MATH",   "AMD_FUTURE_MATH",
      "GPU_MAX_HW_QUEUES",    "HIPBLASLT_FUTURE_MATH", "CUDA_FUTURE_MATH",
      "CUBLAS_FUTURE_MATH",   "NVIDIA_FUTURE_MATH"};
  for (const auto& name : relevant)
    runtime.environment.push_back(name + "=1");
  const auto config = [&](const auto& value) {
    return CheckpointTransferConfiguration(descriptor, "fake-image-hash",
                                           value);
  };
  header.configuration = config(runtime);
  const auto environment_field =
      header.configuration.find("environment_sha256=");
  assert(environment_field != std::string::npos &&
         environment_field ==
             header.configuration.rfind("environment_sha256="));
  assert(header.configuration.find("/private/") == std::string::npos);
  for (const auto& entry : runtime.environment)
    assert(header.configuration.find(entry) == std::string::npos);
  // Independently computed SHA-256 of sorted u64-LE-length-prefixed entries.
  assert(CheckpointTransferEnvironmentDigest(
             std::vector<std::string>{"GUFO_B=2", "GUFO_A=1"}) ==
         "27b847848c617be322650351eedd219b896bfcdaf988c517d4956f8f9679fc9a");
  assert(CheckpointTransferEnvironmentDigest(
             std::vector<std::string>{"GUFO_A=1\nGUFO_B=2"}) !=
         CheckpointTransferEnvironmentDigest(
             std::vector<std::string>{"GUFO_A=1", "GUFO_B=2"}));
  const auto export_with = [&](const std::string& configuration) {
    auto h = header;
    h.configuration = configuration;
    return CheckpointTransferBlob::Export(
        h, [&](auto i, const auto& consume) { consume(payloads[i]); });
  };
  const auto source = export_with(header.configuration);
  const auto wire = Wire(*source);
  const auto accepts = [&](const auto& value) {
    const auto configuration = config(value);
    assert(configuration == header.configuration);
    const auto restored = CheckpointTransferBlob::Import(
        Reader(wire), wire.size(), header.identity, configuration);
    assert(restored->Header().epoch == source->Header().epoch);
    assert(export_with(configuration)->Header().epoch ==
           source->Header().epoch);
  };
  const auto rejects = [&](const auto& value) {
    const auto configuration = config(value);
    assert(configuration != header.configuration);
    assert(export_with(configuration)->Header().epoch !=
           source->Header().epoch);
    Check(
        true,
        [&] {
          (void)CheckpointTransferBlob::Import(Reader(wire), wire.size(),
                                               header.identity, configuration);
        },
        409, "cache_transfer_epoch_mismatch");
  };
  auto changed = runtime;
  std::reverse(changed.environment.begin(), changed.environment.end());
  changed.environment.push_back(changed.environment.front());
  accepts(changed);  // order and duplicate entries do not change the block
  for (const auto& name : {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT",
                           "GUFO_PERF_TRACE", "GUFO_PERF_TRACE_MOE"}) {
    changed = runtime;
    changed.environment.push_back(std::string(name) +
                                  "=/private/bind-or-trace");
    accepts(changed);
  }
  for (const auto& name : relevant) {
    changed = runtime;
    *std::find(changed.environment.begin(), changed.environment.end(),
               name + "=1") = name + "=2";
    rejects(changed);
  }
  changed = runtime;
  changed.environment.front() = "GUFO_FLASH_NEXT_TARGET_PROBE=";
  accepts(changed);  // getenv != nullptr, including empty, means activated
  changed.environment.front() =
      "GUFO_FLASH_NEXT_TARGET_PROBE=/private/dest-probe";
  accepts(changed);
  changed.environment.erase(changed.environment.begin());
  rejects(changed);  // absent is not activated
  changed = runtime;
  changed.executable_sha256 = "different-executable";
  rejects(changed);
  changed = runtime;
  changed.object_sha256.insert("different-runtime-library");
  rejects(changed);
  auto different = descriptor;
  different.persistence->payload_version++;
  assert(CheckpointTransferConfiguration(different, "fake-image-hash",
                                         runtime) != header.configuration);
  different = descriptor;
  different.max_context++;
  assert(CheckpointTransferConfiguration(different, "fake-image-hash",
                                         runtime) != header.configuration);
  assert(CheckpointTransferConfiguration(descriptor, "other-image", runtime) !=
         header.configuration);
}

#endif  // GUFO_TEST_CHECKPOINT_IDENTITY_CHECKS_HPP_
