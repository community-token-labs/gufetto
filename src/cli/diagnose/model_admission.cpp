#include <iostream>
#include <stdexcept>
#include <string>

#include "src/cli/arg_parser.hpp"
#include "src/cli/diagnose/diagnose.h"
#include "src/cli/kv_cache_options.hpp"
#include "src/cli/serve/inference_backend.hpp"
#include "src/core/diagnostics/model_plan.h"
#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/architecture.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace gufo::cli {
int RunModelAdmission(std::span<const char* const> args) {
  bool plan = false, no_gpu = false, no_vision = false, yarn = false;
  std::string model, mmproj, mtp, speculative, ram = "0", disk = "off";
  std::uint32_t context = 0, draft = 7, prefill = 512, vision_max = 0,
                vision_pool = models::qwen::vision::kDefaultOutputPoolRows;
  std::size_t sessions = 1;
  auto kv = models::qwen38_flash_next::KvCacheType::kF16;
  ArgParser parser("gufo diagnose --plan --model",
                   "Allocate, measure, release; no scheduler or HTTP server.");
  parser.AddFlag("", "--plan", "Plan a local model", "Plan", &plan);
  parser.AddFlag("", "--no-gpu", "CPU lower bound only", "Plan", &no_gpu);
  parser.AddOption("", "--model", "PATH", "Local GGUF", "Model", &model);
  parser.AddOption("", "--context", "N", "Context (0 = native)", "Model",
                   &context);
  parser.AddOption("", "--sessions", "N", "Session count", "Model", &sessions);
  parser.AddOption("", "--mmproj", "PATH", "Vision sidecar", "Model", &mmproj);
  parser.AddFlag("", "--no-vision", "Disable vision", "Model", &no_vision);
  parser.AddFlag("", "--enable-yarn", "Static YaRN", "Model", &yarn);
  parser.AddOption("", "--vision-max-tokens-per-slot", "N", "Vision cap",
                   "Model", &vision_max);
  parser.AddOption("", "--vision-pool-tokens", "N", "Vision pool", "Model",
                   &vision_pool);
  parser.AddOption("", "--speculative", "MODE", "mtp or off", "Model",
                   &speculative);
  parser.AddOption("", "--mtp-model", "PATH", "MTP sidecar", "Model", &mtp);
  parser.AddOption("", "--draft-tokens", "N", "Draft bound", "Model", &draft);
  parser.AddOption("", "--prefill-chunk", "N", "Prefill policy", "Model",
                   &prefill);
  parser.AddOption("", "--cache-ram-bytes", "N|off", "RAM cache (0 = AUTO)",
                   "Model", &ram);
  parser.AddOption("", "--cache-disk", "off", "Only off is allowed", "Model",
                   &disk);
  RegisterKvCacheTypeOption(parser, &kv);
  std::string error;
  if (!parser.Parse(args, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 2;
  }
  if (parser.IsHelpRequested()) {
    parser.PrintHelp();
    return 0;
  }
  const auto cache = server::ParseRamCacheOption(ram, &error);
  if (!plan || model.empty() || !sessions || !draft || !prefill ||
      !vision_pool || (no_vision && !mmproj.empty()) || disk != "off" ||
      !cache ||
      (!speculative.empty() && speculative != "off" && speculative != "mtp") ||
      (speculative == "mtp" && mtp.empty())) {
    std::cerr
        << "Error: invalid model admission options; persistence must be off\n";
    return 2;
  }
  try {
    auto reader = core::GgufReader::OpenFile(model, &error);
    if (!reader)
      throw std::invalid_argument("cannot open model: " + error);
    const std::string architecture(
        reader->GetMetadataString("general.architecture").value_or(""));
    if (!ValidateKvCacheTypeForArchitecture(kv, architecture, &error))
      throw std::invalid_argument(error);
    if (!context)
      context =
          reader
              ->GetMetadataUint32(std::string(architecture) + ".context_length")
              .value_or(0);
    if (!context) {
      std::cout << "UNPROVEN: insufficient admission evidence: no native "
                   "context; specify --context\n";
      return 0;
    }
    if (context < 2 ||
        (yarn && !models::qwen::IsQwen4PreviewArch(architecture)))
      throw std::invalid_argument(
          "invalid native/requested context or --enable-yarn family");
    bool gpu = false;
#if defined(ENGINE_ENABLE_HIP)
    if (!no_gpu && models::qwen::IsQwen4PreviewArch(architecture)) {
      int devices = 0;
      const auto status = hipGetDeviceCount(&devices);
      if (status != hipSuccess && status != hipErrorNoDevice)
        throw std::runtime_error(hipGetErrorString(status));
      gpu = status == hipSuccess && devices > 0;
    }
#else
    (void)no_gpu;
#endif
    const auto identity = core::GgufIdentityHex(*reader);
    if (!gpu || !models::qwen::IsQwen4PreviewArch(architecture)) {
      std::cout << "UNPROVEN: native measured admission not run (no GPU, "
                   "disabled, or unsupported provider)\n"
                << diagnostics::DescribeModelLowerBound(model, context)
                << "\ncontext=" << context << " sessions=" << sessions
                << " gguf_identity_sha256=" << identity << "\n";
      return 0;
    }
    reader.reset();  // Release identity-scan mappings before loading native
                     // weights.
    bool accepted = false, refused = false;
    std::string result;
    {
      server::InferenceBackend backend(
          server::InferenceBackend::StartupMode::kAdmissionOnly);
      server::TextSpeculativeConfig spec;
      spec.backend = speculative == "mtp"
                         ? server::TextSpeculativeBackend::kMtp
                         : server::TextSpeculativeBackend::kDisabled;
      spec.draft_model_path = mtp;
      spec.max_draft_tokens = draft;
      accepted = backend.load(
          model, &error, context, sessions,
          server::TextPrefillPolicy{.decode_active_tokens = prefill}, {}, spec,
          {}, mmproj, *cache, yarn, kv, !no_vision, vision_max,
          parser.WasSupplied("--vision-pool-tokens")
              ? std::optional{vision_pool}
              : std::nullopt);
      refused = backend.measured_capacity_refused();
      if (backend.scheduler_started())
        throw std::logic_error(
            "admission-only unexpectedly started a scheduler");
      if (accepted)
        result = "FIT: verified to fit: --context " +
                 std::to_string(backend.max_context()) +
                 " sessions=" + std::to_string(sessions) +
                 " loaded_vision=" + (backend.supports_images() ? "on" : "off");
#if defined(ENGINE_ENABLE_HIP)
      if (hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("admission quiescence failed");
#endif
    }  // Pool, adapters, model bindings and weights die BEFORE reporting FIT.
#if defined(ENGINE_ENABLE_HIP)
    if (hipDeviceSynchronize() != hipSuccess || hipGetLastError() != hipSuccess)
      throw std::runtime_error("admission teardown failed");
#endif
    if (!accepted && !refused)
      throw std::runtime_error(error);
    std::cout << (accepted ? result : "DOES-NOT-FIT: " + error)
              << "\nrequested_kv_cache_type="
              << models::qwen38_flash_next::KvCacheTypeName(kv)
              << " speculative=" << (speculative == "mtp" ? "mtp" : "off")
              << "\ngguf_identity_sha256=" << identity
              << "\nMeasured startup only; not workload/quality qualification. "
                 "Resources released.\n";
    return accepted ? 0 : 1;
  } catch (const std::exception& exception) {
    std::cerr << "ERROR: " << exception.what() << '\n';
    return 2;
  }
}
}  // namespace gufo::cli
