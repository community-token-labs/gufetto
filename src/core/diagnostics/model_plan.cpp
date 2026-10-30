#include "src/core/diagnostics/model_plan.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/hip/heap_placement.hpp"
#include "src/core/memory_budget.hpp"
#include "src/models/qwen/continuation_layout.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::diagnostics {
namespace {

std::vector<std::string> Fields(std::string_view line) {
  std::vector<std::string> fields;
  for (;;) {
    const auto tab = line.find('\t');
    fields.emplace_back(line.substr(0, tab));
    if (tab == std::string_view::npos)
      return fields;
    line.remove_prefix(tab + 1);
  }
}

bool Hex(std::string_view text, std::size_t length) {
  return text.size() == length &&
         std::all_of(text.begin(), text.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

std::size_t Bytes(std::string_view text) {
  std::size_t value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || result.ec != std::errc{} ||
      result.ptr != text.data() + text.size())
    throw std::invalid_argument("expected unsigned byte count");
  return value;
}

std::string Verify(const std::filesystem::path& path, std::size_t bytes,
                   const std::string& digest) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory ||
      status.type() == std::filesystem::file_type::not_found)
    return "MISSING";
  if (error)
    return "ERROR: " + error.message();
  if (!std::filesystem::is_regular_file(status))
    return "INVALID: not a regular file (symlinks are not followed)";
  const auto size = std::filesystem::file_size(path, error);
  if (error)
    return "ERROR: " + error.message();
  // Hash even size-mismatched existing files, so both checks are visible.
  try {
    const bool checksum = crypto::Sha256FileHex(path) == digest;
    return std::string(size == bytes ? "size=OK" : "size=MISMATCH") +
           (checksum ? " sha256=OK" : " sha256=MISMATCH");
  } catch (const std::exception& e) {
    return "ERROR: " + std::string(e.what());
  }
}

struct LowerBound {
  std::size_t weights{}, kv{};
};

std::optional<LowerBound> LocalEvidence(const std::filesystem::path& path,
                                        std::size_t context,
                                        std::string& error) {
  const auto reader = core::GgufReader::OpenFile(path, &error);
  if (!reader)
    return std::nullopt;
  if (reader->GetMappedRegions().size() != 1) {
    error =
        "split GGUF: only this row's shard is verified; sibling shards are not "
        "evaluated";
    return std::nullopt;
  }
  // This loader copies the complete mapped weight regions unchanged. Do not
  // apply its tensor-size lower bound to families with repacking/disk tables.
  if (reader->GetMetadataString("general.architecture") != "qwen35") {
    error = "unsupported metadata-only family (only qwen35 is evaluated)";
    return std::nullopt;
  }
  const auto weights = models::QwenModelWeights::LoadFromGguf(*reader, &error);
  if (!weights)
    return std::nullopt;
  if (context == 0 || context > weights->config.context_length) {
    error = "missing context or context exceeds model metadata";
    return std::nullopt;
  }
  LowerBound bound;
  for (const auto& region : reader->GetMappedRegions())
    bound.weights = core::AddMemoryBytes(bound.weights, region.size);
  const models::qwen::ContinuationLayout layout(
      weights->config, {}, static_cast<std::uint32_t>(context));
  // The constructor checks complete context*row arithmetic for overflow.
  // Private continuation includes publication/frontier copies, which are NOT
  // a lower bound on live session state; omit it rather than overclaim.
  bound.kv = layout.BytesPerToken() * context;
  return bound;
}

}  // namespace

std::string DescribeModelLowerBound(const std::filesystem::path& model,
                                    std::size_t context) {
  std::string error;
  const auto bound = LocalEvidence(model, context, error);
  if (!bound)
    return "metadata lower bound unavailable: " + error;
  return "one-session lower bound: weights_bytes=" +
         std::to_string(bound->weights) +
         " kv_bytes=" + std::to_string(bound->kv) +
         "; not counted: "
         "vision/draft/private-state/arena/workspace/cache/persistence";
}

ModelPlanReport PlanModels(const std::filesystem::path& manifest) {
  std::ifstream input(manifest, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open plan manifest");
  const std::string pinned{std::istreambuf_iterator<char>(input), {}};
  if (input.bad())
    throw std::runtime_error("cannot read plan manifest");
  std::istringstream lines(pinned);
  std::string line;
  if (!std::getline(lines, line))
    throw std::invalid_argument("empty plan manifest");
  if (line.ends_with('\r'))
    line.pop_back();
  const auto header = Fields(line);
  std::map<std::string, std::size_t> columns;
  for (std::size_t i = 0; i < header.size(); ++i) {
    if (header[i].empty() || !columns.emplace(header[i], i).second)
      throw std::invalid_argument("empty or duplicate manifest column");
  }
  constexpr std::array required = {
      "group",   "status", "repo", "revision", "file", "bytes", "sha256",
      "license", "gated",  "role", "why",      "fit",  "source"};
  for (const auto* column : required) {
    if (!columns.contains(column))
      throw std::invalid_argument("missing manifest column: " +
                                  std::string(column));
  }
  constexpr std::array optional = {"context",
                                   "host_available_bytes",
                                   "device_available_bytes",
                                   "os_reserve_bytes",
                                   "weights_bytes",
                                   "kv_bytes",
                                   "vision_bytes",
                                   "draft_bytes",
                                   "workspace_bytes",
                                   "cache_bytes",
                                   "persistence_staging_bytes"};
  for (const auto& name : header) {
    if (std::find(required.begin(), required.end(), name) == required.end() &&
        std::find(optional.begin(), optional.end(), name) == optional.end())
      throw std::invalid_argument("unknown manifest column: " + name);
  }
  std::ostringstream out;
  out << "Model setup plan (dry run; no downloads)\n"
      << "No FIT state: calculated lower bounds are not runtime "
         "qualification.\n"
      << "UNPROVEN needs gufo serve measured startup admission (verified to "
         "fit).\n"
      << "Each row is independent. Weights prefer the device heap, with host "
         "registration fallback; KV needs the device heap. No shared/carve-out "
         "topology is assumed.\n"
      << "Reserve providers: core::kHostExecutionReserveBytes / "
         "ObserveHostSnapshotHeadroom; hip::kDispatchReserve / "
         "ObserveExecutionCapacity. Available capacity inputs are declared, "
         "not host measurements.\n"
      << "Manifest SHA256: "
      << crypto::Sha256Hex(
             std::span(reinterpret_cast<const std::uint8_t*>(pinned.data()),
                       pinned.size()))
      << "\n";
  std::size_t row = 1;
  bool any = false;
  bool has_failures = false;
  while (std::getline(lines, line)) {
    ++row;
    if (line.ends_with('\r'))
      line.pop_back();
    if (line.empty())
      continue;
    const auto fields = Fields(line);
    if (fields.size() != header.size())
      throw std::invalid_argument("manifest row " + std::to_string(row) +
                                  ": column count mismatch");
    const auto get = [&](const char* name) -> const std::string& {
      return fields[columns.at(name)];
    };
    any = true;
    out << "Row " << row << " " << get("group") << " / " << get("file") << ": ";
    for (const auto* pin : {"revision", "sha256"}) {
      if (!get("status").starts_with("watch-only") && get(pin) != "unknown" &&
          !Hex(get(pin), std::string_view(pin) == "revision" ? 40 : 64))
        throw std::invalid_argument("row " + std::to_string(row) + " column " +
                                    pin + ": malformed immutable pin");
    }
    if (get("status").starts_with("watch-only") ||
        get("revision") == "unknown" || get("sha256") == "unknown" ||
        get("bytes") == "unknown") {
      out << "NOT-EVALUATED: missing immutable revision/size/SHA256 pin; "
             "local=NOT CHECKED\n";
      continue;
    }
    const std::filesystem::path relative(get("file"));
    if (relative.empty() || relative.is_absolute() ||
        relative.has_root_name() ||
        std::any_of(relative.begin(), relative.end(),
                    [](const auto& part) { return part == ".."; }) ||
        get("file").find('\\') != std::string::npos)
      throw std::invalid_argument("unsafe artifact path on row " +
                                  std::to_string(row));
    std::size_t bytes = 0;
    try {
      bytes = Bytes(get("bytes"));
    } catch (const std::exception&) {
      throw std::invalid_argument("invalid bytes on row " +
                                  std::to_string(row));
    }
    const auto present = [&](const char* name) {
      return columns.contains(name) && !get(name).empty() &&
             get(name) != "unknown";
    };
    const auto number = [&](const char* name) {
      try {
        return Bytes(get(name));
      } catch (const std::exception&) {
        throw std::invalid_argument("row " + std::to_string(row) + " column " +
                                    name + ": expected unsigned integer");
      }
    };
    // Validate even unused optional inputs, so a typo cannot silently suppress
    // evidence.
    for (const auto* name : optional) {
      if (present(name))
        (void)number(name);
    }
    if (present("os_reserve_bytes") &&
        number("os_reserve_bytes") < core::kHostExecutionReserveBytes)
      throw std::invalid_argument(
          "row " + std::to_string(row) +
          " column os_reserve_bytes: below server reserve");
    const auto path = manifest.parent_path() / relative;
    const auto local = Verify(path, bytes, get("sha256"));
    const auto context = present("context") ? number("context") : 0;
    const bool capacity_known = present("host_available_bytes") &&
                                present("device_available_bytes") &&
                                context > 0;
    std::optional<LowerBound> bound;
    std::size_t extras = 0;
    std::string evidence;
    if (local == "size=OK sha256=OK") {
      bound = LocalEvidence(path, context, evidence);
      if (bound)
        evidence =
            "derived weights=mapped-file-regions "
            "KV=Qwen-F16-continuation-rows; "
            "not counted: mmproj/vision MTP/draft recurrent/private-state "
            "arena/executor/workspace/session cache persistence";
    } else if (local != "MISSING") {
      evidence = "local artifact failed verification";
      has_failures = true;
    } else {
      constexpr std::array declared = {"weights_bytes",
                                       "kv_bytes",
                                       "vision_bytes",
                                       "draft_bytes",
                                       "workspace_bytes",
                                       "cache_bytes",
                                       "persistence_staging_bytes"};
      if (std::all_of(declared.begin(), declared.end(), present)) {
        bound = LowerBound{number("weights_bytes"), number("kv_bytes")};
        for (const char* name :
             {"vision_bytes", "draft_bytes", "workspace_bytes", "cache_bytes",
              "persistence_staging_bytes"})
          extras = core::AddMemoryBytes(extras, number(name));
        evidence =
            "declared (not metadata-derived); not counted: any "
            "undeclared native allocator/session overhead";
        for (const auto* name : declared)
          evidence += " " + std::string(name) + "=" + get(name);
      } else {
        evidence =
            "insufficient admission evidence: missing declared byte budgets";
      }
    }
    if (!bound) {
      out << (local == "size=OK sha256=OK" ? "NOT-EVALUATED" : "UNPROVEN")
          << ": " << evidence;
    } else if (!capacity_known) {
      out << "UNPROVEN: insufficient admission evidence: missing context or "
             "declared host/device available capacity; lower_bound_bytes="
          << core::AddMemoryBytes(
                 core::AddMemoryBytes(bound->weights, bound->kv), extras)
          << " weights_bytes=" << bound->weights << " kv_bytes=" << bound->kv
          << "; " << evidence;
    } else {
      const auto host = number("host_available_bytes");
      const auto device = number("device_available_bytes");
      const auto reserve = present("os_reserve_bytes")
                               ? number("os_reserve_bytes")
                               : core::kHostExecutionReserveBytes;
      const auto pending = core::AddMemoryBytes(
          core::AddMemoryBytes(bound->weights, bound->kv), extras);
      const auto heap =
          device > hip::kDispatchReserve ? device - hip::kDispatchReserve : 0;
      // model_loader.cpp:206-215 tries device weights before registered host.
      // Host reserve is post-loading execution/cache policy, NOT a loader gate:
      // use raw host capacity for this conservative necessary weight-load test.
      const bool kv_fits =
          core::FitsMemoryBudget(device, bound->kv, hip::kDispatchReserve);
      const bool either_placement =
          core::FitsMemoryBudget(heap, bound->weights, bound->kv) ||
          core::FitsMemoryBudget(host, bound->weights, 0);
      // Extras have unknown lanes. Use only the aggregate necessary condition;
      // subtract rather than adding host+heap, which could overflow.
      const bool aggregate =
          core::FitsMemoryBudget(host, pending > heap ? pending - heap : 0, 0);
      const bool within = kv_fits && either_placement && aggregate;
      has_failures = has_failures || !within;
      out << (within ? "UNPROVEN" : "DOES-NOT-FIT")
          << ": lower_bound_bytes=" << pending
          << " weights_bytes=" << bound->weights << " kv_bytes=" << bound->kv
          << " context=" << context << " declared_host_available=" << host
          << " declared_device_available=" << device
          << " post_load_OS_reserve=" << reserve
          << " (not a weight-load gate) device_reserve="
          << hip::kDispatchReserve << "; " << evidence;
      if (!within)
        out << "; lower bound alone exceeds reserved capacity";
    }
    out << "; local=" << local << "\n";
  }
  if (!any)
    throw std::invalid_argument("manifest contains no model rows");
  out << "\nPinned manifest (verbatim):\n" << pinned;
  return {.text = out.str(), .has_failures = has_failures};
}

}  // namespace gufo::diagnostics
