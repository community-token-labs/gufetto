#ifndef GUFO_CORE_DIAGNOSTICS_MODEL_PLAN_H_
#define GUFO_CORE_DIAGNOSTICS_MODEL_PLAN_H_

#include <cstddef>
#include <filesystem>
#include <string>

namespace gufo::diagnostics {

struct ModelPlanReport {
  std::string text;
  bool has_failures = false;
};

/// Read-only planning: no network, allocation probes, or artifact writes.
/// Artifact paths are relative to the manifest's parent directory.
/// Throws for malformed TSV; unavailable budgets are reported per row.
[[nodiscard]] ModelPlanReport PlanModels(const std::filesystem::path& manifest);
[[nodiscard]] std::string DescribeModelLowerBound(
    const std::filesystem::path& model, std::size_t context);

}  // namespace gufo::diagnostics
#endif
