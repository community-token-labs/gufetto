#ifndef GUFO_TEST_NATIVE_MEMORY_ACCOUNTING_HPP_
#define GUFO_TEST_NATIVE_MEMORY_ACCOUNTING_HPP_

// Linux/glibc-only investigative sampler, not production admission policy.
#include <malloc.h>

#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gufo::testing {
struct AnonymousMapping {
  std::uint64_t begin{}, end{}, anonymous{}, rss{};
  std::string path;
  bool direct{};
  std::uint64_t Size() const { return end - begin; }
};
inline std::uint64_t Number(const std::string& text, int base = 10) {
  std::uint64_t n{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), n, base);
  if (error != std::errc{} || end != text.data() + text.size())
    throw std::runtime_error("invalid accounting integer");
  return n;
}
inline unsigned MappingBucket(std::uint64_t size) {
  constexpr std::array<std::uint64_t, 4> upper{64ULL << 10, 1ULL << 20,
                                               16ULL << 20, 256ULL << 20};
  unsigned i = 0;
  while (i < upper.size() && size > upper[i])
    ++i;
  return i;
}
inline std::vector<AnonymousMapping> ParseSmaps(std::istream& input) {
  std::vector<AnonymousMapping> rows;
  AnonymousMapping current;
  bool active = false;
  unsigned fields = 0;
  std::uint64_t previous_end = 0;
  const auto finish = [&] {
    if (!active)
      return;
    if (fields != 7 || current.anonymous > current.rss ||
        current.rss > current.Size())
      throw std::runtime_error("incomplete/inconsistent smaps VMA");
    if (current.direct || current.anonymous)
      rows.push_back(current);
  };
  for (std::string line; std::getline(input, line);) {
    std::istringstream words(line);
    std::string key;
    if (!(words >> key))
      throw std::runtime_error("empty smaps line");
    const auto dash = key.find('-');
    if (dash != std::string::npos && key.find(':') == std::string::npos) {
      finish();
      current = {};
      current.begin = Number(key.substr(0, dash), 16);
      current.end = Number(key.substr(dash + 1), 16);
      if (current.end <= current.begin || current.begin < previous_end)
        throw std::runtime_error("invalid/overlapping smaps range");
      previous_end = current.end;
      std::string permissions, offset, device, inode;
      if (!(words >> permissions >> offset >> device >> inode) ||
          permissions.size() != 4 || device.find(':') == std::string::npos)
        throw std::runtime_error("invalid smaps header");
      std::getline(words >> std::ws, current.path);
      current.direct = current.path.empty() || current.path == "[heap]" ||
                       current.path == "[stack]" ||
                       current.path.starts_with("[stack:") ||
                       current.path.starts_with("[anon:") ||
                       current.path.starts_with("[anon_shmem:");
      if (current.path.empty())
        current.path = "[anon]";
      active = true;
      fields = 0;
    } else if (key == "Size:" || key == "Rss:" || key == "Anonymous:") {
      if (!active)
        throw std::runtime_error("smaps field before header");
      std::string value, unit, extra;
      if (!(words >> value >> unit) || unit != "kB" || (words >> extra))
        throw std::runtime_error("invalid smaps units");
      auto bytes = Number(value);
      if (bytes > std::numeric_limits<std::uint64_t>::max() / 1024)
        throw std::runtime_error("smaps byte overflow");
      bytes *= 1024;
      const unsigned flag = key == "Size:" ? 1 : key == "Rss:" ? 2 : 4;
      if (fields & flag)
        throw std::runtime_error("duplicate smaps field");
      fields |= flag;
      if (key == "Size:" && bytes != current.Size())
        throw std::runtime_error("smaps size/range mismatch");
      if (key == "Rss:")
        current.rss = bytes;
      if (key == "Anonymous:")
        current.anonymous = bytes;
    }
  }
  if (input.bad() || !active)
    throw std::runtime_error("unreadable/empty smaps");
  finish();
  return rows;
}
inline std::string JsonString(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : text) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
    else
      out << c;
  }
  out << '"';
  return out.str();
}
inline void PrintNativeAccounting(std::ostream& out, const char* phase,
                                  int cycle) {
  // Query before parser allocations; post-cycle workers/owners are gone.
  const auto malloc = mallinfo2();
  std::ifstream smaps("/proc/self/smaps");
  if (!smaps)
    throw std::runtime_error("cannot open /proc/self/smaps");
  const auto rows = ParseSmaps(smaps);
  const auto prefix = [&](const char* kind) {
    out << kind << "={\"phase\":" << JsonString(phase)
        << ",\"cycle\":" << cycle;
  };
  prefix("malloc_sample");
  out << ",\"arena\":" << malloc.arena << ",\"uordblks\":" << malloc.uordblks
      << ",\"fordblks\":" << malloc.fordblks << ",\"hblks\":" << malloc.hblks
      << ",\"hblkhd\":" << malloc.hblkhd << ",\"keepcost\":" << malloc.keepcost
      << "}\n";
  using Key = std::tuple<bool, unsigned, std::string>;
  struct Totals {
    std::uint64_t count{}, mapped{}, anonymous{};
  };
  std::map<Key, Totals> groups;
  for (const auto& row : rows) {
    prefix("smaps_vma");
    out << ",\"begin\":" << row.begin << ",\"end\":" << row.end
        << ",\"path\":" << JsonString(row.path)
        << ",\"direct\":" << (row.direct ? "true" : "false")
        << ",\"bucket\":" << MappingBucket(row.Size())
        << ",\"mapped_bytes\":" << row.Size()
        << ",\"anonymous_bytes\":" << row.anonymous << "}\n";
    auto& total = groups[{row.direct, MappingBucket(row.Size()), row.path}];
    // VMAs are nonoverlapping portions of the uint64 address space.
    ++total.count;
    total.mapped += row.Size();
    total.anonymous += row.anonymous;
  }
  for (const auto& [key, total] : groups) {
    const auto& [direct, bucket, path] = key;
    prefix("smaps_bucket");
    out << ",\"direct\":" << (direct ? "true" : "false")
        << ",\"bucket\":" << bucket << ",\"path\":" << JsonString(path)
        << ",\"count\":" << total.count << ",\"mapped_bytes\":" << total.mapped
        << ",\"anonymous_bytes\":" << total.anonymous << "}\n";
  }
}
}  // namespace gufo::testing
#endif
