#include "tests/diagnostics/native_memory_accounting.hpp"

#include <cassert>
#include <iostream>

using namespace gufo::testing;
namespace {
std::string Vma(
    const std::string& path,
    const std::string& fields = "Size: 64 kB\nRss: 8 kB\nAnonymous: 4 kB\n") {
  return "10000-20000 rw-p 00000000 00:00 0 " + path + "\n" + fields;
}
void Reject(const std::string& text) {
  std::istringstream input(text);
  bool rejected = false;
  try {
    (void)ParseSmaps(input);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}
}  // namespace
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--sample") {
    PrintNativeAccounting(std::cout, "cpu_fixture", 0);
    return 0;
  }
  for (const std::string path :
       {"", "[heap]", "[stack]", "[anon:rocr]", "/tmp/a file with spaces"}) {
    std::istringstream input(Vma(path));
    const auto rows = ParseSmaps(input);
    assert(rows.size() == 1 && rows[0].anonymous == 4096 &&
           rows[0].Size() == 65536);
    assert(rows[0].path == (path.empty() ? "[anon]" : path));
    assert(rows[0].direct == !path.starts_with('/'));
  }
  for (const auto upper :
       {64ULL << 10, 1ULL << 20, 16ULL << 20, 256ULL << 20}) {
    assert(MappingBucket(upper + 1) == MappingBucket(upper) + 1);
    assert(MappingBucket(upper - 1) == MappingBucket(upper));
  }
  std::istringstream reserved(
      Vma("", "Size: 64 kB\nRss: 0 kB\nAnonymous: 0 kB\n"));
  assert(ParseSmaps(reserved).size() == 1);
  std::istringstream file(
      Vma("/lib/a.so", "Size: 64 kB\nRss: 4 kB\nAnonymous: 0 kB\n"));
  assert(ParseSmaps(file).empty());
  for (const auto& text :
       {std::string{}, std::string("Size: 1 kB\n"),
        Vma("", "Size: 64 kB\nRss: 8 kB\n"),
        Vma("", "Size: 63 kB\nRss: 8 kB\nAnonymous: 4 kB\n"),
        Vma("", "Size: 64 kB\nRss: 8 kB\nAnonymous: 9 kB\n"),
        Vma("", "Size: 64 kB\nRss: 8 kB\nAnonymous: 4 MB\n"),
        Vma("", "Size: 64 kB\nRss: 8 kB\nAnonymous: 18446744073709551615 kB\n"),
        Vma("", "Size: 64 kB\nRss: 8 kB\nAnonymous: 4 kB\nAnonymous: 4 kB\n"),
        Vma("") + Vma("")})
    Reject(text);
  assert(JsonString("a\"b\\c\t") == "\"a\\\"b\\\\c\\u0009\"");
  std::cout << "CPU smaps parser/boundaries/fail-closed tests PASS\n";
}
