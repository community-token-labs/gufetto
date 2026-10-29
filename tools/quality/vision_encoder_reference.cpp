// Standalone Qwen4-Exp vision reference. Specification: pinned
// modeling_qwen4_exp.py, Qwen4ExpVisionPatchEmbed/Block/PatchMerger/Model
// (lines 1746-2007); tensor names/shapes: retained
// local-projector-inventory.json. This file deliberately does not include Gufo
// model, tensor or GGUF code.
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vision_ref_math.hpp"

namespace {
namespace fs = std::filesystem;
using vision_ref::Matrix;
constexpr size_t kWidth = 1152;
constexpr size_t kLayers = 27;
constexpr uint64_t kMaxPixels = 16777216;
constexpr uint64_t kMaxHeader = 16 * 1024 * 1024;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

uint64_t checked_add(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    fail("integer addition overflow");
  return a + b;
}

uint64_t checked_mul(uint64_t a, uint64_t b) {
  if (a && b > std::numeric_limits<uint64_t>::max() / a)
    fail("integer multiplication overflow");
  return a * b;
}

size_t number(std::string_view text) {
  size_t result = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), result);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size())
    fail("invalid unsigned decimal: " + std::string(text));
  return result;
}

uint32_t le32(const unsigned char* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

// Seek/read are bounded against the original file length, including header
// reads.
class Input {
public:
  explicit Input(const fs::path& path) {
    if (!fs::is_regular_file(path))
      fail("input is not a regular file: " + path.string());
    stream_.open(path, std::ios::binary);
    if (!stream_)
      fail("cannot open input: " + path.string());
    size_ = fs::file_size(path);
    if (size_ > uint64_t(std::numeric_limits<std::streamoff>::max()))
      fail("input too large for stream offsets");
  }

  uint64_t size() const { return size_; }
  uint64_t position() const { return position_; }
  void seek(uint64_t position) {
    if (position > size_)
      fail("input seek beyond file");
    stream_.seekg(static_cast<std::streamoff>(position));
    if (!stream_)
      fail("input seek failed");
    position_ = position;
  }
  void read(void* destination, size_t count) {
    if (uint64_t(count) > size_ - position_ ||
        uint64_t(count) > uint64_t(std::numeric_limits<std::streamsize>::max()))
      fail("truncated input");
    stream_.read(static_cast<char*>(destination),
                 static_cast<std::streamsize>(count));
    if (!stream_)
      fail("input read failed");
    position_ += count;
  }
  uint64_t integer(size_t bytes) {
    std::array<unsigned char, 8> data{};
    if (bytes > data.size())
      fail("invalid integer size");
    read(data.data(), bytes);
    uint64_t value = 0;
    for (size_t i = 0; i < bytes; ++i)
      value |= uint64_t(data[i]) << (8 * i);
    return value;
  }

private:
  std::ifstream stream_;
  uint64_t size_ = 0;
  uint64_t position_ = 0;
};

struct Metadata {
  uint32_t type = 0;
  uint64_t integer = 0;
  double real = 0;
  std::string text;
  std::vector<Metadata> array;
};

struct Tensor {
  std::vector<uint64_t> shape;
  uint32_t type = 0;
  uint64_t offset = 0;
  uint64_t elements = 0;
  uint64_t bytes = 0;
};

struct ExpectedTensor {
  std::vector<uint64_t> shape;
  bool allow_bf16 = false;
};

std::map<std::string, ExpectedTensor> expected_tensors() {
  std::map<std::string, ExpectedTensor> result;
  auto norm = [&](const std::string& name) {
    result.emplace(name + ".weight", ExpectedTensor{{1152}, false});
    result.emplace(name + ".bias", ExpectedTensor{{1152}, false});
  };
  auto linear = [&](const std::string& name, uint64_t in, uint64_t out) {
    result.emplace(name + ".weight", ExpectedTensor{{in, out}, true});
    result.emplace(name + ".bias", ExpectedTensor{{out}, false});
  };
  for (size_t i = 0; i < kLayers; ++i) {
    const auto prefix = "v.blk." + std::to_string(i) + ".";
    linear(prefix + "attn_qkv", 1152, 3456);
    linear(prefix + "attn_out", 1152, 1152);
    linear(prefix + "ffn_up", 1152, 4304);
    linear(prefix + "ffn_down", 4304, 1152);
    norm(prefix + "ln1");
    norm(prefix + "ln2");
  }
  linear("mm.0", 4608, 4608);
  linear("mm.2", 4608, 2560);
  norm("v.post_ln");
  result.emplace("v.patch_embd.bias", ExpectedTensor{{1152}, false});
  result.emplace("v.patch_embd.weight",
                 ExpectedTensor{{16, 16, 3, 1152}, false});
  result.emplace("v.patch_embd.weight.1",
                 ExpectedTensor{{16, 16, 3, 1152}, false});
  result.emplace("v.position_embd.weight", ExpectedTensor{{1152, 2304}, false});
  return result;
}

class Gguf {
public:
  explicit Gguf(const fs::path& path) : input_(path) {
    if (input_.integer(4) != 0x46554747 || input_.integer(4) != 3)
      fail("expected little-endian GGUF v3");
    const auto tensor_count = input_.integer(8);
    const auto metadata_count = input_.integer(8);
    if (tensor_count != 334 || metadata_count > 4096)
      fail("unsupported tensor/metadata count");
    std::map<std::string, Metadata> metadata;
    for (uint64_t i = 0; i < metadata_count; ++i) {
      auto key = header_string(4096);
      const auto type = static_cast<uint32_t>(header_integer(4));
      auto value = metadata_value(type, 0);
      if (!metadata.emplace(std::move(key), std::move(value)).second)
        fail("duplicate GGUF metadata key");
    }
    uint64_t alignment = 32;
    if (const auto it = metadata.find("general.alignment");
        it != metadata.end()) {
      if (it->second.type != 4)
        fail("GGUF alignment must be uint32");
      alignment = it->second.integer;
    }
    if (!alignment || alignment > 4096 || (alignment & (alignment - 1)))
      fail("invalid GGUF alignment");
    validate_metadata(metadata);
    const auto expected = expected_tensors();
    for (uint64_t i = 0; i < tensor_count; ++i) {
      auto name = header_string(4096);
      const auto wanted = expected.find(name);
      if (wanted == expected.end())
        fail("unexpected tensor: " + name);
      const auto rank = header_integer(4);
      if (rank != wanted->second.shape.size())
        fail("incorrect tensor rank: " + name);
      Tensor tensor;
      tensor.elements = 1;
      for (size_t d = 0; d < rank; ++d) {
        const auto dimension = header_integer(8);
        if (dimension != wanted->second.shape[d])
          fail("incorrect tensor shape: " + name);
        tensor.shape.push_back(dimension);
        tensor.elements = checked_mul(tensor.elements, dimension);
      }
      tensor.type = static_cast<uint32_t>(header_integer(4));
      if (tensor.type != 0 && !(wanted->second.allow_bf16 && tensor.type == 30))
        fail("unsupported tensor type: " + name);
      tensor.bytes = checked_mul(tensor.elements, tensor.type == 0 ? 4 : 2);
      tensor.offset = header_integer(8);
      if (tensor.offset % alignment)
        fail("unaligned tensor: " + name);
      if (!tensors_.emplace(std::move(name), std::move(tensor)).second)
        fail("duplicate GGUF tensor name");
    }
    // Exact count + unique names drawn from the exact inventory imply
    // completeness.
    if (tensors_.size() != expected.size())
      fail("incomplete GGUF inventory");
    data_start_ =
        checked_add(input_.position(), alignment - 1) & ~(alignment - 1);
    if (data_start_ > input_.size())
      fail("truncated GGUF header padding");
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (const auto& [name, tensor] : tensors_) {
      const auto begin = checked_add(data_start_, tensor.offset);
      const auto end = checked_add(begin, tensor.bytes);
      if (end > input_.size())
        fail("truncated tensor: " + name);
      ranges.emplace_back(begin, end);
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i)
      if (ranges[i].first < ranges[i - 1].second)
        fail("overlapping GGUF tensor payloads");
  }

  // Decode every loaded value, including BF16 NaN/Inf, before arithmetic.
  // No all-model weight cache: storage dies after each individual operation.
  std::vector<float> values(const std::string& name) {
    const auto& tensor = tensors_.at(name);
    input_.seek(checked_add(data_start_, tensor.offset));
    std::vector<float> result(static_cast<size_t>(tensor.elements));
    const size_t stride = tensor.type == 0 ? 4 : 2;
    std::array<unsigned char, 65536> bytes{};
    for (size_t start = 0; start < result.size();) {
      const auto count = std::min(result.size() - start, bytes.size() / stride);
      input_.read(bytes.data(), count * stride);
      for (size_t i = 0; i < count; ++i) {
        const auto* p = bytes.data() + i * stride;
        const uint32_t bits =
            stride == 4 ? le32(p)
                        : (uint32_t(p[0]) << 16) | (uint32_t(p[1]) << 24);
        const float value = std::bit_cast<float>(bits);
        if (!std::isfinite(value))
          fail("nonfinite tensor value: " + name);
        result[start + i] = value;
      }
      start += count;
    }
    return result;
  }

  Matrix weight(const std::string& name) {
    const auto& shape = tensors_.at(name).shape;
    if (shape.size() != 2)
      fail("linear weight is not rank two");
    const auto raw = values(name);
    Matrix result(static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1]));
    for (size_t out = 0; out < result.cols; ++out)
      for (size_t in = 0; in < result.rows; ++in)
        result.values[in * result.cols + out] = raw[out * result.rows + in];
    return result;
  }

  Matrix patch_weight() {
    Matrix result(1536, kWidth);
    // HF Conv3d is [out,C,T,Y,X]. The retained GGUF inventory stores
    // two [out,C,Y,X] temporal slices, named weight and weight.1.
    // Reconstruct C,T,Y,X, NOT T,C,Y,X, and NEVER add the slices before
    // projection: doing so changes the ordered F32 reduction.
    for (size_t t = 0; t < 2; ++t) {
      const auto raw =
          values(t == 0 ? "v.patch_embd.weight" : "v.patch_embd.weight.1");
      for (size_t out = 0; out < kWidth; ++out)
        for (size_t c = 0; c < 3; ++c)
          for (size_t y = 0; y < 16; ++y)
            for (size_t x = 0; x < 16; ++x) {
              const auto in = (c * 2 + t) * 256 + y * 16 + x;
              result.values[in * kWidth + out] =
                  raw[out * 768 + c * 256 + y * 16 + x];
            }
    }
    return result;
  }

  Matrix positions() {
    Matrix result(2304, kWidth);
    result.values = values("v.position_embd.weight");
    return result;
  }

private:
  Input input_;
  uint64_t data_start_ = 0;
  uint64_t metadata_nodes_ = 0;
  std::map<std::string, Tensor> tensors_;

  void header_room(uint64_t bytes) const {
    if (input_.position() > kMaxHeader ||
        bytes > kMaxHeader - input_.position())
      fail("GGUF header exceeds 16 MiB bound");
  }
  uint64_t header_integer(size_t bytes) {
    header_room(bytes);
    return input_.integer(bytes);
  }
  std::string header_string(uint64_t limit) {
    const auto length = header_integer(8);
    if (length > limit)
      fail("oversized GGUF string");
    header_room(length);
    std::string result(static_cast<size_t>(length), '\0');
    input_.read(result.data(), result.size());
    if (result.find('\0') != std::string::npos)
      fail("embedded NUL in GGUF string");
    return result;
  }
  Metadata metadata_value(uint32_t type, size_t depth) {
    if (depth > 8 || ++metadata_nodes_ > 262144)
      fail("GGUF metadata recursion/element bound exceeded");
    Metadata result;
    result.type = type;
    switch (type) {
      case 0:
      case 1:
      case 7:
        result.integer = header_integer(1);
        if (type == 7 && result.integer > 1)
          fail("invalid GGUF boolean");
        break;
      case 2:
      case 3:
        result.integer = header_integer(2);
        break;
      case 4:
      case 5:
        result.integer = header_integer(4);
        break;
      case 6:
        result.real =
            std::bit_cast<float>(static_cast<uint32_t>(header_integer(4)));
        break;
      case 8:
        result.text = header_string(1024 * 1024);
        break;
      case 9: {
        const auto subtype = static_cast<uint32_t>(header_integer(4));
        const auto count = header_integer(8);
        if (subtype > 12 || count > 262144 - metadata_nodes_)
          fail("invalid/oversized GGUF metadata array");
        result.array.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
          result.array.push_back(metadata_value(subtype, depth + 1));
        break;
      }
      case 10:
      case 11:
        result.integer = header_integer(8);
        break;
      case 12:
        result.real = std::bit_cast<double>(header_integer(8));
        break;
      default:
        fail("unknown GGUF metadata type");
    }
    if ((type == 6 || type == 12) && !std::isfinite(result.real))
      fail("nonfinite GGUF metadata");
    return result;
  }

  static void validate_metadata(
      const std::map<std::string, Metadata>& metadata) {
    // Tensor inventory is mandatory. Optional descriptive metadata may be
    // omitted, but if supplied must not contradict this fixed architecture.
    auto integer = [&](const std::string& key, uint32_t type,
                       uint64_t expected) {
      const auto it = metadata.find(key);
      if (it != metadata.end() &&
          (it->second.type != type || it->second.integer != expected))
        fail("incompatible GGUF metadata: " + key);
    };
    auto text = [&](const std::string& key, const std::string& expected) {
      const auto it = metadata.find(key);
      if (it != metadata.end() &&
          (it->second.type != 8 || it->second.text != expected))
        fail("incompatible GGUF metadata: " + key);
    };
    text("general.architecture", "clip");
    text("general.type", "mmproj");
    text("clip.projector_type", "qwen3vl_merger");
    integer("clip.has_vision_encoder", 7, 1);
    integer("clip.use_gelu", 7, 1);
    integer("clip.vision.projection_dim", 4, 2560);
    integer("clip.vision.patch_size", 4, 16);
    integer("clip.vision.embedding_length", 4, 1152);
    integer("clip.vision.feed_forward_length", 4, 4304);
    integer("clip.vision.block_count", 4, 27);
    integer("clip.vision.attention.head_count", 4, 16);
    integer("clip.vision.spatial_merge_size", 4, 2);
    const auto eps = metadata.find("clip.vision.attention.layer_norm_epsilon");
    if (eps != metadata.end() &&
        (eps->second.type != 6 || eps->second.real != double(1e-6f)))
      fail("incompatible GGUF layer norm epsilon");
    for (const std::string key :
         {"clip.vision.image_mean", "clip.vision.image_std"}) {
      const auto it = metadata.find(key);
      if (it == metadata.end())
        continue;
      if (it->second.type != 9 || it->second.array.size() != 3)
        fail("incompatible GGUF normalization: " + key);
      for (const auto& value : it->second.array)
        if (value.type != 6 || value.real != 0.5)
          fail("incompatible GGUF normalization: " + key);
    }
    const auto deepstack = metadata.find("clip.vision.is_deepstack_layers");
    if (deepstack != metadata.end()) {
      if (deepstack->second.type != 9 ||
          deepstack->second.array.size() != kLayers)
        fail("incompatible GGUF deepstack metadata");
      for (const auto& value : deepstack->second.array)
        if (value.type != 7 || value.integer != 0)
          fail("deepstack is not supported by this reference");
    }
  }
};

// C++20 has no ios::noreplace. Build in an exclusively created staging
// directory, then publish with create_hard_link (which never replaces a path).
// No POSIX/Windows API is needed. The wrapper owns a new private output dir.
class ExclusiveOutput {
public:
  explicit ExclusiveOutput(const fs::path& path)
      : path_(path), staging_(path.string() + ".writing") {
    if (fs::symlink_status(path_).type() != fs::file_type::not_found)
      fail("output already exists: " + path_.string());
    if (!fs::create_directory(staging_))
      fail("output staging path already exists: " + staging_.string());
    temporary_ = staging_ / "payload";
    stream_.open(temporary_, std::ios::binary | std::ios::out);
    if (!stream_) {
      std::error_code ignored;
      fs::remove(staging_, ignored);
      fail("cannot create output: " + path_.string());
    }
  }
  ExclusiveOutput(const ExclusiveOutput&) = delete;
  ExclusiveOutput& operator=(const ExclusiveOutput&) = delete;
  ~ExclusiveOutput() {
    stream_.close();
    std::error_code ignored;
    fs::remove(temporary_, ignored);
    fs::remove(staging_, ignored);
  }
  void floats(const float* values, size_t count) {
    std::array<unsigned char, 65536> buffer{};
    for (size_t start = 0; start < count;) {
      const auto n = std::min(count - start, buffer.size() / 4);
      for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[start + i]))
          fail("nonfinite output value");
        const auto bits = std::bit_cast<uint32_t>(values[start + i]);
        for (size_t j = 0; j < 4; ++j)
          buffer[4 * i + j] = static_cast<unsigned char>(bits >> (8 * j));
      }
      stream_.write(reinterpret_cast<const char*>(buffer.data()),
                    static_cast<std::streamsize>(n * 4));
      if (!stream_)
        fail("output write failed");
      start += n;
    }
  }
  void finish() {
    stream_.flush();
    if (!stream_)
      fail("output flush failed");
    stream_.close();
    if (!stream_)
      fail("output close failed");
    fs::create_hard_link(temporary_, path_);
  }

private:
  fs::path path_, staging_, temporary_;
  std::ofstream stream_;
};

struct Selection {
  std::string stage;
  std::vector<size_t> rows;
};

std::vector<Selection> read_selection(const fs::path& path, size_t patches) {
  Input file(path);
  if (file.size() > 32 * 1024 * 1024)
    fail("selection text exceeds bound");
  std::string content(static_cast<size_t>(file.size()), '\0');
  file.read(content.data(), content.size());
  std::istringstream lines(content);
  std::vector<Selection> result;
  for (size_t i = 0; i < kLayers + 2; ++i) {
    Selection selected;
    selected.stage = i == 0             ? "patch"
                     : i == kLayers + 1 ? "embedding"
                                        : "layer" + std::to_string(i - 1);
    std::string line, stage, count_text;
    if (!std::getline(lines, line))
      fail("missing selection stage: " + selected.stage);
    std::istringstream tokens(line);
    if (!(tokens >> stage >> count_text) || stage != selected.stage)
      fail("selection stages must be patch, layer0..layer26, embedding");
    const auto count = number(count_text);
    const auto row_limit = i == kLayers + 1 ? patches / 4 : patches;
    if (count > row_limit)
      fail("selection count exceeds stage row count");
    selected.rows.reserve(count);
    for (size_t j = 0; j < count; ++j) {
      std::string index;
      if (!(tokens >> index))
        fail("missing selected row");
      const auto row = number(index);
      if (row >= row_limit || (j && row <= selected.rows.back()))
        fail("selected rows must be valid, unique and sorted");
      selected.rows.push_back(row);
    }
    std::string extra;
    if (tokens >> extra)
      fail("extra selection fields");
    result.push_back(std::move(selected));
  }
  std::string extra;
  if (std::getline(lines, extra))
    fail("selection must contain exactly 29 lines");
  return result;
}

void capture(const fs::path& directory, const Selection& selection,
             const Matrix& matrix) {
  const auto width = selection.stage == "embedding" ? 2560 : kWidth;
  if (matrix.cols != width || matrix.values.size() != matrix.rows * matrix.cols)
    fail("incorrect stage matrix shape");
  ExclusiveOutput file(directory / (selection.stage + ".f32le"));
  for (const auto row : selection.rows) {
    if (row >= matrix.rows)
      fail("selected stage row out of range");
    file.floats(matrix.values.data() + row * matrix.cols, matrix.cols);
  }
  file.finish();
  std::cout << "stage " << selection.stage << " rows " << selection.rows.size()
            << std::endl;
}

Matrix project(Gguf& model, const Matrix& x, const std::string& name) {
  const auto weight = model.weight(name + ".weight");
  const auto bias = model.values(name + ".bias");
  return vision_ref::linear(x, weight, bias);
}

Matrix normalize(Gguf& model, const Matrix& x, const std::string& name) {
  const auto gain = model.values(name + ".weight");
  const auto bias = model.values(name + ".bias");
  return vision_ref::layer_norm(x, gain, bias, 1e-6f);
}

Matrix attended(Gguf& model, const Matrix& x, const std::string& prefix,
                size_t width, size_t height) {
  Matrix q(x.rows, kWidth), k(x.rows, kWidth), v(x.rows, kWidth);
  {
    const auto norm = normalize(model, x, prefix + "ln1");
    const auto qkv = project(model, norm, prefix + "attn_qkv");
    for (size_t row = 0; row < x.rows; ++row) {
      const auto* source = qkv.values.data() + row * 3 * kWidth;
      std::copy_n(source, kWidth, q.values.data() + row * kWidth);
      std::copy_n(source + kWidth, kWidth, k.values.data() + row * kWidth);
      std::copy_n(source + 2 * kWidth, kWidth, v.values.data() + row * kWidth);
    }
  }
  vision_ref::apply_rope(q, k, width, height, 16, 10000.0);
  return vision_ref::attention(q, k, v, 16, 128);
}

void encode(const fs::path& model_path, const fs::path& rgb_path, size_t width,
            size_t height, const fs::path& selection_path,
            const fs::path& output) {
  if (width < 32 || height < 32 || width % 32 || height % 32 ||
      checked_mul(width, height) > kMaxPixels)
    fail("RGB dimensions must be multiples of 32, area <=16777216");
  if (fs::symlink_status(output).type() != fs::file_type::directory ||
      !fs::is_empty(output))
    fail("output must be an existing empty private directory");
  const auto patches = (width / 16) * (height / 16);
  const auto selection = read_selection(selection_path, patches);
  Gguf model(model_path);
  Matrix x(patches, kWidth);
  {
    Input rgb_input(rgb_path);
    const auto bytes = checked_mul(checked_mul(width, height), 3);
    if (rgb_input.size() != bytes)
      fail("RGB8 input length does not match dimensions");
    std::vector<unsigned char> rgb(static_cast<size_t>(bytes));
    rgb_input.read(rgb.data(), rgb.size());
    const auto pixels = vision_ref::patchify(rgb, width, height);
    const auto weight = model.patch_weight();
    const auto bias = model.values("v.patch_embd.bias");
    x = vision_ref::linear(pixels, weight, bias);
  }
  {
    const auto positions = model.positions();
    vision_ref::add_positions(x, positions, width, height);
  }
  capture(output, selection[0], x);
  for (size_t i = 0; i < kLayers; ++i) {
    const auto prefix = "v.blk." + std::to_string(i) + ".";
    {
      // Q/K/V and normalized input die before the output projection.
      const auto attention = attended(model, x, prefix, width, height);
      const auto delta = project(model, attention, prefix + "attn_out");
      vision_ref::add_inplace(x, delta);
    }
    {
      auto hidden = normalize(model, x, prefix + "ln2");
      hidden = project(model, hidden, prefix + "ffn_up");
      vision_ref::gelu(hidden, true);  // HF configured gelu_pytorch_tanh.
      hidden = project(model, hidden, prefix + "ffn_down");
      vision_ref::add_inplace(x, hidden);
    }
    capture(output, selection[i + 1], x);
  }
  // HF use_postshuffle_norm=False: normalize each 1152-vector BEFORE
  // grouping four adjacent merge-block-major rows into a 4608-vector.
  x = normalize(model, x, "v.post_ln");
  x.rows /= 4;
  x.cols *= 4;
  x = project(model, x, "mm.0");
  vision_ref::gelu(x, false);  // Merger uses default nn.GELU: erf, not tanh.
  x = project(model, x, "mm.2");
  capture(output, selection.back(), x);
}

void attention_cli(const fs::path& input_path, const fs::path& output_path) {
  Input input(input_path);
  const auto rows = static_cast<size_t>(input.integer(4));
  const auto heads = static_cast<size_t>(input.integer(4));
  const auto dim = static_cast<size_t>(input.integer(4));
  if (!rows || rows > 257 || !heads || heads > 16 || !dim || dim > 128)
    fail("attention dimensions exceed analytic-test bounds");
  const auto columns = heads * dim;
  const auto count = rows * columns;
  if (input.size() != 12 + 3 * count * 4)
    fail("attention input must have exact length");
  Matrix q(rows, columns), k(rows, columns), v(rows, columns);
  for (Matrix* matrix : {&q, &k, &v}) {
    for (auto& value : matrix->values) {
      value = std::bit_cast<float>(static_cast<uint32_t>(input.integer(4)));
      if (!std::isfinite(value))
        fail("nonfinite attention input");
    }
  }
  const auto result = vision_ref::attention(q, k, v, heads, 128);
  ExclusiveOutput output(output_path);
  output.floats(result.values.data(), result.values.size());
  output.finish();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 8 && std::string_view(argv[1]) == "--encode") {
      encode(argv[2], argv[3], number(argv[4]), number(argv[5]), argv[6],
             argv[7]);
    } else if (argc == 4 && std::string_view(argv[1]) == "--attention") {
      attention_cli(argv[2], argv[3]);
    } else {
      fail(
          "usage: vision_encoder_reference --encode MODEL_GGUF RGB8 WIDTH "
          "HEIGHT SELECTION_TXT NEW_OUTPUT_DIR\n"
          "       vision_encoder_reference --attention INPUT OUTPUT");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "vision_encoder_reference: " << error.what() << '\n';
    return 1;
  }
}
