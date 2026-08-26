#include "presto/runtime/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace presto::runtime {
namespace {

bool multiply_overflows(const std::uint64_t a, const std::uint64_t b) {
  return b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b;
}

struct Reader {
  std::ifstream file;
  std::uint64_t size = 0;
  std::uint64_t position = 0;

  explicit Reader(const std::string& path) : file(path, std::ios::binary | std::ios::ate) {
    if (file) {
      const auto end = file.tellg();
      if (end >= 0) size = static_cast<std::uint64_t>(end);
      file.seekg(0);
    }
  }
  bool read(void* destination, const std::size_t count) {
    if (count > size - std::min(size, position)) return false;
    file.read(static_cast<char*>(destination), static_cast<std::streamsize>(count));
    if (!file) return false;
    position += count;
    return true;
  }
  bool skip(const std::uint64_t count) {
    if (count > size - std::min(size, position) ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
    file.seekg(static_cast<std::streamoff>(count), std::ios::cur);
    if (!file) return false;
    position += count;
    return true;
  }
  template <typename T> bool integer(T& value) {
    unsigned char bytes[sizeof(T)]{};
    if (!read(bytes, sizeof bytes)) return false;
    value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i)
      value |= static_cast<T>(bytes[i]) << (8 * i);
    return true;
  }
};

Result<GgufIndex> index_failure(const StatusCode code, const std::string& message) {
  return Result<GgufIndex>::failure(Status::error(code, "read_gguf_index", message));
}

bool read_string(Reader& reader, const bool v1, std::string& value) {
  std::uint64_t length = 0;
  if (v1) {
    std::uint32_t short_length = 0;
    if (!reader.integer(short_length)) return false;
    length = short_length;
  } else if (!reader.integer(length)) {
    return false;
  }
  if (length > reader.size - std::min(reader.size, reader.position) || length > (1ULL << 30))
    return false;
  value.resize(static_cast<std::size_t>(length));
  return length == 0 || reader.read(value.data(), static_cast<std::size_t>(length));
}

std::uint32_t scalar_bytes(const std::uint32_t type) {
  switch (type) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
  }
}

bool read_metadata_value(Reader& reader, const bool v1, const std::uint32_t type,
                         std::string& rendered, GgufMetadataValue& typed) {
  typed.type = type;
  if (type == 8) {
    if (!read_string(reader, v1, rendered)) return false;
    typed.string = rendered;
    return true;
  }
  if (type == 9) {
    std::uint32_t element_type = 0;
    std::uint64_t count = 0;
    if (!reader.integer(element_type)) return false;
    if (v1) {
      std::uint32_t count32 = 0;
      if (!reader.integer(count32)) return false;
      count = count32;
    } else if (!reader.integer(count)) return false;
    if (element_type == 9) return false;
    const auto width = scalar_bytes(element_type);
    if (count > 10000000) return false;
    if (element_type == 8) {
      typed.strings.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t i = 0; i < count; ++i) {
        std::string value;
        if (!read_string(reader, v1, value)) return false;
        typed.strings.push_back(std::move(value));
      }
    } else if (element_type == 4 || element_type == 5 || element_type == 6) {
      for (std::uint64_t i = 0; i < count; ++i) {
        std::uint32_t bits = 0;
        if (!reader.integer(bits)) return false;
        if (element_type == 4) typed.unsigned_integers.push_back(bits);
        else if (element_type == 5) typed.signed_integers.push_back(static_cast<std::int32_t>(bits));
        else { float value = 0; std::memcpy(&value, &bits, sizeof value); typed.numbers.push_back(value); }
      }
    } else if (width != 0) {
      if (multiply_overflows(count, width) || !reader.skip(count * width)) return false;
    } else return false;
    rendered = "<array:" + std::to_string(count) + ">";
    return true;
  }
  const auto width = scalar_bytes(type);
  if (width == 0) return false;
  unsigned char bytes[8]{};
  if (!reader.read(bytes, width)) return false;
  std::uint64_t bits = 0;
  for (std::uint32_t i = 0; i < width; ++i) bits |= std::uint64_t(bytes[i]) << (8 * i);
  std::ostringstream output;
  if (type == 6) {
    std::uint32_t raw = static_cast<std::uint32_t>(bits); float value = 0;
    std::memcpy(&value, &raw, sizeof value); output << value;
  } else if (type == 12) {
    double value = 0; std::memcpy(&value, &bits, sizeof value); output << value;
  } else if (type == 7) output << (bits != 0 ? "true" : "false");
  else if (type == 1) output << static_cast<int>(static_cast<std::int8_t>(bits));
  else if (type == 3) output << static_cast<std::int16_t>(bits);
  else if (type == 5) output << static_cast<std::int32_t>(bits);
  else if (type == 11) output << static_cast<std::int64_t>(bits);
  else output << bits;
  rendered = output.str();
  return true;
}

} // namespace

Result<GgufTypeTraits> gguf_type_traits(const std::uint32_t type) {
  GgufTypeTraits t{};
  switch (type) {
    case 0: t = {"f32", 1, 4, false}; break;
    case 1: t = {"f16", 1, 2, false}; break;
    case 2: t = {"q4_0", 32, 18, true}; break;
    case 3: t = {"q4_1", 32, 20, true}; break;
    case 6: t = {"q5_0", 32, 22, true}; break;
    case 7: t = {"q5_1", 32, 24, true}; break;
    case 8: t = {"q8_0", 32, 34, true}; break;
    case 9: t = {"q8_1", 32, 36, true}; break;
    case 10: t = {"q2_K", 256, 84, true}; break;
    case 11: t = {"q3_K", 256, 110, true}; break;
    case 12: t = {"q4_K", 256, 144, true}; break;
    case 13: t = {"q5_K", 256, 176, true}; break;
    case 14: t = {"q6_K", 256, 210, true}; break;
    case 15: t = {"q8_K", 256, 292, true}; break;
    case 16: t = {"iq2_xxs", 256, 66, true}; break;
    case 17: t = {"iq2_xs", 256, 74, true}; break;
    case 18: t = {"iq3_xxs", 256, 98, true}; break;
    case 19: t = {"iq1_s", 256, 50, true}; break;
    case 20: t = {"iq4_nl", 32, 18, true}; break;
    case 21: t = {"iq3_s", 256, 110, true}; break;
    case 22: t = {"iq2_s", 256, 82, true}; break;
    case 23: t = {"iq4_xs", 256, 136, true}; break;
    case 24: t = {"i8", 1, 1, false}; break;
    case 25: t = {"i16", 1, 2, false}; break;
    case 26: t = {"i32", 1, 4, false}; break;
    case 27: t = {"i64", 1, 8, false}; break;
    case 28: t = {"f64", 1, 8, false}; break;
    case 29: t = {"iq1_m", 256, 56, true}; break;
    case 30: t = {"bf16", 1, 2, false}; break;
    case 34: t = {"tq1_0", 256, 54, true}; break;
    case 35: t = {"tq2_0", 256, 66, true}; break;
    case 39: t = {"mxfp4", 32, 17, true}; break;
    case 40: t = {"nvfp4", 64, 36, true}; break;
    case 41: t = {"q1_0", 128, 18, true}; break;
    case 42: t = {"q2_0", 64, 18, true}; break;
    default:
      return Result<GgufTypeTraits>::failure(Status::error(
          StatusCode::Unsupported, "gguf_type_traits",
          "unknown or removed GGUF tensor type " + std::to_string(type)));
  }
  return Result<GgufTypeTraits>::success(t);
}

Result<std::uint64_t> checked_gguf_tensor_size(const std::uint32_t type,
                                               const Shape& shape) {
  const auto traits = gguf_type_traits(type);
  if (!traits.ok()) return Result<std::uint64_t>::failure(traits.status);
  if (shape.empty()) {
    return Result<std::uint64_t>::failure(Status::error(
        StatusCode::InvalidArgument, "gguf_tensor_size", "tensor rank must be non-zero"));
  }
  for (const auto dimension : shape) {
    if (dimension == 0) {
      return Result<std::uint64_t>::failure(Status::error(
          StatusCode::InvalidArgument, "gguf_tensor_size", "tensor dimensions must be non-zero"));
    }
  }
  if (shape[0] % traits.value.block_elements != 0) {
    return Result<std::uint64_t>::failure(Status::error(
        StatusCode::InvalidArgument, "gguf_tensor_size",
        "first tensor dimension is not a full quantization block"));
  }
  std::uint64_t bytes = (shape[0] / traits.value.block_elements) * traits.value.block_bytes;
  for (std::size_t i = 1; i < shape.size(); ++i) {
    if (multiply_overflows(bytes, shape[i])) {
      return Result<std::uint64_t>::failure(Status::error(
          StatusCode::Overflow, "gguf_tensor_size", "tensor byte size overflows uint64"));
    }
    bytes *= shape[i];
  }
  return Result<std::uint64_t>::success(bytes);
}

const GgufTensorRecord* GgufIndex::find_tensor(const std::string_view name) const {
  const auto it = std::find_if(tensors.begin(), tensors.end(),
                               [name](const auto& tensor) { return tensor.name == name; });
  return it == tensors.end() ? nullptr : &*it;
}

const GgufMetadataValue* GgufIndex::metadata_value(const std::string_view key) const {
  const auto it = typed_metadata.find(std::string(key));
  return it == typed_metadata.end() ? nullptr : &it->second;
}

Result<GgufIndex> read_gguf_index(const std::string& path) {
  Reader reader(path);
  if (!reader.file) return index_failure(StatusCode::NotFound, "cannot open GGUF file");
  char magic[4]{};
  if (!reader.read(magic, sizeof magic) || std::memcmp(magic, "GGUF", 4) != 0)
    return index_failure(StatusCode::InvalidArgument, "missing GGUF magic");
  GgufIndex index;
  index.file_size = reader.size;
  if (!reader.integer(index.version) || index.version < 1 || index.version > 3)
    return index_failure(StatusCode::Unsupported, "unsupported GGUF version");
  const bool v1 = index.version == 1;
  std::uint64_t tensor_count = 0, metadata_count = 0;
  if (v1) {
    std::uint32_t tc = 0, mc = 0;
    if (!reader.integer(tc) || !reader.integer(mc))
      return index_failure(StatusCode::InvalidArgument, "truncated GGUF header");
    tensor_count = tc; metadata_count = mc;
  } else if (!reader.integer(tensor_count) || !reader.integer(metadata_count)) {
    return index_failure(StatusCode::InvalidArgument, "truncated GGUF header");
  }
  if (tensor_count > 10000000 || metadata_count > 10000000)
    return index_failure(StatusCode::InvalidArgument, "unreasonable GGUF entry count");
  for (std::uint64_t i = 0; i < metadata_count; ++i) {
    std::string key, value;
    std::uint32_t type = 0;
    GgufMetadataValue typed;
    if (!read_string(reader, v1, key) || !reader.integer(type) ||
        !read_metadata_value(reader, v1, type, value, typed))
      return index_failure(StatusCode::InvalidArgument, "truncated or invalid GGUF metadata");
    index.metadata[key] = std::move(value);
    index.typed_metadata[key] = std::move(typed);
  }
  if (const auto it = index.metadata.find("general.alignment"); it != index.metadata.end()) {
    try { index.alignment = static_cast<std::uint32_t>(std::stoul(it->second)); }
    catch (...) { return index_failure(StatusCode::InvalidArgument, "invalid GGUF alignment"); }
  }
  if (index.alignment == 0 || (index.alignment & (index.alignment - 1)) != 0)
    return index_failure(StatusCode::InvalidArgument, "GGUF alignment must be a non-zero power of two");

  std::unordered_set<std::string> names;
  index.tensors.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    GgufTensorRecord tensor;
    std::uint32_t rank = 0;
    if (!read_string(reader, v1, tensor.name) || !reader.integer(rank) || rank == 0 || rank > 4)
      return index_failure(StatusCode::InvalidArgument, "invalid GGUF tensor descriptor");
    if (!names.insert(tensor.name).second)
      return index_failure(StatusCode::AlreadyExists, "duplicate GGUF tensor name");
    tensor.shape.resize(rank);
    for (auto& dimension : tensor.shape)
      if (!reader.integer(dimension) || dimension == 0)
        return index_failure(StatusCode::InvalidArgument, "invalid GGUF tensor dimension");
    if (!reader.integer(tensor.type) || !reader.integer(tensor.relative_offset))
      return index_failure(StatusCode::InvalidArgument, "truncated GGUF tensor descriptor");
    if (tensor.relative_offset % index.alignment != 0)
      return index_failure(StatusCode::InvalidArgument, "misaligned GGUF tensor offset");
    const auto bytes = checked_gguf_tensor_size(tensor.type, tensor.shape);
    if (!bytes.ok()) return index_failure(bytes.status.code, bytes.status.message);
    tensor.byte_size = bytes.value;
    index.tensors.push_back(std::move(tensor));
  }
  const std::uint64_t mask = index.alignment - 1;
  if (reader.position > std::numeric_limits<std::uint64_t>::max() - mask)
    return index_failure(StatusCode::Overflow, "GGUF data offset overflows");
  index.data_offset = (reader.position + mask) & ~mask;
  for (auto& tensor : index.tensors) {
    if (tensor.relative_offset > std::numeric_limits<std::uint64_t>::max() - index.data_offset)
      return index_failure(StatusCode::Overflow, "GGUF tensor offset overflows");
    tensor.absolute_offset = index.data_offset + tensor.relative_offset;
    if (tensor.byte_size > index.file_size - std::min(index.file_size, tensor.absolute_offset))
      return index_failure(StatusCode::InvalidArgument, "GGUF tensor payload extends beyond file");
  }
  std::vector<const GgufTensorRecord*> sorted;
  for (const auto& tensor : index.tensors) sorted.push_back(&tensor);
  std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
    return a->absolute_offset < b->absolute_offset;
  });
  for (std::size_t i = 1; i < sorted.size(); ++i) {
    if (sorted[i - 1]->byte_size > sorted[i]->absolute_offset - sorted[i - 1]->absolute_offset)
      return index_failure(StatusCode::InvalidArgument, "overlapping GGUF tensor payloads");
  }
  return Result<GgufIndex>::success(std::move(index));
}

} // namespace presto::runtime
