#include "presto/runtime/gguf.hpp"
#include "test_util.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace presto::runtime;

PRESTO_TEST(runtime_gguf_type_traits_match_disk_format) {
  struct Case { std::uint32_t type; std::uint32_t elements; std::uint32_t bytes; };
  constexpr std::array<Case, 10> cases{{
      {0, 1, 4}, {1, 1, 2}, {2, 32, 18}, {8, 32, 34}, {10, 256, 84},
      {12, 256, 144}, {14, 256, 210}, {30, 1, 2}, {39, 32, 17}, {40, 64, 36},
  }};
  for (const auto& c : cases) {
    const auto traits = gguf_type_traits(c.type);
    PRESTO_EXPECT(traits.ok());
    PRESTO_EXPECT(traits.value.block_elements == c.elements);
    PRESTO_EXPECT(traits.value.block_bytes == c.bytes);
  }
  PRESTO_EXPECT(gguf_type_traits(31).status.code == StatusCode::Unsupported);
}

PRESTO_TEST(runtime_gguf_tensor_size_checks_rows_and_overflow) {
  const auto q4 = checked_gguf_tensor_size(2, {64, 3});
  PRESTO_EXPECT(q4.ok() && q4.value == 108);
  PRESTO_EXPECT(checked_gguf_tensor_size(2, {33, 3}).status.code ==
                StatusCode::InvalidArgument);
  PRESTO_EXPECT(checked_gguf_tensor_size(0, {0, 3}).status.code ==
                StatusCode::InvalidArgument);
  PRESTO_EXPECT(!checked_gguf_tensor_size(0, {0xffffffffffffffffULL, 2}).ok());
}

namespace {
void put_u32(std::vector<unsigned char>& out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<unsigned char>(value >> (8 * i)));
}
void put_u64(std::vector<unsigned char>& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<unsigned char>(value >> (8 * i)));
}
void put_string(std::vector<unsigned char>& out, const std::string& value) {
  put_u64(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}
void put_tensor(std::vector<unsigned char>& out, const std::string& name,
                const Shape& shape, std::uint32_t type, std::uint64_t offset) {
  put_string(out, name);
  put_u32(out, static_cast<std::uint32_t>(shape.size()));
  for (const auto dimension : shape) put_u64(out, dimension);
  put_u32(out, type);
  put_u64(out, offset);
}
std::filesystem::path write_index_fixture(std::uint32_t quant_type = 2,
                                          Shape quant_shape = {32},
                                          std::uint64_t quant_offset = 32,
                                          std::string quant_name = "quant",
                                          bool include_payload = true) {
  std::vector<unsigned char> bytes{'G', 'G', 'U', 'F'};
  put_u32(bytes, 3);
    put_u64(bytes, 2);
    put_u64(bytes, 5);
  put_string(bytes, "general.alignment");
  put_u32(bytes, 4);
  put_u32(bytes, 32);
  put_string(bytes, "test.signed");
  put_u32(bytes, 5);
  put_u32(bytes, 0xfffffff9U);
  put_string(bytes, "test.strings");
  put_u32(bytes, 9); put_u32(bytes, 8); put_u64(bytes, 2);
  put_string(bytes, "hello"); put_string(bytes, "한국어");
  put_string(bytes, "test.integers");
  put_u32(bytes, 9); put_u32(bytes, 4); put_u64(bytes, 2);
  put_u32(bytes, 7); put_u32(bytes, 9);
  put_string(bytes, "test.floats");
  put_u32(bytes, 9); put_u32(bytes, 6); put_u64(bytes, 2);
  put_u32(bytes, std::bit_cast<std::uint32_t>(1.5F));
  put_u32(bytes, std::bit_cast<std::uint32_t>(-2.0F));
  put_tensor(bytes, "matrix", {2, 2}, 0, 0);
  put_tensor(bytes, quant_name, quant_shape, quant_type, quant_offset);
  while (bytes.size() % 32 != 0) bytes.push_back(0);
  if (include_payload) {
    bytes.resize(bytes.size() + 32, 0);
    for (unsigned char value = 1; value <= 18; ++value) bytes.push_back(value);
  }
  const auto path = std::filesystem::temp_directory_path() / "presto_runtime_index.gguf";
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  return path;
}

PRESTO_TEST(runtime_gguf_rejects_invalid_tensor_descriptors) {
  auto path = write_index_fixture(99);
  PRESTO_EXPECT(read_gguf_index(path.string()).status.code == StatusCode::Unsupported);
  path = write_index_fixture(2, {33});
  PRESTO_EXPECT(read_gguf_index(path.string()).status.code == StatusCode::InvalidArgument);
  path = write_index_fixture(2, {32}, 16);
  PRESTO_EXPECT(read_gguf_index(path.string()).status.code == StatusCode::InvalidArgument);
  path = write_index_fixture(2, {32}, 32, "matrix");
  PRESTO_EXPECT(read_gguf_index(path.string()).status.code == StatusCode::AlreadyExists);
  path = write_index_fixture(2, {32}, 0);
  PRESTO_EXPECT(read_gguf_index(path.string()).status.code == StatusCode::InvalidArgument);
  path = write_index_fixture(2, {32}, 32, "quant", false);
  PRESTO_EXPECT(read_gguf_index(path.string()).status.code == StatusCode::InvalidArgument);
  std::error_code ec;
  std::filesystem::remove(path, ec);
}
} // namespace

PRESTO_TEST(runtime_gguf_reads_complete_tensor_index) {
  const auto path = write_index_fixture();
  const auto index = read_gguf_index(path.string());
  PRESTO_EXPECT(index.ok());
  if (!index.ok()) {
    std::fprintf(stderr, "  GGUF parse error: %s\n", index.status.message.c_str());
    return;
  }
  PRESTO_EXPECT(index.value.version == 3);
  PRESTO_EXPECT(index.value.alignment == 32);
  PRESTO_EXPECT(index.value.metadata.at("test.signed") == "-7");
  const auto* strings = index.value.metadata_value("test.strings");
  const auto* integers = index.value.metadata_value("test.integers");
  const auto* floats = index.value.metadata_value("test.floats");
  PRESTO_EXPECT(strings != nullptr && strings->strings.size() == 2);
  PRESTO_EXPECT(strings != nullptr && strings->strings[1] == "한국어");
  PRESTO_EXPECT(integers != nullptr && integers->unsigned_integers ==
                                       std::vector<std::uint64_t>({7, 9}));
  PRESTO_EXPECT(floats != nullptr && floats->numbers == std::vector<double>({1.5, -2.0}));
  PRESTO_EXPECT(index.value.data_offset % 32 == 0);
  PRESTO_EXPECT(index.value.tensors.size() == 2);
  const auto* matrix = index.value.find_tensor("matrix");
  const auto* quant = index.value.find_tensor("quant");
  PRESTO_EXPECT(matrix != nullptr && matrix->byte_size == 16);
  PRESTO_EXPECT(matrix != nullptr && matrix->absolute_offset == index.value.data_offset);
  PRESTO_EXPECT(quant != nullptr && quant->shape == Shape{32});
  PRESTO_EXPECT(quant != nullptr && quant->absolute_offset == index.value.data_offset + 32);
  PRESTO_EXPECT(quant != nullptr && quant->byte_size == 18);
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

PRESTO_TEST(runtime_gguf_maps_exact_tensor_payload) {
  const auto path = write_index_fixture();
  auto weights = GgufWeights::open(path.string());
  PRESTO_EXPECT(weights.ok());
  const auto bytes = weights.value.tensor_bytes("quant");
  PRESTO_EXPECT(bytes.ok());
  PRESTO_EXPECT(bytes.value.size() == 18);
  PRESTO_EXPECT(std::to_integer<unsigned>(bytes.value.front()) == 1);
  PRESTO_EXPECT(std::to_integer<unsigned>(bytes.value.back()) == 18);
  PRESTO_EXPECT(weights.value.tensor_bytes("missing").status.code == StatusCode::NotFound);
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

PRESTO_TEST(runtime_mapped_file_rejects_out_of_bounds_slice) {
  const auto path = write_index_fixture();
  auto mapping = MappedFile::open_readonly(path.string());
  PRESTO_EXPECT(mapping.ok());
  PRESTO_EXPECT(mapping.value.bytes(mapping.value.size(), 1).status.code ==
                StatusCode::InvalidArgument);
  std::error_code ec;
  std::filesystem::remove(path, ec);
}
