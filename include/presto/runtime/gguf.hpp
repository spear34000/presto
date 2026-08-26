#pragma once

#include "presto/runtime/status.hpp"
#include "presto/runtime/tensor.hpp"

#include <cstdint>
#include <cstddef>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace presto::runtime {

enum class GgufTensorType : std::uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
  Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12,
  Q5_K = 13, Q6_K = 14, Q8_K = 15, IQ2_XXS = 16, IQ2_XS = 17,
  IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22,
  IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28,
  IQ1_M = 29, BF16 = 30, TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39,
  NVFP4 = 40, Q1_0 = 41, Q2_0 = 42,
};

struct GgufTypeTraits {
  const char* name = nullptr;
  std::uint32_t block_elements = 0;
  std::uint32_t block_bytes = 0;
  bool quantized = false;
};

Result<GgufTypeTraits> gguf_type_traits(std::uint32_t raw_type);
Result<std::uint64_t> checked_gguf_tensor_size(std::uint32_t raw_type,
                                               const Shape& shape);

struct GgufTensorRecord {
  std::string name;
  Shape shape;
  std::uint32_t type = 0;
  std::uint64_t relative_offset = 0;
  std::uint64_t absolute_offset = 0;
  std::uint64_t byte_size = 0;
};

struct GgufMetadataValue {
  std::uint32_t type = 0;
  std::string string;
  std::vector<std::string> strings;
  std::vector<std::uint64_t> unsigned_integers;
  std::vector<std::int64_t> signed_integers;
  std::vector<double> numbers;
};

struct GgufIndex {
  std::uint32_t version = 0;
  std::uint32_t alignment = 32;
  std::uint64_t data_offset = 0;
  std::uint64_t file_size = 0;
  std::map<std::string, std::string> metadata;
  std::map<std::string, GgufMetadataValue> typed_metadata;
  std::vector<GgufTensorRecord> tensors;

  [[nodiscard]] const GgufTensorRecord* find_tensor(std::string_view name) const;
  [[nodiscard]] const GgufMetadataValue* metadata_value(std::string_view key) const;
};

Result<GgufIndex> read_gguf_index(const std::string& path);

class MappedFile {
 public:
  MappedFile();
  ~MappedFile();
  MappedFile(MappedFile&&) noexcept;
  MappedFile& operator=(MappedFile&&) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  static Result<MappedFile> open_readonly(const std::string& path);
  [[nodiscard]] std::uint64_t size() const;
  Result<std::span<const std::byte>> bytes(std::uint64_t offset,
                                           std::uint64_t length) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class GgufWeights {
 public:
  GgufWeights() = default;
  GgufWeights(GgufWeights&&) noexcept = default;
  GgufWeights& operator=(GgufWeights&&) noexcept = default;
  GgufWeights(const GgufWeights&) = delete;
  GgufWeights& operator=(const GgufWeights&) = delete;

  static Result<GgufWeights> open(const std::string& path);
  [[nodiscard]] const GgufIndex& index() const { return index_; }
  Result<std::span<const std::byte>> tensor_bytes(std::string_view name) const;

 private:
  GgufIndex index_;
  MappedFile mapping_;
};

} // namespace presto::runtime
