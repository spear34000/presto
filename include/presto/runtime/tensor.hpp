#pragma once

#include "presto/runtime/status.hpp"

#include <cstdint>
#include <vector>

namespace presto::runtime {

enum class DataType {
  F32,
  F16,
  BF16,
  I8,
  Q4_0,
  Q8_0,
};

using Shape = std::vector<std::uint64_t>;

struct TensorLayout {
  DataType dtype = DataType::F32;
  Shape shape;
  std::vector<std::uint64_t> byte_strides;
};

Result<std::uint64_t> checked_element_count(const Shape& shape);
Result<std::uint64_t> checked_byte_size(DataType dtype, const Shape& shape);
Status validate_layout(const TensorLayout& layout);

} // namespace presto::runtime
