#include "presto/runtime/tensor.hpp"

#include <limits>

namespace presto::runtime {
namespace {

struct TypeStorage {
  std::uint64_t block_elements;
  std::uint64_t block_bytes;
};

TypeStorage storage_for(const DataType dtype) {
  switch (dtype) {
    case DataType::F32: return {1, 4};
    case DataType::F16:
    case DataType::BF16: return {1, 2};
    case DataType::I8: return {1, 1};
    case DataType::Q4_0: return {32, 18};
    case DataType::Q8_0: return {32, 34};
  }
  return {0, 0};
}

bool multiply_overflows(const std::uint64_t a, const std::uint64_t b) {
  return b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b;
}

} // namespace

Result<std::uint64_t> checked_element_count(const Shape& shape) {
  if (shape.empty()) {
    return Result<std::uint64_t>::failure(
        Status::error(StatusCode::InvalidArgument, "element_count", "shape has no dimensions"));
  }
  std::uint64_t count = 1;
  for (const std::uint64_t dimension : shape) {
    if (dimension == 0) {
      return Result<std::uint64_t>::failure(Status::error(
          StatusCode::InvalidArgument, "element_count", "shape dimensions must be non-zero"));
    }
    if (multiply_overflows(count, dimension)) {
      return Result<std::uint64_t>::failure(
          Status::error(StatusCode::Overflow, "element_count", "shape product overflows uint64"));
    }
    count *= dimension;
  }
  return Result<std::uint64_t>::success(count);
}

Result<std::uint64_t> checked_byte_size(const DataType dtype, const Shape& shape) {
  const auto count = checked_element_count(shape);
  if (!count.ok()) return Result<std::uint64_t>::failure(count.status);
  const TypeStorage storage = storage_for(dtype);
  if (storage.block_elements == 0) {
    return Result<std::uint64_t>::failure(
        Status::error(StatusCode::Unsupported, "byte_size", "unknown data type"));
  }
  if (count.value % storage.block_elements != 0) {
    return Result<std::uint64_t>::failure(Status::error(
        StatusCode::InvalidArgument, "byte_size", "element count is not a full quantization block"));
  }
  const std::uint64_t blocks = count.value / storage.block_elements;
  if (multiply_overflows(blocks, storage.block_bytes)) {
    return Result<std::uint64_t>::failure(
        Status::error(StatusCode::Overflow, "byte_size", "tensor byte size overflows uint64"));
  }
  return Result<std::uint64_t>::success(blocks * storage.block_bytes);
}

Status validate_layout(const TensorLayout& layout) {
  if (layout.shape.empty() || layout.byte_strides.size() != layout.shape.size()) {
    return Status::error(StatusCode::InvalidArgument, "validate_layout",
                         "shape and stride ranks must be equal and non-zero");
  }
  const TypeStorage storage = storage_for(layout.dtype);
  if (storage.block_elements != 1) {
    return Status::error(StatusCode::Unsupported, "validate_layout",
                         "explicit quantized strides are not supported yet");
  }
  std::uint64_t required = storage.block_bytes;
  for (std::size_t i = layout.shape.size(); i-- > 0;) {
    if (layout.byte_strides[i] < required) {
      return Status::error(StatusCode::InvalidArgument, "validate_layout",
                           "tensor strides overlap");
    }
    if (multiply_overflows(layout.byte_strides[i], layout.shape[i])) {
      return Status::error(StatusCode::Overflow, "validate_layout", "stride extent overflows uint64");
    }
    required = layout.byte_strides[i] * layout.shape[i];
  }
  return Status::success();
}

} // namespace presto::runtime
