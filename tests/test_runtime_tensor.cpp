#include "presto/runtime/tensor.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <limits>

using namespace presto::runtime;

PRESTO_TEST(runtime_tensor_counts_elements_without_overflow) {
  const auto count = checked_element_count({2, 3, 4});
  PRESTO_EXPECT(count.ok());
  PRESTO_EXPECT(count.value == 24);
}

PRESTO_TEST(runtime_tensor_rejects_element_count_overflow) {
  const auto count = checked_element_count({std::numeric_limits<std::uint64_t>::max(), 2});
  PRESTO_EXPECT(!count.ok());
  PRESTO_EXPECT(count.status.code == StatusCode::Overflow);
}

PRESTO_TEST(runtime_tensor_rejects_zero_sized_dimensions) {
  const auto count = checked_element_count({2, 0, 4});
  PRESTO_EXPECT(!count.ok());
  PRESTO_EXPECT(count.status.code == StatusCode::InvalidArgument);
}

PRESTO_TEST(runtime_tensor_computes_scalar_and_quantized_sizes) {
  const auto f32 = checked_byte_size(DataType::F32, {2, 3});
  const auto q4 = checked_byte_size(DataType::Q4_0, {64});
  const auto q8 = checked_byte_size(DataType::Q8_0, {64});
  PRESTO_EXPECT(f32.ok() && f32.value == 24);
  PRESTO_EXPECT(q4.ok() && q4.value == 36);
  PRESTO_EXPECT(q8.ok() && q8.value == 68);
}

PRESTO_TEST(runtime_tensor_rejects_partial_quantization_blocks) {
  const auto size = checked_byte_size(DataType::Q4_0, {33});
  PRESTO_EXPECT(!size.ok());
  PRESTO_EXPECT(size.status.code == StatusCode::InvalidArgument);
}

PRESTO_TEST(runtime_tensor_validates_layout_rank_and_strides) {
  TensorLayout layout{DataType::F32, {2, 3}, {12, 4}};
  PRESTO_EXPECT(validate_layout(layout).ok());
  layout.byte_strides = {12};
  PRESTO_EXPECT(validate_layout(layout).code == StatusCode::InvalidArgument);
  layout.byte_strides = {4, 4};
  PRESTO_EXPECT(validate_layout(layout).code == StatusCode::InvalidArgument);
}
