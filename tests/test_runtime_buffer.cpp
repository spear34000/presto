#include "presto/runtime/buffer.hpp"
#include "test_util.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

using namespace presto::runtime;

PRESTO_TEST(runtime_host_buffer_is_aligned_and_zero_initialized) {
  const auto buffer = HostBuffer::allocate(128, 64);
  PRESTO_EXPECT(buffer.ok());
  PRESTO_EXPECT(buffer.value->size_bytes() == 128);
  PRESTO_EXPECT(reinterpret_cast<std::uintptr_t>(buffer.value->data()) % 64 == 0);
  bool all_zero = true;
  for (std::size_t i = 0; i < 128; ++i)
    all_zero = all_zero && static_cast<std::byte*>(buffer.value->data())[i] == std::byte{0};
  PRESTO_EXPECT(all_zero);
}

PRESTO_TEST(runtime_tensor_view_accepts_bounded_storage) {
  const auto buffer = HostBuffer::allocate(128);
  TensorLayout layout{DataType::F32, {2, 3}, {12, 4}};
  const auto view = TensorView::create(buffer.value, 16, layout);
  PRESTO_EXPECT(view.ok());
  PRESTO_EXPECT(view.value.offset_bytes() == 16);
  PRESTO_EXPECT(view.value.data() == static_cast<std::byte*>(buffer.value->data()) + 16);
}

PRESTO_TEST(runtime_tensor_view_rejects_out_of_bounds_extent) {
  const auto buffer = HostBuffer::allocate(32);
  TensorLayout layout{DataType::F32, {2, 3}, {12, 4}};
  const auto view = TensorView::create(buffer.value, 16, layout);
  PRESTO_EXPECT(!view.ok());
  PRESTO_EXPECT(view.status.code == StatusCode::InvalidArgument);
}

PRESTO_TEST(runtime_tensor_view_rejects_overflowing_offset) {
  const auto buffer = HostBuffer::allocate(32);
  TensorLayout layout{DataType::F32, {1}, {4}};
  const auto view = TensorView::create(buffer.value, std::numeric_limits<std::uint64_t>::max(), layout);
  PRESTO_EXPECT(!view.ok());
  PRESTO_EXPECT(view.status.code == StatusCode::Overflow);
}
