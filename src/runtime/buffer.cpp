#include "presto/runtime/buffer.hpp"

#include <cstring>
#include <limits>
#include <new>

namespace presto::runtime {
namespace {

bool add_overflows(const std::uint64_t a, const std::uint64_t b) {
  return a > std::numeric_limits<std::uint64_t>::max() - b;
}

bool multiply_overflows(const std::uint64_t a, const std::uint64_t b) {
  return b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b;
}

bool valid_alignment(const std::size_t alignment) {
  return alignment >= alignof(void*) && (alignment & (alignment - 1)) == 0;
}

} // namespace

HostBuffer::~HostBuffer() {
  if (data_) ::operator delete(data_, std::align_val_t(alignment_));
}

Result<std::shared_ptr<HostBuffer>> HostBuffer::allocate(const std::uint64_t size_bytes,
                                                         const std::size_t alignment) {
  if (size_bytes == 0 || !valid_alignment(alignment) ||
      size_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Result<std::shared_ptr<HostBuffer>>::failure(Status::error(
        StatusCode::InvalidArgument, "host_allocate", "invalid size or alignment"));
  }
  void* data = ::operator new(static_cast<std::size_t>(size_bytes),
                              std::align_val_t(alignment), std::nothrow);
  if (!data) {
    return Result<std::shared_ptr<HostBuffer>>::failure(
        Status::error(StatusCode::AllocationFailed, "host_allocate", "allocation failed"));
  }
  std::memset(data, 0, static_cast<std::size_t>(size_bytes));
  return Result<std::shared_ptr<HostBuffer>>::success(
      std::shared_ptr<HostBuffer>(new HostBuffer(data, size_bytes, alignment)));
}

Result<TensorView> TensorView::create(std::shared_ptr<Buffer> buffer,
                                      const std::uint64_t offset_bytes,
                                      TensorLayout layout) {
  if (!buffer) {
    return Result<TensorView>::failure(
        Status::error(StatusCode::InvalidArgument, "tensor_view", "buffer is null"));
  }
  const Status layout_status = validate_layout(layout);
  if (!layout_status.ok()) return Result<TensorView>::failure(layout_status);

  std::uint64_t extent = layout.dtype == DataType::F32 ? 4 :
                         (layout.dtype == DataType::F16 || layout.dtype == DataType::BF16 ? 2 : 1);
  for (std::size_t i = 0; i < layout.shape.size(); ++i) {
    const std::uint64_t steps = layout.shape[i] - 1;
    if (multiply_overflows(steps, layout.byte_strides[i])) {
      return Result<TensorView>::failure(
          Status::error(StatusCode::Overflow, "tensor_view", "view extent overflows uint64"));
    }
    const std::uint64_t contribution = steps * layout.byte_strides[i];
    if (add_overflows(extent, contribution)) {
      return Result<TensorView>::failure(
          Status::error(StatusCode::Overflow, "tensor_view", "view extent overflows uint64"));
    }
    extent += contribution;
  }
  if (add_overflows(offset_bytes, extent)) {
    return Result<TensorView>::failure(
        Status::error(StatusCode::Overflow, "tensor_view", "offset plus extent overflows uint64"));
  }
  if (offset_bytes + extent > buffer->size_bytes()) {
    return Result<TensorView>::failure(
        Status::error(StatusCode::InvalidArgument, "tensor_view", "view exceeds buffer bounds"));
  }

  TensorView view;
  view.buffer_ = std::move(buffer);
  view.offset_bytes_ = offset_bytes;
  view.layout_ = std::move(layout);
  return Result<TensorView>::success(std::move(view));
}

void* TensorView::data() const {
  if (!buffer_) return nullptr;
  return static_cast<std::byte*>(buffer_->data()) + offset_bytes_;
}

} // namespace presto::runtime
