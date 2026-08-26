#pragma once

#include "presto/runtime/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace presto::runtime {

class Buffer {
public:
  virtual ~Buffer() = default;
  [[nodiscard]] virtual void* data() = 0;
  [[nodiscard]] virtual const void* data() const = 0;
  [[nodiscard]] virtual std::uint64_t size_bytes() const = 0;
};

class HostBuffer final : public Buffer {
public:
  ~HostBuffer() override;
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;

  static Result<std::shared_ptr<HostBuffer>> allocate(std::uint64_t size_bytes,
                                                       std::size_t alignment = 64);

  [[nodiscard]] void* data() override { return data_; }
  [[nodiscard]] const void* data() const override { return data_; }
  [[nodiscard]] std::uint64_t size_bytes() const override { return size_bytes_; }

private:
  HostBuffer(void* data, std::uint64_t size_bytes, std::size_t alignment)
      : data_(data), size_bytes_(size_bytes), alignment_(alignment) {}

  void* data_ = nullptr;
  std::uint64_t size_bytes_ = 0;
  std::size_t alignment_ = 0;
};

class TensorView {
public:
  TensorView() = default;

  static Result<TensorView> create(std::shared_ptr<Buffer> buffer,
                                   std::uint64_t offset_bytes,
                                   TensorLayout layout);

  [[nodiscard]] void* data() const;
  [[nodiscard]] std::uint64_t offset_bytes() const { return offset_bytes_; }
  [[nodiscard]] const TensorLayout& layout() const { return layout_; }
  [[nodiscard]] const std::shared_ptr<Buffer>& buffer() const { return buffer_; }

private:
  std::shared_ptr<Buffer> buffer_;
  std::uint64_t offset_bytes_ = 0;
  TensorLayout layout_;
};

} // namespace presto::runtime
