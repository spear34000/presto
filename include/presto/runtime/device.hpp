#pragma once

#include "presto/runtime/buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace presto::runtime {

enum class DeviceKind {
  Host,
  Vulkan,
  Cuda,
  Sycl,
  Hip,
};

struct DeviceInfo {
  std::string id;
  std::string name;
  DeviceKind kind = DeviceKind::Host;
  std::uint64_t total_memory_bytes = 0;
};

class Device {
public:
  virtual ~Device() = default;
  [[nodiscard]] virtual const DeviceInfo& info() const = 0;
  virtual Result<std::shared_ptr<Buffer>> allocate(std::uint64_t size_bytes,
                                                    std::size_t alignment = 64) = 0;
};

class HostDevice final : public Device {
public:
  HostDevice(std::string id, std::string name)
      : info_{std::move(id), std::move(name), DeviceKind::Host, 0} {}

  [[nodiscard]] const DeviceInfo& info() const override { return info_; }
  Result<std::shared_ptr<Buffer>> allocate(std::uint64_t size_bytes,
                                           std::size_t alignment = 64) override;

private:
  DeviceInfo info_;
};

} // namespace presto::runtime
