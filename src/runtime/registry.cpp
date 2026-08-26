#include "presto/runtime/registry.hpp"

namespace presto::runtime {

Result<std::shared_ptr<Buffer>> HostDevice::allocate(const std::uint64_t size_bytes,
                                                      const std::size_t alignment) {
  auto host = HostBuffer::allocate(size_bytes, alignment);
  if (!host.ok()) return Result<std::shared_ptr<Buffer>>::failure(std::move(host.status));
  return Result<std::shared_ptr<Buffer>>::success(std::move(host.value));
}

Status ProviderRegistry::add(std::shared_ptr<Device> device) {
  if (!device || device->info().id.empty()) {
    return Status::error(StatusCode::InvalidArgument, "registry_add",
                         "device and stable id are required");
  }
  const std::string id = device->info().id;
  if (devices_.find(id) != devices_.end()) {
    return Status::error(StatusCode::AlreadyExists, "registry_add",
                         "device stable id already exists: " + id);
  }
  devices_.emplace(id, std::move(device));
  return Status::success();
}

std::vector<DeviceInfo> ProviderRegistry::devices() const {
  std::vector<DeviceInfo> result;
  result.reserve(devices_.size());
  for (const auto& [id, device] : devices_) {
    (void)id;
    result.push_back(device->info());
  }
  return result;
}

Result<std::shared_ptr<Device>> ProviderRegistry::find(const std::string& stable_id) const {
  const auto it = devices_.find(stable_id);
  if (it == devices_.end()) {
    return Result<std::shared_ptr<Device>>::failure(
        Status::error(StatusCode::NotFound, "registry_find", "device not found: " + stable_id));
  }
  return Result<std::shared_ptr<Device>>::success(it->second);
}

} // namespace presto::runtime
