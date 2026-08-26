#pragma once

#include "presto/runtime/device.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace presto::runtime {

class ProviderRegistry {
public:
  Status add(std::shared_ptr<Device> device);
  [[nodiscard]] std::vector<DeviceInfo> devices() const;
  [[nodiscard]] Result<std::shared_ptr<Device>> find(const std::string& stable_id) const;

private:
  std::map<std::string, std::shared_ptr<Device>> devices_;
};

} // namespace presto::runtime
