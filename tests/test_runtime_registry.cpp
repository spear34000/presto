#include "presto/runtime/registry.hpp"
#include "test_util.hpp"

#include <memory>

using namespace presto::runtime;

PRESTO_TEST(runtime_registry_enumerates_host_device) {
  ProviderRegistry registry;
  PRESTO_EXPECT(registry.add(std::make_shared<HostDevice>("host:0", "Native CPU")).ok());
  const auto devices = registry.devices();
  PRESTO_EXPECT(devices.size() == 1);
  PRESTO_EXPECT(devices[0].id == "host:0");
  PRESTO_EXPECT(devices[0].kind == DeviceKind::Host);
}

PRESTO_TEST(runtime_registry_rejects_duplicate_stable_ids) {
  ProviderRegistry registry;
  PRESTO_EXPECT(registry.add(std::make_shared<HostDevice>("host:0", "CPU A")).ok());
  const Status duplicate = registry.add(std::make_shared<HostDevice>("host:0", "CPU B"));
  PRESTO_EXPECT(duplicate.code == StatusCode::AlreadyExists);
}

PRESTO_TEST(runtime_registry_orders_devices_by_stable_id) {
  ProviderRegistry registry;
  PRESTO_EXPECT(registry.add(std::make_shared<HostDevice>("host:2", "CPU 2")).ok());
  PRESTO_EXPECT(registry.add(std::make_shared<HostDevice>("host:0", "CPU 0")).ok());
  const auto devices = registry.devices();
  PRESTO_EXPECT(devices.size() == 2);
  PRESTO_EXPECT(devices[0].id == "host:0");
  PRESTO_EXPECT(devices[1].id == "host:2");
}

PRESTO_TEST(runtime_registry_reports_missing_device) {
  ProviderRegistry registry;
  const auto missing = registry.find("cuda:0");
  PRESTO_EXPECT(!missing.ok());
  PRESTO_EXPECT(missing.status.code == StatusCode::NotFound);
}

PRESTO_TEST(runtime_host_device_allocates_through_common_contract) {
  HostDevice device("host:0", "Native CPU");
  const auto buffer = device.allocate(256, 64);
  PRESTO_EXPECT(buffer.ok());
  PRESTO_EXPECT(buffer.value->size_bytes() == 256);
}
