#include "presto/runtime/context.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <limits>

using namespace presto::runtime;

PRESTO_TEST(runtime_context_reports_per_sequence_capacity) {
  const auto capacity = context_capacity(2048, 4);
  PRESTO_EXPECT(capacity.ok());
  PRESTO_EXPECT(capacity.value.total_tokens == 2048);
  PRESTO_EXPECT(capacity.value.per_sequence_tokens == 512);
}

PRESTO_TEST(runtime_context_grows_for_logical_sequence_capacity) {
  const auto capacity = grow_context_capacity(2048, 4, 700);
  PRESTO_EXPECT(capacity.ok());
  PRESTO_EXPECT(capacity.value.total_tokens == 4096);
  PRESTO_EXPECT(capacity.value.per_sequence_tokens == 1024);
}

PRESTO_TEST(runtime_context_rejects_zero_slots_and_overflow) {
  PRESTO_EXPECT(context_capacity(2048, 0).status.code == StatusCode::InvalidArgument);
  const auto overflow = grow_context_capacity(
      std::numeric_limits<std::uint32_t>::max(), 2,
      std::numeric_limits<std::uint32_t>::max());
  PRESTO_EXPECT(!overflow.ok());
  PRESTO_EXPECT(overflow.status.code == StatusCode::Overflow);
}
