#pragma once

#include "presto/runtime/status.hpp"

#include <cstdint>

namespace presto::runtime {

struct ContextCapacity {
  std::uint32_t total_tokens = 0;
  std::uint32_t per_sequence_tokens = 0;
  std::uint32_t sequence_count = 0;
};

Result<ContextCapacity> context_capacity(std::uint32_t total_tokens,
                                         std::uint32_t sequence_count);
Result<ContextCapacity> grow_context_capacity(std::uint32_t current_total_tokens,
                                              std::uint32_t sequence_count,
                                              std::uint32_t required_per_sequence_tokens);

} // namespace presto::runtime
