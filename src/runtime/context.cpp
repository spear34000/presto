#include "presto/runtime/context.hpp"

#include <limits>

namespace presto::runtime {

Result<ContextCapacity> context_capacity(const std::uint32_t total_tokens,
                                         const std::uint32_t sequence_count) {
  if (total_tokens == 0 || sequence_count == 0) {
    return Result<ContextCapacity>::failure(Status::error(
        StatusCode::InvalidArgument, "context_capacity",
        "total token capacity and sequence count must be non-zero"));
  }
  if (total_tokens < sequence_count) {
    return Result<ContextCapacity>::failure(Status::error(
        StatusCode::InvalidArgument, "context_capacity",
        "total token capacity must provide at least one token per sequence"));
  }
  return Result<ContextCapacity>::success(
      {total_tokens, total_tokens / sequence_count, sequence_count});
}

Result<ContextCapacity> grow_context_capacity(
    const std::uint32_t current_total_tokens, const std::uint32_t sequence_count,
    const std::uint32_t required_per_sequence_tokens) {
  if (required_per_sequence_tokens == 0) {
    return Result<ContextCapacity>::failure(Status::error(
        StatusCode::InvalidArgument, "grow_context_capacity",
        "required per-sequence capacity must be non-zero"));
  }
  auto capacity = context_capacity(current_total_tokens, sequence_count);
  if (!capacity.ok()) return capacity;

  while (capacity.value.per_sequence_tokens < required_per_sequence_tokens) {
    if (capacity.value.total_tokens > std::numeric_limits<std::uint32_t>::max() / 2) {
      return Result<ContextCapacity>::failure(Status::error(
          StatusCode::Overflow, "grow_context_capacity",
          "context pool cannot grow without overflowing uint32"));
    }
    capacity = context_capacity(capacity.value.total_tokens * 2, sequence_count);
  }
  return capacity;
}

} // namespace presto::runtime
