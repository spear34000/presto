#pragma once

#include <string>
#include <utility>

namespace presto::runtime {

enum class StatusCode {
  Ok = 0,
  InvalidArgument,
  Overflow,
  Unsupported,
  NotFound,
  AlreadyExists,
  AllocationFailed,
  Internal,
};

struct Status {
  StatusCode code = StatusCode::Ok;
  std::string operation;
  std::string message;

  [[nodiscard]] bool ok() const { return code == StatusCode::Ok; }

  static Status success() { return {}; }
  static Status error(StatusCode code, std::string operation, std::string message) {
    return {code, std::move(operation), std::move(message)};
  }
};

template <typename T>
struct Result {
  Status status;
  T value{};

  [[nodiscard]] bool ok() const { return status.ok(); }

  static Result success(T value) { return {Status::success(), std::move(value)}; }
  static Result failure(Status status) { return {std::move(status), {}}; }
};

} // namespace presto::runtime
