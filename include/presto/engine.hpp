// presto - engine facade + backend registry API
#pragma once

#include "presto/backend.hpp"
#include "presto/format.hpp"

#include <memory>
#include <string>

namespace presto {

struct BackendCaps {
  bool llamacpp = false;
  bool mlx = false;
};

// Compile-time capability report (what this binary can actually execute).
BackendCaps backend_caps();

// Choose a compiled backend able to execute the detected model.
// Returns nullptr (with err set) when nothing suitable is compiled-in;
// err always carries an actionable hint.
std::unique_ptr<IBackend> select_backend(const Detection& d, std::string& err);

} // namespace presto
