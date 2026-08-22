// presto - format detection API
#pragma once

#include "presto/format.hpp"

namespace presto {

// Inspect a filesystem path (file or directory) and classify the model format.
// Never throws; malformed input yields UNKNOWN with an explanatory summary.
Detection detect_format(const std::string& path);

} // namespace presto
