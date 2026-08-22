// presto - OpenAI-compatible HTTP server API
#pragma once

#include "presto/format.hpp"

#include <string>

namespace presto {

// Blocks serving requests until killed. Returns process exit code.
int run_openai_server(const Detection& d, const std::string& host, int port);

} // namespace presto
