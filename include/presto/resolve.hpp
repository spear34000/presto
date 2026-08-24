// presto - model path resolution (ollama-style name lookup)
#pragma once

#include <string>
#include <vector>

namespace presto {

// Resolve a user-supplied model reference to an existing file path.
// Search order:
//   1. as-given (exact path)
//   2. ./models/<ref>
//   3. recursive scan of known library roots (LM Studio, presto dir)
//      matching: exact filename, <ref>.gguf, or <ref>*.gguf prefix
// Returns empty string when nothing matches.
std::string resolve_model_path(const std::string& ref);

// All model files discovered in the known roots (for `presto models`).
std::vector<std::string> list_known_models();

} // namespace presto
