// presto - GGUF header/metadata parser API (pure C++, no external deps)
// Reads ONLY the header + metadata key/values; never tensor payloads.
#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace presto {

struct GgufInfo {
  std::uint32_t version = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t kv_count = 0;
  std::int64_t file_type = -1;      // general.file_type when present
  std::string architecture;         // general.architecture when present
  std::string name;                 // general.name when present
  std::map<std::string, std::string> kvs;  // scalar/string values flattened;
                                           // arrays summarized as "<array:N>"

  std::map<std::string, std::string> to_meta() const;
};

// Human-readable name for the GGML "general.file_type" enum value.
const char* ggml_ftype_name(std::int64_t ft);

bool parse_gguf_header(const std::string& path, GgufInfo& out, std::string& err);

} // namespace presto
