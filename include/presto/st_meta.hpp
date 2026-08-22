// presto - safetensors header parser API
// Reads ONLY the JSON header of a safetensors archive (never tensor payloads).
#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace presto {

struct StInfo {
  std::uint64_t header_len = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t data_bytes = 0;         // highest data_offsets[1]
  std::map<std::string, int> dtypes;    // dtype -> histogram

  std::map<std::string, std::string> to_meta() const;
};

bool parse_st_header(const std::string& path, StInfo& out, std::string& err);

} // namespace presto
