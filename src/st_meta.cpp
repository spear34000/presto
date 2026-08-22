// presto - safetensors header parser (pure C++)
#include "presto/st_meta.hpp"

#include "json_mini.hpp"

#include <algorithm>
#include <fstream>

namespace presto {
namespace {

template <typename T>
T read_le(const unsigned char* b) {
  T v{};
  for (std::size_t i = 0; i < sizeof(T); ++i)
    v |= static_cast<T>(b[i]) << (8 * i);
  return v;
}

} // namespace

std::map<std::string, std::string> StInfo::to_meta() const {
  std::map<std::string, std::string> m;
  m["tensor_count"] = std::to_string(tensor_count);
  m["data_bytes"] = std::to_string(data_bytes);
  std::string dt;
  for (const auto& [k, v] : dtypes) {
    if (!dt.empty()) dt += ",";
    dt += k + ":" + std::to_string(v);
  }
  if (!dt.empty()) m["dtypes"] = dt;
  return m;
}

bool parse_st_header(const std::string& path, StInfo& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open file";
    return false;
  }
  unsigned char hb[8];
  f.read(reinterpret_cast<char*>(hb), 8);
  if (f.gcount() != 8) {
    err = "file too small";
    return false;
  }
  const std::uint64_t header_len = read_le<std::uint64_t>(hb);
  if (header_len == 0 || header_len > (256ull << 20)) {
    err = "header length out of range";
    return false;
  }
  out.header_len = header_len;

  std::string text(header_len, '\0');
  f.read(text.data(), static_cast<std::streamsize>(header_len));
  if (static_cast<std::uint64_t>(f.gcount()) != header_len) {
    err = "truncated header";
    return false;
  }

  json::Node root;
  if (!json::parse(text, root, err) || !root.is_object()) {
    if (err.empty()) err = "header is not a JSON object";
    return false;
  }

  for (const auto& [name, node] : root.members()) {
    if (name == "__metadata__") continue;
    if (!node.is_object()) {
      err = "tensor entry not an object: " + name;
      return false;
    }
    const json::Node* dt = node.find("dtype");
    if (!dt || !dt->is_string()) {
      err = "missing dtype for tensor: " + name;
      return false;
    }
    ++out.dtypes[dt->as_string()];
    ++out.tensor_count;

    if (const json::Node* off = node.find("data_offsets")) {
      if (off->is_array() && off->items().size() == 2) {
        const std::uint64_t end =
            static_cast<std::uint64_t>(off->items()[1].as_int());
        out.data_bytes = std::max(out.data_bytes, end);
      }
    }
  }
  return true;
}

} // namespace presto
