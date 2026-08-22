// presto tests - GGUF metadata parser (synthetic buffers per ggml gguf spec)
#include "test_util.hpp"

#include "presto/gguf_meta.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace presto::testing {
namespace {

void put_u32(std::vector<unsigned char>& v, std::uint32_t x) {
  for (int i = 0; i < 4; ++i) v.push_back(static_cast<unsigned char>((x >> (8 * i)) & 0xFF));
}
void put_u64(std::vector<unsigned char>& v, std::uint64_t x) {
  for (int i = 0; i < 8; ++i) v.push_back(static_cast<unsigned char>((x >> (8 * i)) & 0xFF));
}
void put_str(std::vector<unsigned char>& v, bool v1, const std::string& s) {
  if (v1)
    put_u32(v, static_cast<std::uint32_t>(s.size()));
  else
    put_u64(v, s.size());
  for (char c : s) v.push_back(static_cast<unsigned char>(c));
}
void put_kv_string(std::vector<unsigned char>& v, bool v1, const std::string& key,
                   const std::string& val) {
  put_str(v, v1, key);
  put_u32(v, 8);  // GGUF_TYPE_STRING
  put_str(v, v1, val);
}
void write_file(const std::filesystem::path& p, const std::vector<unsigned char>& bytes) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
}

std::vector<unsigned char> make_v3_buffer() {
  std::vector<unsigned char> b;
  b.insert(b.end(), {'G', 'G', 'U', 'F'});
  put_u32(b, 3);   // version
  put_u64(b, 0);   // tensor_count
  put_u64(b, 4);   // kv_count
  put_kv_string(b, false, "general.architecture", "llama");
  put_kv_string(b, false, "general.name", "presto-test");
  // general.file_type: UINT32 = 12 (Q4_K)
  put_str(b, false, "general.file_type");
  put_u32(b, 4);   // GGUF_TYPE_UINT32
  put_u32(b, 12);
  // array kv: UINT32 [7,8,9]
  put_str(b, false, "arr.test");
  put_u32(b, 9);   // GGUF_TYPE_ARRAY
  put_u32(b, 4);   // elem type UINT32
  put_u64(b, 3);
  put_u32(b, 7);
  put_u32(b, 8);
  put_u32(b, 9);
  return b;
}

PRESTO_TEST(gguf_v3_header_parses) {
  const auto dir = std::filesystem::temp_directory_path();
  const auto path = dir / "presto_test_model_v3.gguf";
  write_file(path, make_v3_buffer());

  GgufInfo info;
  std::string err;
  PRESTO_EXPECT(parse_gguf_header(path.string(), info, err));
  PRESTO_EXPECT_STR_EQ(err.empty() ? "" : err.c_str(), "");
  PRESTO_EXPECT(info.version == 3);
  PRESTO_EXPECT(info.tensor_count == 0);
  PRESTO_EXPECT(info.kv_count == 4);
  PRESTO_EXPECT(info.architecture == "llama");
  PRESTO_EXPECT(info.name == "presto-test");
  PRESTO_EXPECT(info.file_type == 12);
  const auto meta = info.to_meta();
  PRESTO_EXPECT(meta.at("arch") == "llama");
  PRESTO_EXPECT(meta.at("quant") == "Q4_K");

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

PRESTO_TEST(gguf_truncated_rejected) {
  const auto dir = std::filesystem::temp_directory_path();
  const auto path = dir / "presto_test_trunc.gguf";
  auto buf = make_v3_buffer();
  buf.resize(20);  // cut inside kvs
  write_file(path, buf);

  GgufInfo info;
  std::string err;
  PRESTO_EXPECT(!parse_gguf_header(path.string(), info, err));
  PRESTO_EXPECT(!err.empty());

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

PRESTO_TEST(gguf_wrong_version_rejected) {
  const auto dir = std::filesystem::temp_directory_path();
  const auto path = dir / "presto_test_badver.gguf";
  std::vector<unsigned char> b;
  b.insert(b.end(), {'G', 'G', 'U', 'F'});
  put_u32(b, 99);
  put_u64(b, 0);
  put_u64(b, 0);
  write_file(path, b);

  GgufInfo info;
  std::string err;
  PRESTO_EXPECT(!parse_gguf_header(path.string(), info, err));

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

} // namespace
} // namespace presto::testing
