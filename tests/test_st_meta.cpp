// presto tests - safetensors header parser (synthetic archives)
#include "test_util.hpp"

#include "presto/st_meta.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace presto::testing {
namespace {

void put_u64(std::vector<unsigned char>& v, std::uint64_t x) {
  for (int i = 0; i < 8; ++i) v.push_back(static_cast<unsigned char>((x >> (8 * i)) & 0xFF));
}

std::vector<unsigned char> make_st_buffer(const std::string& header,
                                          std::size_t data_bytes) {
  std::vector<unsigned char> b;
  put_u64(b, header.size());
  for (char c : header) b.push_back(static_cast<unsigned char>(c));
  b.resize(b.size() + data_bytes, 0);
  return b;
}

void write_file(const std::filesystem::path& p, const std::vector<unsigned char>& bytes) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
}

PRESTO_TEST(st_header_parses) {
  const auto dir = std::filesystem::temp_directory_path();
  const auto path = dir / "presto_test_model.safetensors";
  const std::string header =
      R"({"w1":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
      R"("w2":{"dtype":"F16","shape":[3],"data_offsets":[16,22]},"__metadata__":{"fmt":"presto"}})";
  write_file(path, make_st_buffer(header, 32));

  StInfo st;
  std::string err;
  PRESTO_EXPECT(parse_st_header(path.string(), st, err));
  if (!err.empty()) std::fprintf(stderr, "  parse error: %s\n", err.c_str());
  PRESTO_EXPECT(st.tensor_count == 2);   // __metadata__ excluded
  PRESTO_EXPECT(st.dtypes.at("F32") == 1);
  PRESTO_EXPECT(st.dtypes.at("F16") == 1);
  PRESTO_EXPECT(st.data_bytes == 22);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

PRESTO_TEST(st_truncated_rejected) {
  const auto dir = std::filesystem::temp_directory_path();
  const auto path = dir / "presto_test_bad.safetensors";
  // claims a 100-byte header but provides only 10 bytes -> must be rejected
  std::vector<unsigned char> b;
  put_u64(b, 100);
  for (int i = 0; i < 10; ++i) b.push_back('{');
  write_file(path, b);

  StInfo st;
  std::string err;
  PRESTO_EXPECT(!parse_st_header(path.string(), st, err));
  PRESTO_EXPECT(!err.empty());

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

} // namespace
} // namespace presto::testing
