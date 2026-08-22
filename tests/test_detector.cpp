// presto tests - format detection (synthetic fixtures)
#include "test_util.hpp"

#include "presto/detector.hpp"
#include "presto/format.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace presto::testing {
namespace {

std::filesystem::path make_fixture_dir() {
  const auto base = std::filesystem::temp_directory_path() /
                     ("presto_det_" + std::to_string(std::rand()));
  std::filesystem::create_directories(base);
  return base;
}

void write_bytes(const std::filesystem::path& p, const std::vector<unsigned char>& b) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

void write_text(const std::filesystem::path& p, const std::string& s) {
  write_bytes(p, std::vector<unsigned char>(s.begin(), s.end()));
}

std::vector<unsigned char> minimal_gguf_v3() {
  std::vector<unsigned char> b{'G', 'G', 'U', 'F'};
  auto u32 = [&b](std::uint32_t x) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<unsigned char>((x >> (8 * i)) & 0xFF));
  };
  auto u64 = [&b](std::uint64_t x) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<unsigned char>((x >> (8 * i)) & 0xFF));
  };
  u32(3);   // version
  u64(0);   // tensor_count
  u64(0);   // kv_count
  return b;
}

std::vector<unsigned char> minimal_safetensors() {
  const std::string header =
      R"({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
  std::vector<unsigned char> b;
  for (int i = 0; i < 8; ++i)
    b.push_back(static_cast<unsigned char>((header.size() >> (8 * i)) & 0xFF));
  for (char c : header) b.push_back(static_cast<unsigned char>(c));
  b.resize(b.size() + 8, 0);
  return b;
}

PRESTO_TEST(detect_gguf_file) {
  const auto dir = make_fixture_dir();
  const auto p = dir / "model.gguf";
  write_bytes(p, minimal_gguf_v3());
  const Detection d = detect_format(p.string());
  PRESTO_EXPECT(d.format == presto::ModelFormat::GGUF);
  PRESTO_EXPECT(d.meta.at("gguf_version") == "3");
  std::filesystem::remove_all(dir);
}

PRESTO_TEST(detect_safetensors_file) {
  const auto dir = make_fixture_dir();
  const auto p = dir / "weights.safetensors";
  write_bytes(p, minimal_safetensors());
  const Detection d = detect_format(p.string());
  PRESTO_EXPECT(d.format == presto::ModelFormat::SAFETENSORS_FILE);
  PRESTO_EXPECT(d.meta.at("tensor_count") == "1");
  std::filesystem::remove_all(dir);
}

PRESTO_TEST(detect_pytorch_zip) {
  const auto dir = make_fixture_dir();
  const auto p = dir / "ckpt.pt";
  write_bytes(p, {'P', 'K', 0x03, 0x04, 1, 2, 3});
  const Detection d = detect_format(p.string());
  PRESTO_EXPECT(d.format == presto::ModelFormat::PYTORCH_FILE);
  PRESTO_EXPECT(d.meta.at("container") == "zip");
  std::filesystem::remove_all(dir);
}

PRESTO_TEST(detect_awq_dir) {
  const auto dir = make_fixture_dir();
  write_text(dir / "config.json",
             R"({"model_type":"llama","quantization_config":{"quant_method":"awq","bits":4,"group_size":128}})");
  const Detection d = detect_format(dir.string());
  PRESTO_EXPECT(d.format == presto::ModelFormat::HF_DIR_AWQ);
  PRESTO_EXPECT(d.meta.at("bits") == "4");
  PRESTO_EXPECT(d.meta.at("group_size") == "128");
  std::filesystem::remove_all(dir);
}

PRESTO_TEST(detect_gptq_dir) {
  const auto dir = make_fixture_dir();
  write_text(dir / "config.json",
             R"({"model_type":"llama","quantization_config":{"quant_method":"gptq","bits":8,"group_size":32}})");
  const Detection d = detect_format(dir.string());
  PRESTO_EXPECT(d.format == presto::ModelFormat::HF_DIR_GPTQ);
  PRESTO_EXPECT(d.meta.at("bits") == "8");
  std::filesystem::remove_all(dir);
}

PRESTO_TEST(detect_mlx_dir) {
  const auto dir = make_fixture_dir();
  write_text(dir / "config.json",
             R"({"model_type":"llama","quantization":{"group_size":64,"bits":4}})");
  const Detection d = detect_format(dir.string());
  PRESTO_EXPECT(d.format == presto::ModelFormat::MLX_DIR);
  PRESTO_EXPECT(d.meta.at("mlx_bits") == "4");
  PRESTO_EXPECT(d.meta.at("mlx_group_size") == "64");
  std::filesystem::remove_all(dir);
}

PRESTO_TEST(detect_plain_hf_dir_is_unknown) {
  const auto dir = make_fixture_dir();
  write_text(dir / "config.json", R"({"model_type":"llama","hidden_size":4096})");
  const Detection d = detect_format(dir.string());
  PRESTO_EXPECT(d.format == presto::ModelFormat::UNKNOWN);
  PRESTO_EXPECT(d.summary.find("HF-style") != std::string::npos);
  std::filesystem::remove_all(dir);
}

PRESTO_TEST(detect_never_crashes_on_garbage) {
  const auto dir = make_fixture_dir();
  // truncated gguf
  write_bytes(dir / "trunc.gguf", {'G', 'G'});
  PRESTO_EXPECT(detect_format((dir / "trunc.gguf").string()).format ==
                presto::ModelFormat::UNKNOWN);
  // garbage file with safetensors extension
  write_bytes(dir / "junk.safetensors", {1, 2, 3});
  PRESTO_EXPECT(detect_format((dir / "junk.safetensors").string()).format ==
                presto::ModelFormat::UNKNOWN);
  // invalid config json
  write_text(dir / "badcfg" / "", "");  // ensure parent creation via dirs below
  std::filesystem::create_directories(dir / "badcfg");
  write_text(dir / "badcfg" / "config.json", "{not json");
  PRESTO_EXPECT(detect_format((dir / "badcfg").string()).format ==
                presto::ModelFormat::UNKNOWN);
  // nonexistent path
  PRESTO_EXPECT(detect_format((dir / "nope.bin").string()).format ==
                presto::ModelFormat::UNKNOWN);
  std::filesystem::remove_all(dir);
}

} // namespace
} // namespace presto::testing
