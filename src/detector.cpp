// presto - format detection implementation (pure stdlib)
#include "presto/detector.hpp"

#include "json_mini.hpp"
#include "presto/gguf_meta.hpp"
#include "presto/st_meta.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace presto {
namespace {

bool read_prefix(const fs::path& p, std::size_t n, std::string& out) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return false;
  out.resize(n);
  f.read(out.data(), static_cast<std::streamsize>(n));
  out.resize(static_cast<std::size_t>(f.gcount()));
  return true;
}

bool ends_with_any(const std::string& s, std::initializer_list<const char*> exts) {
  for (const char* e : exts) {
    const std::string ext(e);
    if (s.size() >= ext.size() && s.compare(s.size() - ext.size(), ext.size(), ext) == 0) return true;
  }
  return false;
}

Detection unknown_det(const std::string& path, const std::string& why) {
  Detection d;
  d.format = ModelFormat::UNKNOWN;
  d.path = path;
  d.summary = why;
  return d;
}

Detection detect_file(const std::string& path) {
  std::string head;
  if (!read_prefix(fs::path(path), 8, head)) {
    return unknown_det(path, "cannot read file");
  }

  if (head.size() >= 4 && head.compare(0, 4, "GGUF") == 0) {
    GgufInfo info;
    std::string err;
    if (!parse_gguf_header(path, info, err)) {
      Detection d = unknown_det(path, "GGUF magic ok but header parse failed: " + err);
      d.meta["gguf_version"] = std::to_string(info.version);
      return d;
    }
    Detection d;
    d.format = ModelFormat::GGUF;
    d.path = path;
    d.summary = "GGUF v" + std::to_string(info.version) + ", " +
                std::to_string(info.tensor_count) + " tensors" +
                (info.architecture.empty() ? "" : ", arch=" + info.architecture);
    d.meta = info.to_meta();
    return d;
  }

  if (head.size() >= 2 &&
      static_cast<unsigned char>(head[0]) == 0x50 /*P*/ &&
      static_cast<unsigned char>(head[1]) == 0x4B /*K*/ &&
      ends_with_any(path, {".pt", ".pth", ".ckpt"})) {
    Detection d;
    d.format = ModelFormat::PYTORCH_FILE;
    d.path = path;
    d.summary = "PyTorch checkpoint (zip container)";
    d.meta["container"] = "zip";
    return d;
  }

  if (ends_with_any(path, {".safetensors"})) {
    StInfo st;
    std::string err;
    if (!parse_st_header(path, st, err)) {
      return unknown_det(path, "safetensors header parse failed: " + err);
    }
    Detection d;
    d.format = ModelFormat::SAFETENSORS_FILE;
    d.path = path;
    d.summary = "safetensors archive, " + std::to_string(st.tensor_count) + " tensors";
    d.meta = st.to_meta();
    return d;
  }

  return unknown_det(path, "unrecognized file signature/extension");
}

Detection detect_dir(const std::string& path) {
  const fs::path cfg_path = fs::path(path) / "config.json";
  std::error_code ec;
  if (!fs::is_regular_file(cfg_path, ec)) {
    return unknown_det(path, "directory without config.json");
  }

  std::ifstream f(cfg_path, std::ios::binary);
  if (!f) return unknown_det(path, "config.json unreadable");
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();

  json::Node root;
  std::string err;
  if (!json::parse(text, root, err) || !root.is_object()) {
    return unknown_det(path, "config.json is not valid JSON (" +
                                 (err.empty() ? "not an object" : err) + ")");
  }

  Detection d;
  d.path = path;
  if (const json::Node* mt = root.find("model_type")) {
    if (mt->is_string()) {
      d.meta["model_type"] = mt->as_string();
      d.meta["has_model_type"] = "yes";
    } else {
      d.meta["has_model_type"] = "no";
    }
  } else {
    d.meta["has_model_type"] = "no";
  }

  if (root.contains("quantization")) {
    d.format = ModelFormat::MLX_DIR;
    d.meta["quantization_style"] = "mlx";
    if (const json::Node* q = root.find("quantization")) {
      if (q->is_object()) {
        if (const json::Node* g = q->find("group_size"))
          d.meta["mlx_group_size"] = std::to_string(g->as_int());
        if (const json::Node* b = q->find("bits"))
          d.meta["mlx_bits"] = std::to_string(b->as_int());
      }
    }
    d.summary = "MLX converted model directory (mlx-lm layout)";
    return d;
  }

  if (const json::Node* qc = root.find("quantization_config")) {
    if (qc->is_object()) {
      std::string method;
      if (const json::Node* m = qc->find("quant_method")) {
        if (m->is_string()) method = m->as_string();
      }
      for (const auto& key : {"bits", "group_size"}) {
        if (const json::Node* v = qc->find(key)) d.meta[key] = std::to_string(v->as_int());
      }
      if (method == "awq") {
        d.format = ModelFormat::HF_DIR_AWQ;
        d.summary = "AWQ-quantized HF directory";
        return d;
      }
      if (method == "gptq") {
        d.format = ModelFormat::HF_DIR_GPTQ;
        d.summary = "GPTQ-quantized HF directory";
        return d;
      }
      d.meta["quant_method"] = method;
      d.summary = "HF dir with unsupported quantization_config.quant_method='" + method + "'";
      d.format = ModelFormat::UNKNOWN;
      return d;
    }
  }

  d.format = ModelFormat::UNKNOWN;
  d.summary = "HF-style directory (fp safetensors/pytorch weights); execution roadmap";
  return d;
}

} // namespace

Detection detect_format(const std::string& path) {
  std::error_code ec;
  if (!fs::exists(fs::path(path), ec)) {
    return unknown_det(path, "path does not exist");
  }
  if (fs::is_directory(fs::path(path), ec)) return detect_dir(path);
  return detect_file(path);
}

} // namespace presto
