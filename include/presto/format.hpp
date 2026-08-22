// presto - core format types
#pragma once

#include <map>
#include <string>

namespace presto {

enum class ModelFormat {
  GGUF,            // single .gguf file (llama.cpp family)
  SAFETENSORS_FILE,// single .safetensors file (HF tensor archive)
  PYTORCH_FILE,    // .pt/.pth/.ckpt zip container
  HF_DIR_AWQ,      // directory with config.json quantization_config.quant_method=awq
  HF_DIR_GPTQ,     // directory with config.json quantization_config.quant_method=gptq
  MLX_DIR,         // mlx-lm converted directory (config.json with "quantization")
  UNKNOWN,
};

inline const char* format_name(ModelFormat f) {
  switch (f) {
    case ModelFormat::GGUF: return "gguf";
    case ModelFormat::SAFETENSORS_FILE: return "safetensors";
    case ModelFormat::PYTORCH_FILE: return "pytorch";
    case ModelFormat::HF_DIR_AWQ: return "awq";
    case ModelFormat::HF_DIR_GPTQ: return "gptq";
    case ModelFormat::MLX_DIR: return "mlx";
    case ModelFormat::UNKNOWN: break;
  }
  return "unknown";
}

struct Detection {
  ModelFormat format = ModelFormat::UNKNOWN;
  std::string path;
  std::string summary;                         // human readable one-liner
  std::map<std::string, std::string> meta;     // parsed key/values
};

} // namespace presto
