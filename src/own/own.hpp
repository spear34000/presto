// presto own-engine contract: zero llama.cpp/ggml dependency.
// v0 strategy: full-recompute greedy forward (no kv cache) targeting
// bit-identical token ids vs the reference backend on tiny models.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace presto::own {

// Dequantized f32 tensor, row-major, ne[0]=cols, ne[1]=rows.
struct Tensor {
  std::string name;
  std::vector<uint32_t> ne;
  std::vector<float> data;
};

class GGufModel {
public:
  static bool load(const std::string& path, GGufModel& out, std::string& err);

  const Tensor* tensor(const std::string& name) const;
  std::string meta_str(const std::string& key, const std::string& dflt = "") const;
  int64_t meta_int(const std::string& key, int64_t dflt) const;
  float meta_float(const std::string& key, float dflt) const;

  // sentencepiece BPE over GGUF-embedded vocab (tokenizer.ggml.* arrays),
  // byte-fallback, no sentencepiece library.
  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;

  // llama-arch config (read from metadata at load)
  std::string arch;
  int n_vocab = 0, n_ctx_train = 0, n_embd = 0, n_layer = 0, n_head = 0;
  int n_head_kv = 0, n_rot = 0, ffn_dim = 0;
  float rope_theta = 10000.f, norm_eps = 1e-5f;
  bool add_bos = true;

private:
  std::vector<char> buf_;                       // whole-file image
  std::vector<Tensor> tensors_;
  std::vector<std::pair<std::string, std::string>> kv_;
};

// Full-sequence forward, llama arch (rmsnorm, rope, swiglu, gqa).
// Writes final-position logits (size n_vocab) into logits_out.
bool llama_forward(const GGufModel& m, const std::vector<int>& tokens,
                   std::vector<float>& logits_out, std::string& err);

} // namespace presto::own
