// presto - MLX backend implementation (Apple Silicon only)
// Minimal token-id-level Llama-family forward pass built on mlx core ops.
// Supports UNQUANTIZED mlx-lm converted directories (HF-style weight names,
// F32/F16/BF16 safetensors). Quantized MLX models are rejected with a clear
// error rather than mis-executed.
#include "backends/mlx_backend.hpp"

#include "json_mini.hpp"
#include "presto/log.hpp"

#include "mlx/array.h"
#include "mlx/ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace presto {
namespace {

constexpr double kDefaultRopeTheta = 10000.0;
constexpr float kNegHuge = -1e30f;

std::size_t elem_size(const std::string& dtype) {
  if (dtype == "F32") return 4;
  if (dtype == "F16" || dtype == "BF16") return 2;
  return 0;
}

float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp10 = (h >> 10) & 0x1Fu;
  std::uint32_t man10 = h & 0x03FFu;
  std::uint32_t out;
  if (exp10 == 0) {
    if (man10 == 0) {
      out = sign;  // +/-0
    } else {
      int e = -14;
      while (!(man10 & 0x0400u)) {
        man10 <<= 1;
        --e;
      }
      man10 &= 0x03FFu;
      out = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (man10 << 13);
    }
  } else if (exp10 == 31) {
    out = sign | 0x7F800000u | (man10 << 13);  // inf / nan
  } else {
    out = sign | ((exp10 - 15 + 127) << 23) | (man10 << 13);
  }
  float f = 0.f;
  std::memcpy(&f, &out, sizeof f);
  return f;
}

float bf16_to_f32(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float f = 0.f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

struct TensorData {
  std::vector<int> shape;
  std::vector<float> data;
};

std::uint64_t read_le64(const unsigned char* b) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<std::uint64_t>(b[i]) << (8 * i);
  return v;
}

bool read_all_bytes(const fs::path& p, std::string& buf, std::string& err) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) {
    err = "cannot open " + p.string();
    return false;
  }
  const std::streamoff size = f.tellg();
  if (size <= 0 || size > (1LL << 40)) {
    err = "file size unreasonable: " + p.string();
    return false;
  }
  buf.resize(static_cast<std::size_t>(size));
  f.seekg(0);
  f.read(buf.data(), size);
  if (static_cast<std::size_t>(f.gcount()) != buf.size()) {
    err = "short read: " + p.string();
    return false;
  }
  return true;
}

bool tensor_bytes_to_floats(const std::string& dtype, const unsigned char* bytes,
                            std::size_t n_elems, TensorData& out, std::string& err) {
  out.data.resize(n_elems);
  if (dtype == "F32") {
    std::memcpy(out.data.data(), bytes, n_elems * sizeof(float));
    return true;
  }
  if (dtype == "F16") {
    for (std::size_t i = 0; i < n_elems; ++i) {
      std::uint16_t h;
      std::memcpy(&h, bytes + i * 2, 2);
      out.data[i] = f16_to_f32(h);
    }
    return true;
  }
  if (dtype == "BF16") {
    for (std::size_t i = 0; i < n_elems; ++i) {
      std::uint16_t h;
      std::memcpy(&h, bytes + i * 2, 2);
      out.data[i] = bf16_to_f32(h);
    }
    return true;
  }
  err = "unsupported dtype '" + dtype + "' (quantized MLX checkpoints are not supported yet)";
  return false;
}

// Load every non-quantized tensor from all *.safetensors shards in dir.
bool load_weights(const std::string& dir,
                  std::unordered_map<std::string, TensorData>& weights, std::string& err) {
  bool found_any_shard = false;
  std::error_code ec;
  fs::directory_iterator it(fs::path(dir), ec);
  if (ec) {
    err = "cannot list directory: " + dir;
    return false;
  }
  for (const auto& entry : it) {
    if (!entry.is_regular_file()) continue;
    const std::string name = entry.path().filename().string();
    if (name.size() < 12 || name.compare(name.size() - 12, 12, ".safetensors") != 0) continue;
    found_any_shard = true;

    std::string buf;
    if (!read_all_bytes(entry.path(), buf, err)) return false;
    if (buf.size() < 8) {
      err = "shard too small: " + name;
      return false;
    }
    const std::uint64_t header_len =
        read_le64(reinterpret_cast<const unsigned char*>(buf.data()));
    if (header_len == 0 || header_len > (256ull << 20) || header_len + 8 > buf.size()) {
      err = "bad header length in " + name;
      return false;
    }
    const std::string header_text = buf.substr(8, static_cast<std::size_t>(header_len));
    const unsigned char* data_base =
        reinterpret_cast<const unsigned char*>(buf.data()) + 8 + header_len;

    json::Node root;
    std::string jerr;
    if (!json::parse(header_text, root, jerr) || !root.is_object()) {
      err = "bad JSON header in " + name + ": " + jerr;
      return false;
    }

    for (const auto& [tname, tnode] : root.members()) {
      if (tname == "__metadata__" || !tnode.is_object()) continue;
      const json::Node* dt = tnode.find("dtype");
      const json::Node* shp = tnode.find("shape");
      const json::Node* off = tnode.find("data_offsets");
      if (!dt || !dt->is_string() || !shp || !shp->is_array() || !off ||
          !off->is_array() || off->items().size() != 2) {
        err = "malformed tensor entry '" + tname + "' in " + name;
        return false;
      }
      if (tname.size() > 7 && tname.compare(tname.size() - 7, 7, ".scales") == 0) {
        err = "quantized MLX checkpoint detected ('" + tname +
              "'); presto currently supports unquantized MLX dirs";
        return false;
      }

      TensorData td;
      td.shape.reserve(shp->items().size());
      std::size_t elems = 1;
      for (const auto& dim : shp->items()) {
        const int d = static_cast<int>(dim.as_int());
        td.shape.push_back(d);
        elems *= static_cast<std::size_t>(d);
      }

      const auto start = static_cast<std::uint64_t>(off->items()[0].as_int());
      const auto end = static_cast<std::uint64_t>(off->items()[1].as_int());
      const std::size_t esz = elem_size(dt->as_string());
      if (esz == 0 || end < start || end - start != elems * esz) {
        err = "tensor data range/dtype invalid for '" + tname + "' in " + name;
        return false;
      }
      if (!tensor_bytes_to_floats(dt->as_string(), data_base + start, elems, td, err))
        return false;
      weights[tname] = std::move(td);
    }
  }
  if (!found_any_shard) {
    err = "no *.safetensors files found in directory";
    return false;
  }
  return true;
}

} // namespace

struct MlxBackend::Impl {
  int hidden_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int intermediate_size = 0;
  int vocab_size = 0;
  float rms_eps = 1e-5f;
  double rope_theta = kDefaultRopeTheta;
  int eos_token_id = -1;
  bool tied_embeddings = false;

  std::unordered_map<std::string, TensorData> weights;
};

MlxBackend::MlxBackend(std::string dir) : dir_(std::move(dir)) {}
MlxBackend::~MlxBackend() = default;

bool MlxBackend::load(std::string& err) {
  impl_ = std::make_unique<Impl>();

  std::string cfg_buf;
  if (!read_all_bytes(fs::path(dir_) / "config.json", cfg_buf, err)) return false;
  json::Node cfg;
  std::string jerr;
  if (!json::parse(cfg_buf, cfg, jerr) || !cfg.is_object()) {
    err = "config.json is not valid JSON";
    return false;
  }
  auto need_int = [&](const char* key, int& dst) -> bool {
    const json::Node* n = cfg.find(key);
    if (!n || !n->is_number()) {
      err = std::string("config.json missing numeric field '") + key + "'";
      return false;
    }
    dst = static_cast<int>(n->as_int());
    return true;
  };
  if (!need_int("hidden_size", impl_->hidden_size)) return false;
  if (!need_int("num_hidden_layers", impl_->num_layers)) return false;
  if (!need_int("num_attention_heads", impl_->num_heads)) return false;
  if (const json::Node* n = cfg.find("num_key_value_heads"); n && n->is_number())
    impl_->num_kv_heads = static_cast<int>(n->as_int());
  else
    impl_->num_kv_heads = impl_->num_heads;
  if (!need_int("intermediate_size", impl_->intermediate_size)) return false;
  if (!need_int("vocab_size", impl_->vocab_size)) return false;
  if (impl_->hidden_size % impl_->num_heads != 0) {
    err = "hidden_size not divisible by num_attention_heads";
    return false;
  }
  if (impl_->num_heads % impl_->num_kv_heads != 0) {
    err = "num_attention_heads not divisible by num_key_value_heads";
    return false;
  }
  if (const json::Node* n = cfg.find("rms_norm_eps"))
    impl_->rms_eps = static_cast<float>(n->as_double(1e-5));
  if (const json::Node* n = cfg.find("rope_theta"))
    impl_->rope_theta = n->as_double(kDefaultRopeTheta);
  if (const json::Node* n = cfg.find("eos_token_id"))
    impl_->eos_token_id = static_cast<int>(n->as_int());
  if (const json::Node* n = cfg.find("tie_word_embeddings"))
    impl_->tied_embeddings = n->as_bool(false);

  const auto t0 = std::chrono::steady_clock::now();
  if (!load_weights(dir_, impl_->weights, err)) return false;
  load_sec_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::vector<std::string> required = {"model.embed_tokens.weight", "model.norm.weight"};
  for (int l = 0; l < impl_->num_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    for (const char* s : {"input_layernorm.weight", "post_attention_layernorm.weight",
                          "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                          "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                          "mlp.gate_proj.weight", "mlp.up_proj.weight",
                          "mlp.down_proj.weight"})
      required.push_back(p + s);
  }
  if (!impl_->tied_embeddings && !impl_->weights.count("lm_head.weight"))
    required.push_back("lm_head.weight");
  for (const auto& r : required) {
    if (!impl_->weights.count(r)) {
      err = "missing weight tensor: " + r;
      return false;
    }
  }
  PRESTO_LOG_INFO("mlx", "loaded " + std::to_string(impl_->weights.size()) + " tensors in " +
                             std::to_string(load_sec_) + "s");
  return true;
}

namespace {

mx::array make_f32_array(const TensorData& t) {
  return mx::array(t.data.data(), t.shape);
}

// x[T,H]: (x * w[H]) / sqrt(mean(x^2)+eps)
mx::array rms_norm(const mx::array& x, const mx::array& w, float eps) {
  const mx::array msq = mx::mean(mx::multiply(x, x), {-1}, true);
  return mx::multiply(
      x, mx::divide(w, mx::sqrt(mx::add(msq, mx::array(static_cast<double>(eps))))));
}

} // namespace

bool MlxBackend::generate(const GenerateParams& gp, GenerateResult& r, std::string& err) {
  if (!impl_) {
    err = "backend not loaded";
    return false;
  }
  Impl& I = *impl_;

  std::vector<int> ids;
  if (!gp.prompt_tokens.empty())
    ids = gp.prompt_tokens;
  else
    ids = {1};  // no tokenizer here; token-id-level backend seeds with a bos-ish id
  if (ids.empty()) {
    err = "empty prompt";
    return false;
  }

  const int H = I.hidden_size;
  const int nh = I.num_heads;
  const int nkv = I.num_kv_heads;
  const int dh = H / nh;
  const bool use_separate_lm_head =
      !I.tied_embeddings && I.weights.count("lm_head.weight") != 0;

  std::vector<std::int32_t> kv_idx;
  kv_idx.reserve(static_cast<std::size_t>(nh));
  for (int h = 0; h < nh; ++h) kv_idx.push_back(static_cast<std::int32_t>(h / (nh / nkv)));

  try {
    const mx::array embed = make_f32_array(I.weights.at("model.embed_tokens.weight"));
    const mx::array norm_w = make_f32_array(I.weights.at("model.norm.weight"));

    const auto gen_t0 = std::chrono::steady_clock::now();
    std::vector<int> out_tokens;
    out_tokens.reserve(static_cast<std::size_t>(std::max(0, gp.max_tokens)));

    for (int step = 0; step < gp.max_tokens; ++step) {
      const int T = static_cast<int>(ids.size());

      // RoPE tables [T, dh]
      std::vector<float> cosv(static_cast<std::size_t>(T) * dh);
      std::vector<float> sinv(cosv.size());
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < dh / 2; ++i) {
          const double ang = t * std::pow(I.rope_theta, -2.0 * i / dh);
          cosv[static_cast<std::size_t>(t) * dh + i] =
              cosv[static_cast<std::size_t>(t) * dh + i + dh / 2] =
                  static_cast<float>(std::cos(ang));
          sinv[static_cast<std::size_t>(t) * dh + i] =
              sinv[static_cast<std::size_t>(t) * dh + i + dh / 2] =
                  static_cast<float>(std::sin(ang));
        }
      const mx::array costab_h =
          mx::expand_dims(mx::array(cosv.data(), {T, dh}), 1);  // [T,1,dh]
      const mx::array sintab_h = mx::expand_dims(mx::array(sinv.data(), {T, dh}), 1);

      // causal mask [1,T,T]
      std::vector<float> maskv(static_cast<std::size_t>(T) * T, 0.f);
      for (int i = 0; i < T; ++i)
        for (int j = i + 1; j < T; ++j)
          maskv[static_cast<std::size_t>(i) * T + j] = kNegHuge;
      const mx::array mask = mx::expand_dims(mx::array(maskv.data(), {T, T}), 0);

      mx::array x = mx::take(embed, mx::array(ids.data(), {T}), 0);

      auto rope_apply = [&](const mx::array& flat /*[T, heads*dh]*/, int heads) {
        const mx::array x3 = mx::reshape(flat, {T, heads, dh});
        auto sl = [&](int s, int e) {
          return mx::slice(x3, {0, 0, s}, {T, heads, e}, {1, 1, 1});
        };
        const mx::array rot = mx::concat({mx::negative(sl(dh / 2, dh)), sl(0, dh / 2)}, -1);
        return mx::add(mx::multiply(x3, costab_h), mx::multiply(rot, sintab_h));
      };

      for (int l = 0; l < I.num_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const auto W = [&](const char* n) { return make_f32_array(I.weights.at(p + n)); };

        // --- self attention ---
        mx::array h = rms_norm(x, W("input_layernorm.weight"), I.rms_eps);

        const mx::array q = rope_apply(mx::matmul(h, W("self_attn.q_proj.weight")), nh);
        const mx::array k = rope_apply(mx::matmul(h, W("self_attn.k_proj.weight")), nkv);
        const mx::array v =
            mx::reshape(mx::matmul(h, W("self_attn.v_proj.weight")), {T, nkv, dh});

        const mx::array kv_index = mx::array(kv_idx.data(), {nh});
        const mx::array k_rep = mx::take(k, kv_index, 1);
        const mx::array v_rep = mx::take(v, kv_index, 1);

        mx::array scores = mx::matmul(q, mx::transpose(k_rep, {0, 2, 1}));
        scores = mx::divide(scores, mx::array(std::sqrt(static_cast<double>(dh))));
        scores = mx::add(scores, mask);
        const mx::array probs = mx::softmax(scores, -1);
        mx::array attn = mx::matmul(probs, v_rep);  // [T,nh,dh]
        attn = mx::reshape(attn, {T, H});
        attn = mx::matmul(attn, W("self_attn.o_proj.weight"));
        x = mx::add(x, attn);

        // --- mlp ---
        h = rms_norm(x, W("post_attention_layernorm.weight"), I.rms_eps);
        const mx::array g = mx::matmul(h, W("mlp.gate_proj.weight"));
        const mx::array u = mx::matmul(h, W("mlp.up_proj.weight"));
        const mx::array sig =
            mx::divide(mx::array(1.0), mx::add(mx::array(1.0), mx::exp(mx::negative(g))));
        const mx::array mlp = mx::matmul(mx::multiply(g, sig), u, W("mlp.down_proj.weight"));
        x = mx::add(x, mlp);
      }

      x = rms_norm(x, norm_w, I.rms_eps);
      mx::array logits = use_separate_lm_head
                             ? mx::matmul(x, make_f32_array(I.weights.at("lm_head.weight")))
                             : mx::matmul(x, embed);
      mx::eval(logits);

      const mx::array last_row = mx::reshape(
          mx::slice(logits, {T - 1, 0}, {T, I.vocab_size}, {1, 1}), {I.vocab_size});
      const int next =
          static_cast<int>(mx::argmax(last_row, -1).item<std::int32_t>());

      if (I.eos_token_id >= 0 && next == I.eos_token_id) break;
      ids.push_back(next);
      out_tokens.push_back(next);
    }

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - gen_t0).count();
    r.tokens = out_tokens;
    r.tok_per_sec = secs > 0 ? static_cast<double>(out_tokens.size()) / secs : 0.0;
    r.load_sec = load_sec_;
    return true;
  } catch (const std::exception& e) {
    err = std::string("step=mlx_forward msg=\"") + e.what() + "\"";
    return false;
  }
}

std::unique_ptr<IBackend> make_mlx_backend(const std::string& path) {
  return std::make_unique<MlxBackend>(path);
}

} // namespace presto
