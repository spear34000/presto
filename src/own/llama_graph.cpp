// presto own-engine: llama-architecture forward graph.
// Zero ggml/llama dependency - plain C++20, f32 reference math, full recompute
// per call (no KV cache). Determinism over speed; single-threaded by design.
//
// Conventions mirror the ggml f32 reference kernels:
//   rmsnorm : scale = 1/sqrt(mean(x^2) + eps); out = (x*scale)*w
//   rope    : adjacent pairs (2i, 2i+1), angle = pos * pow(base, -2i/n_rot)
//   attn    : scores scaled by 1/sqrt(head_dim) after the dot product,
//             causal mask, softmax with max subtraction, ascending-j V mix
//   silu    : x / (1 + exp(-x))
#include "own.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace presto {
namespace own {
namespace {

// Internal control-flow exception; never escapes llama_forward (caught at the
// boundary and converted into `err`).
struct fwd_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

const Tensor& need(const GGufModel& m, const std::string& name) {
  const Tensor* t = m.tensor(name);
  if (!t) {
    throw fwd_error("missing tensor '" + name + "'");
  }
  return *t;
}

void expect_shape(const Tensor& t, int64_t rows, int64_t cols, const char* role) {
  const bool ok = t.ne.size() >= 2 && static_cast<int64_t>(t.ne[0]) == cols &&
                  static_cast<int64_t>(t.ne[1]) == rows &&
                  t.data.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (!ok) {
    throw fwd_error("tensor '" + t.name + "' (" + role + ") has shape [" +
                    std::to_string(t.ne.size() >= 2 ? t.ne[0] : 0) + " x " +
                    std::to_string(t.ne.size() >= 2 ? t.ne[1] : 0) + "], expected [" +
                    std::to_string(cols) + " x " + std::to_string(rows) + "]");
  }
}

void expect_elems(const Tensor& t, int64_t n, const char* role) {
  size_t total = 1;
  for (uint32_t d : t.ne) total *= d;
  if (total != static_cast<size_t>(n) || t.data.size() != static_cast<size_t>(n)) {
    throw fwd_error("tensor '" + t.name + "' (" + role + ") has " +
                    std::to_string(t.data.size()) + " elements, expected " +
                    std::to_string(n));
  }
}

inline float vec_dot(const float* a, const float* b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

// y[rows] = W[rows x cols] * x[cols], W row-major (ne[0]=cols, ne[1]=rows).
void matvec(const Tensor& w, const float* x, float* y) {
  const int cols = static_cast<int>(w.ne[0]);
  const int rows = static_cast<int>(w.ne[1]);
  const float* data = w.data.data();
  for (int r = 0; r < rows; ++r) {
    y[r] = vec_dot(data + static_cast<size_t>(r) * cols, x, cols);
  }
}

void rms_norm(const float* x, const float* w, int n, float eps, float* out) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }
  const float scale = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) {
    out[i] = x[i] * scale * w[i];
  }
}

// In-place GGML-convention rope on [n_head x head_dim]; rotates the first
// n_rot dims of every head as adjacent pairs (2i, 2i+1), tail passes through.
void rope_inplace(float* v, int n_head, int head_dim, int n_rot, int pos,
                  float theta_base) {
  const float theta_scale = std::pow(theta_base, -2.0f / static_cast<float>(n_rot));
  const int pairs = n_rot / 2;
  for (int h = 0; h < n_head; ++h) {
    float* head = v + static_cast<size_t>(h) * head_dim;
    for (int i = 0; i < pairs; ++i) {
      const float theta = static_cast<float>(pos) * std::pow(theta_scale, static_cast<float>(i));
      const float c = std::cos(theta);
      const float s = std::sin(theta);
      const float x0 = head[2 * i + 0];
      const float x1 = head[2 * i + 1];
      head[2 * i + 0] = x0 * c - x1 * s;
      head[2 * i + 1] = x0 * s + x1 * c;
    }
  }
}

void softmax(float* v, int n) {
  float mx = v[0];
  for (int i = 1; i < n; ++i) {
    mx = std::max(mx, v[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    v[i] = std::exp(v[i] - mx);
    sum += v[i];
  }
  for (int i = 0; i < n; ++i) {
    v[i] /= sum;
  }
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

} // namespace

bool llama_forward(const GGufModel& m, const std::vector<int>& tokens,
                   std::vector<float>& logits_out, std::string& err) {
  try {
    // ---- config ----
    if (tokens.empty()) {
      throw fwd_error("empty token list");
    }
    const int n_vocab = m.n_vocab;
    const int n_embd = m.n_embd;
    const int n_layer = m.n_layer;
    const int n_head = m.n_head;
    if (n_vocab <= 0 || n_embd <= 0 || n_layer <= 0 || n_head <= 0) {
      throw fwd_error("model metadata incomplete (need positive n_vocab/n_embd/n_layer/n_head)");
    }
    if (n_embd % n_head != 0) {
      throw fwd_error("n_embd (" + std::to_string(n_embd) + ") not divisible by n_head (" +
                      std::to_string(n_head) + ")");
    }
    const int head_dim = n_embd / n_head;
    const int n_head_kv = m.n_head_kv > 0 ? m.n_head_kv : n_head;
    if (n_head_kv <= 0 || n_head % n_head_kv != 0) {
      throw fwd_error("n_head_kv (" + std::to_string(n_head_kv) + ") must divide n_head (" +
                      std::to_string(n_head) + ")");
    }
    const int kv_rep = n_head / n_head_kv;
    const int n_embd_kv = head_dim * n_head_kv;
    int n_rot = m.n_rot > 0 ? m.n_rot : head_dim;
    if (n_rot > head_dim) {
      n_rot = head_dim;
    }
    if (n_rot % 2 != 0) {
      throw fwd_error("n_rot (" + std::to_string(n_rot) + ") must be even");
    }
    const float eps = m.norm_eps;
    const float theta_base = m.rope_theta;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int tok : tokens) {
      if (tok < 0 || tok >= n_vocab) {
        throw fwd_error("token id out of range: " + std::to_string(tok) +
                        " (n_vocab=" + std::to_string(n_vocab) + ")");
      }
    }

    // ---- gather tensors up front so any missing weight fails fast ----
    const Tensor& tok_embd = need(m, "token_embd.weight");
    expect_shape(tok_embd, n_vocab, n_embd, "token embeddings");
    const Tensor& out_norm = need(m, "output_norm.weight");
    expect_elems(out_norm, n_embd, "output norm");
    const Tensor* out_proj = m.tensor("output.weight");
    if (out_proj) {
      expect_shape(*out_proj, n_vocab, n_embd, "output head");
    }

    struct layer_tensors {
      const Tensor* attn_norm = nullptr;
      const Tensor* attn_q = nullptr;
      const Tensor* attn_k = nullptr;
      const Tensor* attn_v = nullptr;
      const Tensor* attn_o = nullptr;
      const Tensor* ffn_norm = nullptr;
      const Tensor* ffn_gate = nullptr;       // separate-gate layout
      const Tensor* ffn_up = nullptr;
      const Tensor* ffn_down = nullptr;
      const Tensor* ffn_fused_gate_up = nullptr; // fused fallback
    };
    std::vector<layer_tensors> layers(static_cast<size_t>(n_layer));

    int ffn_dim = m.ffn_dim > 0 ? m.ffn_dim : 0;
    for (int l = 0; l < n_layer; ++l) {
      const std::string p = "blk." + std::to_string(l) + ".";
      layer_tensors& lt = layers[static_cast<size_t>(l)];

      lt.attn_norm = &need(m, p + "attn_norm.weight");
      expect_elems(*lt.attn_norm, n_embd, "attn norm");
      lt.attn_q = &need(m, p + "attn_q.weight");
      expect_shape(*lt.attn_q, n_embd, n_embd, "attn q proj");
      lt.attn_k = &need(m, p + "attn_k.weight");
      expect_shape(*lt.attn_k, n_embd_kv, n_embd, "attn k proj");
      lt.attn_v = &need(m, p + "attn_v.weight");
      expect_shape(*lt.attn_v, n_embd_kv, n_embd, "attn v proj");
      lt.attn_o = &need(m, p + "attn_output.weight");
      expect_shape(*lt.attn_o, n_embd, n_embd, "attn output proj");

      lt.ffn_norm = &need(m, p + "ffn_norm.weight");
      expect_elems(*lt.ffn_norm, n_embd, "ffn norm");

      lt.ffn_gate = m.tensor(p + "ffn_gate.weight");
      if (lt.ffn_gate) {
        lt.ffn_fused_gate_up = nullptr;
      } else {
        // Fused gate+up layout: single [2*ffn, n_embd] tensor, gate rows first.
        lt.ffn_fused_gate_up = m.tensor(p + "ffn_gate_inp.weight");
        if (!lt.ffn_fused_gate_up) {
          throw fwd_error("missing tensor '" + p + "ffn_gate.weight' (and no fused '" +
                          p + "ffn_gate_inp.weight')");
        }
      }

      lt.ffn_up = &need(m, p + "ffn_up.weight");
      const int layer_ffn = static_cast<int>(lt.ffn_up->ne[1]);
      if (ffn_dim <= 0) {
        ffn_dim = layer_ffn;
      } else if (layer_ffn != ffn_dim) {
        throw fwd_error("tensor '" + lt.ffn_up->name + "' has inconsistent ffn dim " +
                        std::to_string(layer_ffn) + ", expected " + std::to_string(ffn_dim));
      }
      if (lt.ffn_gate) {
        expect_shape(*lt.ffn_gate, ffn_dim, n_embd, "ffn gate proj");
      } else {
        expect_shape(*lt.ffn_fused_gate_up, 2 * ffn_dim, n_embd, "ffn fused gate+up proj");
      }
      expect_shape(*lt.ffn_up, ffn_dim, n_embd, "ffn up proj");
      lt.ffn_down = &need(m, p + "ffn_down.weight");
      expect_shape(*lt.ffn_down, n_embd, ffn_dim, "ffn down proj");
    }
    if (ffn_dim <= 0) {
      throw fwd_error("could not determine ffn dimension");
    }

    // ---- forward ----
    const int T = static_cast<int>(tokens.size());
    const size_t ne = static_cast<size_t>(n_embd);

    std::vector<float> x(static_cast<size_t>(T) * ne);   // residual stream
    for (int t = 0; t < T; ++t) {
      const float* row = tok_embd.data.data() + static_cast<size_t>(tokens[t]) * ne;
      std::copy(row, row + ne, x.begin() + static_cast<size_t>(t) * ne);
    }

    std::vector<float> xn(static_cast<size_t>(T) * ne);        // normed input
    std::vector<float> q(static_cast<size_t>(T) * ne);         // roped queries
    std::vector<float> k(static_cast<size_t>(T) * n_embd_kv);  // roped keys
    std::vector<float> v(static_cast<size_t>(T) * n_embd_kv);  // values
    std::vector<float> att(static_cast<size_t>(T) * ne);       // attention output
    std::vector<float> scores(static_cast<size_t>(T));
    std::vector<float> gate(ffn_dim);
    std::vector<float> up(ffn_dim);
    std::vector<float> act(ffn_dim);
    std::vector<float> fused(2 * ffn_dim);
    std::vector<float> tmp(ne);

    for (int l = 0; l < n_layer; ++l) {
      const layer_tensors& lt = layers[static_cast<size_t>(l)];

      // (a) q/k/v projections + rope for every position
      for (int t = 0; t < T; ++t) {
        const float* xt = &x[static_cast<size_t>(t) * ne];
        float* xnt = &xn[static_cast<size_t>(t) * ne];
        rms_norm(xt, lt.attn_norm->data.data(), n_embd, eps, xnt);
        matvec(*lt.attn_q, xnt, &q[static_cast<size_t>(t) * ne]);
        matvec(*lt.attn_k, xnt, &k[static_cast<size_t>(t) * n_embd_kv]);
        matvec(*lt.attn_v, xnt, &v[static_cast<size_t>(t) * n_embd_kv]);
        rope_inplace(&q[static_cast<size_t>(t) * ne], n_head, head_dim, n_rot, t, theta_base);
        rope_inplace(&k[static_cast<size_t>(t) * n_embd_kv], n_head_kv, head_dim, n_rot, t,
                     theta_base);
      }

      // causal attention, GQA via kv-head repeat
      for (int t = 0; t < T; ++t) {
        for (int h = 0; h < n_head; ++h) {
          const int kvh = h / kv_rep;
          const float* qh = &q[static_cast<size_t>(t) * ne + static_cast<size_t>(h) * head_dim];
          for (int j = 0; j <= t; ++j) {
            const float* kj =
                &k[static_cast<size_t>(j) * n_embd_kv + static_cast<size_t>(kvh) * head_dim];
            scores[static_cast<size_t>(j)] = vec_dot(qh, kj, head_dim) * kq_scale;
          }
          softmax(scores.data(), t + 1);

          float* oh = &att[static_cast<size_t>(t) * ne + static_cast<size_t>(h) * head_dim];
          std::fill(oh, oh + head_dim, 0.0f);
          for (int j = 0; j <= t; ++j) {
            const float pj = scores[static_cast<size_t>(j)];
            const float* vj =
                &v[static_cast<size_t>(j) * n_embd_kv + static_cast<size_t>(kvh) * head_dim];
            for (int c = 0; c < head_dim; ++c) {
              oh[c] += pj * vj[c];
            }
          }
        }
        matvec(*lt.attn_o, &att[static_cast<size_t>(t) * ne], tmp.data());
        for (int e = 0; e < n_embd; ++e) {
          x[static_cast<size_t>(t) * ne + e] += tmp[e];
        }
      }

      // (b) swiglu feed-forward
      for (int t = 0; t < T; ++t) {
        const float* xt = &x[static_cast<size_t>(t) * ne];
        float* xnt = &xn[static_cast<size_t>(t) * ne];
        rms_norm(xt, lt.ffn_norm->data.data(), n_embd, eps, xnt);
        if (lt.ffn_fused_gate_up) {
          matvec(*lt.ffn_fused_gate_up, xnt, fused.data());
          for (int i = 0; i < ffn_dim; ++i) {
            act[i] = silu(fused[i]) * fused[ffn_dim + i];
          }
        } else {
          matvec(*lt.ffn_gate, xnt, gate.data());
          matvec(*lt.ffn_up, xnt, up.data());
          for (int i = 0; i < ffn_dim; ++i) {
            act[i] = silu(gate[i]) * up[i];
          }
        }
        matvec(*lt.ffn_down, act.data(), tmp.data());
        for (int e = 0; e < n_embd; ++e) {
          x[static_cast<size_t>(t) * ne + e] += tmp[e];
        }
      }
      if (std::getenv("PRESTO_OWN_DEBUG")) {
        std::fprintf(stderr, "[own-debug] L%d x0[0..4]:", l);
        for (int e = 0; e < 5; ++e)
          std::fprintf(stderr, " %.5f", x[e]);
        std::fprintf(stderr, "\n");
      }
    }

    // ---- final norm + lm head on the last position ----
    rms_norm(&x[static_cast<size_t>(T - 1) * ne], out_norm.data.data(), n_embd, eps, xn.data());
    logits_out.assign(static_cast<size_t>(n_vocab), 0.0f);
    matvec(out_proj ? *out_proj : tok_embd, xn.data(), logits_out.data());

    if (logits_out.size() != static_cast<size_t>(n_vocab)) {
      throw fwd_error("internal error: logits size mismatch");
    }
    return true;
  } catch (const std::exception& e) {
    err = std::string("llama_forward: ") + e.what();
    return false;
  } catch (...) {
    err = "llama_forward: unknown failure";
    return false;
  }
}

} // namespace own
} // namespace presto
