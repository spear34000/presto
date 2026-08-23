// presto - llama.cpp GGUF backend implementation
// Persistent-context design: buffers are allocated once at load(); each
// generate() clears KV state (clean semantics) without paying allocation
// cost again, so server-mode tail latency stays flat.
#include "backends/llamacpp_backend.hpp"

#include "presto/log.hpp"

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace presto {
namespace {
std::string& spec_draft_path() {
  static std::string s;
  return s;
}
} // namespace

void set_spec_draft_path(const std::string& path) { spec_draft_path() = path; }

namespace {

int32_t env_int(const char* key, int32_t dflt) {
  if (const char* v = std::getenv(key); v && *v) {
    char* e = nullptr;
    const long long parsed = std::strtoll(v, &e, 10);
    if (e && *e == '\0' && parsed > 0) return static_cast<int32_t>(parsed);
  }
  return dflt;
}

double now_sec() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

llama_token sample_token(const float* logits, int n_vocab, float temp, unsigned& rng_state) {
  if (temp <= 0.0f || n_vocab <= 0) {
    int best = 0;
    for (int i = 1; i < n_vocab; ++i)
      if (logits[i] > logits[best]) best = i;
    return static_cast<llama_token>(best);
  }
  std::vector<double> probs(static_cast<std::size_t>(n_vocab));
  double maxl = -1e30;
  for (int i = 0; i < n_vocab; ++i) maxl = std::max(maxl, static_cast<double>(logits[i]));
  double sum = 0.0;
  for (int i = 0; i < n_vocab; ++i) {
    probs[static_cast<std::size_t>(i)] = std::exp((logits[i] - maxl) / temp);
    sum += probs[static_cast<std::size_t>(i)];
  }
  // xorshift-based sampling: deterministic under a fixed seed, no MT19937 alloc
  std::mt19937 rng(rng_state);
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  rng_state = rng();
  return static_cast<llama_token>(dist(rng));
}

} // namespace

struct LlamaCppBackend::Impl {
  llama_model* model = nullptr;
  const llama_vocab* vocab = nullptr;
  llama_context* ctx = nullptr;
  uint32_t n_ctx = 0;
  int32_t threads = 0;
  ggml_threadpool_t tpool = nullptr;
  ggml_threadpool_t tpool_batch = nullptr;

  // Prefix KV cache: token history currently resident in llama_memory.
  // A new prompt sharing this prefix skips prefill for the shared part -
  // multi-turn chats and repeated system prompts then cost near-zero TTFT.
  std::vector<llama_token> cache_tokens;
  bool cache_valid = false;

  // speculative decoding companion (greedy only)
  llama_model* d_model = nullptr;
  const llama_vocab* d_vocab = nullptr;
  llama_context* d_ctx = nullptr;
  std::vector<llama_token> d_cache_tokens;
};

LlamaCppBackend::LlamaCppBackend(std::string path) : path_(std::move(path)) {}

LlamaCppBackend::~LlamaCppBackend() {
  if (impl_) {
    if (impl_->d_ctx) llama_free(impl_->d_ctx);
    if (impl_->d_model) llama_model_free(impl_->d_model);
    if (impl_->ctx) llama_free(impl_->ctx);
    if (impl_->tpool_batch) ggml_threadpool_free(impl_->tpool_batch);
    if (impl_->tpool) ggml_threadpool_free(impl_->tpool);
    if (impl_->model) llama_model_free(impl_->model);
    llama_backend_free();
  }
}

bool LlamaCppBackend::load(std::string& err) {
  const double t0 = now_sec();
  llama_backend_init();
  impl_ = std::make_unique<Impl>();

  if (ggml_backend_reg_count() == 0) {
    ggml_backend_load_all();
  }

  llama_model_params mparams = llama_model_default_params();
  // -1 (llama.cpp default) offloads as many layers as fit on available GPU
  // devices once a GPU backend is compiled in; 0 forces CPU-only.
  mparams.n_gpu_layers =
      static_cast<int32_t>(env_int("PRESTO_GPU_LAYERS", -1));
  impl_->model = llama_model_load_from_file(path_.c_str(), mparams);
  if (!impl_->model) {
    err = "step=model_load msg=\"llama_model_load_from_file failed for " + path_ + "\"";
    return false;
  }
  impl_->vocab = llama_model_get_vocab(impl_->model);
  if (!impl_->vocab) {
    err = "step=get_vocab msg=\"null vocab\"";
    return false;
  }

  {
    // Report every registered compute device so logs show exactly where the
    // model will run (CPU / CUDA / Vulkan / HIP / Metal).
    const size_t n_regs = ggml_backend_reg_count();
    for (size_t i = 0; i < n_regs; ++i) {
      PRESTO_LOG_INFO("llamacpp",
                      std::string("device[") + std::to_string(i) + "]=" +
                          ggml_backend_reg_name(ggml_backend_reg_get(i)));
    }
    if (n_regs == 0) {
      PRESTO_LOG_WARN("llamacpp", "no ggml backends registered after load_all");
    }
  }

  impl_->n_ctx = static_cast<uint32_t>(env_int("PRESTO_CTX", 4096));
  const int32_t hw = static_cast<int32_t>(std::thread::hardware_concurrency());
  impl_->threads =
      env_int("PRESTO_THREADS", hw > 0 ? std::min<int32_t>(hw, 8) : 4);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = impl_->n_ctx;
  cparams.n_batch = impl_->n_ctx;
  cparams.n_threads = impl_->threads;
  cparams.n_threads_batch = impl_->threads;

  // Flash Attention: opt-in via PRESTO_FLASH_ATTN=1. Default stays AUTO -
  // forced FA measured slower than the naive path on some small models.
  if (env_int("PRESTO_FLASH_ATTN", 0) != 0) {
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  }

  // Optional KV-cache quantization: PRESTO_KV=q8_0|q4_0 trades a little
  // quality for lower memory bandwidth on decode (needs FA for V quant).
  if (const char* kv = std::getenv("PRESTO_KV"); kv && *kv) {
    const std::string s(kv);
    if (s == "q8_0") {
      cparams.type_k = GGML_TYPE_Q8_0;
      if (cparams.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED)
        cparams.type_v = GGML_TYPE_Q8_0;
    } else if (s == "q4_0") {
      cparams.type_k = GGML_TYPE_Q4_0;
      if (cparams.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED)
        cparams.type_v = GGML_TYPE_Q4_0;
    }
    PRESTO_LOG_INFO("llamacpp", "kv cache dtype override: " + s);
  }

  impl_->ctx = llama_init_from_model(impl_->model, cparams);
  if (!impl_->ctx && cparams.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED) {
    PRESTO_LOG_WARN("llamacpp", "flash attention rejected; retrying with auto");
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    impl_->ctx = llama_init_from_model(impl_->model, cparams);
  }
  if (!impl_->ctx) {
    err = "step=create_context msg=\"llama_init_from_model failed\"";
    return false;
  }

  // Spinning threadpools are OPT-IN (PRESTO_POLL=100): they pin cores at
  // 100% even while idle, which is ideal for latency-critical servers but
  // hostile to anything else sharing the machine. Off by default.
  if (env_int("PRESTO_POLL", 0) > 0 && !impl_->tpool) {
    auto tp = ggml_threadpool_params_default(impl_->threads);
    tp.poll = static_cast<uint32_t>(env_int("PRESTO_POLL", 0));
    impl_->tpool = ggml_threadpool_new(&tp);
    impl_->tpool_batch = ggml_threadpool_new(&tp);
    if (impl_->tpool) {
      llama_attach_threadpool(impl_->ctx, impl_->tpool, impl_->tpool_batch);
      PRESTO_LOG_INFO("llamacpp", "spinning threadpool: " +
                                      std::to_string(impl_->threads) + " threads, poll=" +
                                      std::to_string(tp.poll));
    }
  }

  // Optional speculative decoding companion (greedy-only acceleration).
  if (!spec_draft_path().empty()) {
    llama_model_params dparams = llama_model_default_params();
    impl_->d_model = llama_model_load_from_file(spec_draft_path().c_str(), dparams);
    if (!impl_->d_model) {
      PRESTO_LOG_WARN("llamacpp", "draft model failed to load; speculative decoding off");
    } else {
      impl_->d_vocab = llama_model_get_vocab(impl_->d_model);
      if (!impl_->d_vocab ||
          llama_n_vocab(impl_->d_vocab) != llama_n_vocab(impl_->vocab)) {
        PRESTO_LOG_WARN("llamacpp", "draft vocab mismatch; speculative decoding off");
        llama_model_free(impl_->d_model);
        impl_->d_model = nullptr;
      } else {
        llama_context_params dc = llama_context_default_params();
        dc.n_ctx = impl_->n_ctx;
        dc.n_batch = impl_->n_ctx;
        dc.n_threads = impl_->threads;
        dc.n_threads_batch = impl_->threads;
        impl_->d_ctx = llama_init_from_model(impl_->d_model, dc);
        if (!impl_->d_ctx) {
          PRESTO_LOG_WARN("llamacpp", "draft context failed; speculative decoding off");
          llama_model_free(impl_->d_model);
          impl_->d_model = nullptr;
        } else {
          PRESTO_LOG_INFO("llamacpp", "speculative decoding ready: draft=" +
                                          spec_draft_path());
        }
      }
    }
  }

  // Warmup decode: pages in weights/compute buffers so the first real request
  // does not pay cold-start costs. The BOS token stays resident in KV and
  // seeds the prefix cache. Failure here is non-fatal.
  {
    llama_token bos = llama_vocab_bos(impl_->vocab);
    llama_batch b = llama_batch_get_one(&bos, 1);
    if (llama_decode(impl_->ctx, b) != 0) {
      PRESTO_LOG_WARN("llamacpp", "warmup decode failed; continuing");
      llama_memory_clear(llama_get_memory(impl_->ctx), true);
      impl_->cache_valid = false;
    } else {
      impl_->cache_tokens.assign(1, bos);
      impl_->cache_valid = true;
    }
  }

  load_sec_ = now_sec() - t0;
  PRESTO_LOG_INFO("llamacpp",
                  "ready in " + std::to_string(load_sec_) + "s (ctx=" +
                      std::to_string(impl_->n_ctx) + ", threads=" +
                      std::to_string(impl_->threads) + ")");
  return true;
}

bool LlamaCppBackend::generate(const GenerateParams& gp, GenerateResult& r, std::string& err) {
  if (!impl_ || !impl_->model || !impl_->vocab || !impl_->ctx) {
    err = "step=state msg=\"backend not loaded\"";
    return false;
  }

  std::vector<llama_token> prompt;
  if (!gp.prompt_text.empty()) {
    const int32_t text_len = static_cast<int32_t>(gp.prompt_text.size());
    // llama.h contract: null buffer -> negative required count; INT32_MIN/-1
    // are hard failures. Some BPE vocabs without a usable BOS return -1 when
    // add_special=true, so retry once with specials disabled.
    bool add_special = true;
    int32_t need = llama_tokenize(impl_->vocab, gp.prompt_text.c_str(), text_len,
                                  nullptr, 0, add_special, true);
    if (need == -1 || need == INT32_MIN) {
      PRESTO_LOG_DEBUG("llamacpp", "tokenize sizing with specials returned " +
                                       std::to_string(need) + "; retrying without");
      add_special = false;
      need = llama_tokenize(impl_->vocab, gp.prompt_text.c_str(), text_len,
                            nullptr, 0, add_special, true);
    }
    if (need == INT32_MIN || (need < 0 && -need > (1 << 20))) {
      err = "step=tokenize msg=\"sizing pass failed (" + std::to_string(need) + ")\"";
      return false;
    }
    if (need < 0) need = -need;
    if (need <= 0) {
      err = "step=tokenize msg=\"empty tokenization\"";
      return false;
    }
    prompt.resize(static_cast<std::size_t>(need));
    int32_t n = llama_tokenize(impl_->vocab, gp.prompt_text.c_str(), text_len,
                               prompt.data(), need, add_special, true);
    if (n < 0 && n != INT32_MIN) {
      need = -n;
      prompt.resize(static_cast<std::size_t>(need));
      n = llama_tokenize(impl_->vocab, gp.prompt_text.c_str(), text_len,
                         prompt.data(), need, add_special, true);
    }
    if (n < 0) {
      err = "step=tokenize msg=\"tokenization failed after sizing\"";
      return false;
    }
    prompt.resize(static_cast<std::size_t>(n));
  } else {
    prompt.reserve(gp.prompt_tokens.size());
    for (int t : gp.prompt_tokens) prompt.push_back(static_cast<llama_token>(t));
  }
  if (prompt.empty()) {
    err = "step=prompt msg=\"empty prompt\"";
    return false;
  }

  // Recreate the context only when a request could exceed its capacity.
  const uint32_t needed = static_cast<uint32_t>(prompt.size() + gp.max_tokens + 8);
  if (needed > impl_->n_ctx) {
    uint32_t grown = impl_->n_ctx;
    while (grown < needed) grown *= 2;
    PRESTO_LOG_INFO("llamacpp",
                    "growing context " + std::to_string(impl_->n_ctx) + " -> " +
                        std::to_string(grown));
    llama_free(impl_->ctx);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = grown;
    cparams.n_batch = grown;
    cparams.n_threads = impl_->threads;
    cparams.n_threads_batch = impl_->threads;
    impl_->ctx = llama_init_from_model(impl_->model, cparams);
    if (!impl_->ctx) {
      impl_->ctx = nullptr;
      impl_->cache_valid = false;
      err = "step=grow_context msg=\"failed to recreate context at " +
            std::to_string(grown) + "\"";
      return false;
    }
    impl_->n_ctx = grown;
    impl_->cache_valid = false;  // fresh memory: nothing resident to reuse
    if (impl_->tpool) {
      llama_attach_threadpool(impl_->ctx, impl_->tpool, impl_->tpool_batch);
    }
  }

  // ---- prefix-cache aware prefill ----
  // KV entries are exact functions of their token prefix, so a shared token
  // prefix with the previous request can skip re-evaluation of those tokens
  // and produce identical results.
  llama_memory_t mem = llama_get_memory(impl_->ctx);
  const bool cache_allowed =
      [] {
        const char* v = std::getenv("PRESTO_PREFIX_CACHE");
        return !(v && *v && v[0] == '0');
      }();

  size_t feed_from = 0;
  if (cache_allowed && impl_->cache_valid) {
    size_t common = 0;
    const size_t limit = std::min(impl_->cache_tokens.size(), prompt.size());
    while (common < limit && impl_->cache_tokens[common] == prompt[common]) ++common;
    // keep [0, common), drop everything after it
    if (!llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(common), -1)) {
      common = 0;  // partial removal unsupported by backend: full re-prefill
      if (!llama_memory_seq_rm(mem, 0, -1, -1)) {
        err = "step=prefill msg=\"failed to reset kv memory\"";
        return false;
      }
    }
    feed_from = common;
    if (feed_from > 0) {
      PRESTO_LOG_INFO("llamacpp",
                      "prefix reuse: kept " + std::to_string(feed_from) + "/" +
                          std::to_string(prompt.size()) + " tokens");
    }
  } else {
    llama_memory_clear(mem, true);
  }
  // identical prompt to last request leaves zero new tokens; force at least
  // one eval by dropping back one position
  if (cache_allowed && impl_->cache_valid && feed_from >= prompt.size()) {
    feed_from = prompt.size() >= 1 ? prompt.size() - 1 : 0;
    llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(feed_from), -1);
  }

  const double prefill_t0 = now_sec();
  int32_t processed = static_cast<int32_t>(feed_from);
  while (processed < static_cast<int32_t>(prompt.size())) {
    const int32_t chunk =
        std::min<int32_t>(static_cast<int32_t>(impl_->n_ctx),
                          static_cast<int32_t>(prompt.size()) - processed);
    if (llama_decode(impl_->ctx,
                     llama_batch_get_one(prompt.data() + processed, chunk))) {
      err = "step=prefill msg=\"decode failed at offset " + std::to_string(processed) + "\"";
      impl_->cache_valid = false;
      return false;
    }
    processed += chunk;
  }
  r.prefill_sec = now_sec() - prefill_t0;

  const int n_vocab = llama_n_vocab(impl_->vocab);
  unsigned rng_state =
      gp.seed >= 0
          ? static_cast<unsigned>(gp.seed)
          : static_cast<unsigned>(
                std::chrono::steady_clock::now().time_since_epoch().count());

  // ---- decode ----
  const double decode_t0 = now_sec();
  std::vector<llama_token> out_tokens;
  out_tokens.reserve(static_cast<std::size_t>(std::max(0, gp.max_tokens)));
  llama_memory_t tmem = impl_->ctx ? llama_get_memory(impl_->ctx) : nullptr;
  llama_memory_t dmem = impl_->d_ctx ? llama_get_memory(impl_->d_ctx) : nullptr;

  // Greedy + draft companion available -> speculative decoding. Outputs are
  // bit-identical to the loop below (greedy equivalence), just cheaper.
  if (impl_->d_ctx && gp.temp <= 0.f && gp.max_tokens >= 4) {
    const int K = env_int("PRESTO_SPEC_K", 6);
    llama_memory_t tmem = llama_get_memory(impl_->ctx);
    llama_memory_t dmem = llama_get_memory(impl_->d_ctx);

    // draft prefill: same prompt (v1 = full re-prefill per request)
    llama_memory_clear(llama_get_memory(impl_->d_ctx), true);
    impl_->d_cache_tokens.clear();
    for (int32_t off = 0; off < static_cast<int32_t>(prompt.size());
         off += static_cast<int32_t>(impl_->n_ctx)) {
      const int32_t chunk = std::min<int32_t>(
          static_cast<int32_t>(impl_->n_ctx),
          static_cast<int32_t>(prompt.size()) - off);
      if (llama_decode(impl_->d_ctx,
                       llama_batch_get_one(prompt.data() + off, chunk))) {
        PRESTO_LOG_WARN("llamacpp", "draft prefill failed; spec decoding disabled");
        llama_free(impl_->d_ctx);
        impl_->d_ctx = nullptr;
      }
    }
    if (!impl_->d_ctx) {
      err = "step=spec msg=\"draft prefill failed\"";
      return false;
    }
    impl_->d_cache_tokens = prompt;
  }

  bool speculative = false;
  bool plain_fallback = false;
  if (impl_->d_ctx && gp.temp <= 0.f && gp.max_tokens >= 4) {
    speculative = true;
    const int Kmax = env_int("PRESTO_SPEC_K", 6);

    std::vector<llama_token> fed = prompt;  // exact mirror of target KV

    // draft prefill: same prompt (v1 = full re-prefill per request)
    llama_memory_clear(llama_get_memory(impl_->d_ctx), true);
    impl_->d_cache_tokens.clear();
    for (int32_t off = 0; off < static_cast<int32_t>(prompt.size());
         off += static_cast<int32_t>(impl_->n_ctx)) {
      const int32_t chunk = std::min<int32_t>(
          static_cast<int32_t>(impl_->n_ctx),
          static_cast<int32_t>(prompt.size()) - off);
      if (llama_decode(impl_->d_ctx,
                       llama_batch_get_one(prompt.data() + off, chunk))) {
        PRESTO_LOG_WARN("llamacpp", "draft prefill failed; spec decoding off");
        llama_free(impl_->d_ctx);
        impl_->d_ctx = nullptr;
        break;
      }
    }
    if (!impl_->d_ctx) {
      err = "step=spec msg=\"draft prefill failed\"";
      return false;
    }
    impl_->d_cache_tokens = prompt;

    auto decode_single_at = [&](llama_context* c, llama_token tok, llama_pos pos) {
      llama_batch b = llama_batch_init(1, 0, 1);
      b.n_tokens = 1;
      b.token[0] = tok;
      b.pos[0] = pos;
      b.n_seq_id[0] = 1;
      b.seq_id[0][0] = 0;
      b.logits[0] = true;
      const int rc = llama_decode(c, b);
      llama_batch_free(b);
      return rc != 0;
    };
    auto draft_argmax_last = [&] {
      const float* dl = llama_get_logits_ith(impl_->d_ctx, -1);
      int best = 0;
      for (int v = 1; v < n_vocab; ++v)
        if (dl[v] > dl[best]) best = v;
      return static_cast<llama_token>(best);
    };
    auto target_argmax_row = [&](int row) {
      const float* lg = llama_get_logits_ith(impl_->ctx, row);
      int best = 0;
      for (int v = 1; v < n_vocab; ++v)
        if (lg[v] > lg[best]) best = v;
      return static_cast<llama_token>(best);
    };

    // emit the target's first greedy token from the prefill logits, feeding
    // it into both models so every later round starts uniform.
    llama_token pred_carry = 0;
    {
      const llama_token first_tok = target_argmax_row(-1);
      if (llama_vocab_is_eog(impl_->vocab, first_tok)) {
        impl_->cache_valid = false;
        return true;
      }
      llama_token f1 = first_tok;
      if (decode_single_at(impl_->ctx, f1, static_cast<llama_pos>(fed.size()))) {
        err = "step=spec msg=\"first decode failed\"";
        impl_->cache_valid = false;
        return false;
      }
      if (decode_single_at(impl_->d_ctx, f1,
                           static_cast<llama_pos>(impl_->d_cache_tokens.size()))) {
        err = "step=spec msg=\"draft first decode failed\"";
        llama_free(impl_->d_ctx);
        impl_->d_ctx = nullptr;
        return false;
      }
      out_tokens.push_back(first_tok);
      fed.push_back(first_tok);
      impl_->d_cache_tokens.push_back(first_tok);
      pred_carry = target_argmax_row(-1);
    }

    while (static_cast<int>(out_tokens.size()) < gp.max_tokens) {
      const int room = gp.max_tokens - static_cast<int>(out_tokens.size());
      const int K = std::min(Kmax, std::max(room - 1, 1));
      PRESTO_LOG_DEBUG("llamacpp",
                       "round-start out=" + std::to_string(out_tokens.size()) +
                           " room=" + std::to_string(room) + " K=" + std::to_string(K));
      const llama_pos base = static_cast<llama_pos>(fed.size());
      if (base + K + 1 > static_cast<llama_pos>(impl_->n_ctx)) break;

      // --- draft proposals: continue from the token D last absorbed ---
      std::vector<llama_token> props;
      llama_token dtok = draft_argmax_last();
      while (static_cast<int>(props.size()) < K) {
        if (llama_vocab_is_eog(impl_->vocab, dtok)) break;
        props.push_back(dtok);
        if (static_cast<int>(props.size()) >= K) break;
        const llama_pos dpos = static_cast<llama_pos>(impl_->d_cache_tokens.size());
        if (decode_single_at(impl_->d_ctx, dtok, dpos)) break;
        impl_->d_cache_tokens.push_back(dtok);
      }
      const int n = static_cast<int>(props.size());
      if (n == 0) {
        PRESTO_LOG_WARN("llamacpp", "draft produced no proposals; spec off");
        llama_free(impl_->d_ctx);
        impl_->d_ctx = nullptr;
        break;
      }

      // --- verify: feed d_1..d_n together, every row carries logits ---
      // row i (position base+i, content d_{i+1}) predicts slot base+i+1,
      // i.e. the correctness of proposal d_{i+2}.
      if (!llama_memory_seq_rm(tmem, 0, base, -1)) {
        err = "step=spec msg=\"target seq_rm failed\"";
        impl_->cache_valid = false;
        return false;
      }
      llama_batch vb = llama_batch_init(n, 0, 1);
      vb.n_tokens = n;
      for (int i = 0; i < n; ++i) {
        vb.token[i] = props[i];
        vb.pos[i] = base + i;
        vb.n_seq_id[i] = 1;
        vb.seq_id[i][0] = 0;
        vb.logits[i] = true;
      }
      const int dec = llama_decode(impl_->ctx, vb);
      llama_batch_free(vb);
      if (dec) {
        err = "step=spec msg=\"verify decode failed\"";
        impl_->cache_valid = false;
        return false;
      }

      // acceptance walk: pred_carry validates d_1; each accepted proposal
      // shifts validation to the next row; the final row yields the
      // correction token that follows the longest accepted chain.
      int acc = 0;
      llama_token pred = pred_carry;
      while (acc < n && pred == props[acc]) {
        ++acc;
        pred = target_argmax_row(acc < n ? acc - 1 : n - 1);
      }
      if (acc == n) pred = target_argmax_row(n - 1);

      const bool stop = llama_vocab_is_eog(impl_->vocab, pred);

      // commit accepted chain, then feed the correction token so the next
      // round starts from a uniform state (logits(-1) valid again).
      for (int i = 0; i < acc; ++i) {
        out_tokens.push_back(props[i]);
        fed.push_back(props[i]);
      }
      if (!stop) {
        out_tokens.push_back(pred);
        fed.push_back(pred);
        // slot base+acc still holds the rejected proposal; clear it and the
        // tail, then write `pred` at the exact position. Hybrid caches
        // (Qwen3.5-style) may not support partial removal at all - in that
        // case fall back to a full re-prefill of the accepted history.
        if (!llama_memory_seq_rm(tmem, 0, base + acc, -1)) {
          PRESTO_LOG_WARN("llamacpp",
                          "kv rollback unsupported by this architecture; "
                          "disabling speculative decoding for this request");
          llama_memory_clear(tmem, true);
          impl_->cache_valid = false;
          for (int32_t off = 0; off < static_cast<int32_t>(fed.size());
               off += static_cast<int32_t>(impl_->n_ctx)) {
            const int32_t chunk = std::min<int32_t>(
                static_cast<int32_t>(impl_->n_ctx),
                static_cast<int32_t>(fed.size()) - off);
            llama_batch rb = llama_batch_init(chunk, 0, 1);
            rb.n_tokens = chunk;
            for (int i = 0; i < chunk; ++i) {
              rb.token[i] = fed[off + i];
              rb.pos[i] = off + i;
              rb.n_seq_id[i] = 1;
              rb.seq_id[i][0] = 0;
              rb.logits[i] = (off + i == static_cast<int>(fed.size()) - 1);
            }
            const int rc = llama_decode(impl_->ctx, rb);
            llama_batch_free(rb);
            if (rc) {
              err = "step=spec msg=\"fallback re-prefill failed\"";
              return false;
            }
          }
          pred_carry = target_argmax_row(-1);
          PRESTO_LOG_DEBUG("llamacpp",
                           "fallback entry out=" +
                               std::to_string(out_tokens.size()));
          plain_fallback = true;
          break;
        }
        if (decode_single_at(impl_->ctx, pred, base + acc)) {
          err = "step=spec msg=\"correction decode failed\"";
          impl_->cache_valid = false;
          return false;
        }
      } else {
        if (!llama_memory_seq_rm(tmem, 0, base + acc, -1)) {
          llama_memory_clear(tmem, true);
          impl_->cache_valid = false;
        }
      }

      // draft mirror: rewind rejected proposals, absorb `pred`, so D stays
      // token-aligned with T for the next round of proposals.
      const size_t keep_d =
          impl_->d_cache_tokens.size() - static_cast<size_t>(n - acc);
      llama_memory_seq_rm(dmem, 0, static_cast<llama_pos>(keep_d), -1);
      impl_->d_cache_tokens.resize(keep_d);
      if (!stop) {
        llama_token nx = pred;
        if (decode_single_at(impl_->d_ctx, nx, static_cast<llama_pos>(keep_d))) {
          PRESTO_LOG_WARN("llamacpp", "draft resync failed; spec off");
          llama_free(impl_->d_ctx);
          impl_->d_ctx = nullptr;
        } else {
          impl_->d_cache_tokens.push_back(nx);
        }
      }

      if (stop || !impl_->d_ctx) break;
      // T just absorbed `pred`; refresh the carried prediction from its row.
      pred_carry = target_argmax_row(-1);
    }

    // the final correction is intentionally left unfed when stopped; the
    // prefix cache is invalidated rather than allowed to lie about state.
    impl_->cache_valid = false;
  }

  if (!speculative || plain_fallback) {
    // bound by REMAINING tokens: the spec-fallback path arrives here with
    // part of the sequence already emitted.
    const int remaining = gp.max_tokens - static_cast<int>(out_tokens.size());
    for (int step = 0; step < remaining; ++step) {
      const float* logits = llama_get_logits_ith(impl_->ctx, -1);
      if (!logits) {
        err = "step=get_logits msg=\"null logits at step " + std::to_string(step) + "\"";
        return false;
      }
      const llama_token next = sample_token(logits, n_vocab, gp.temp, rng_state);
      if (llama_vocab_is_eog(impl_->vocab, next)) break;
      out_tokens.push_back(next);

      if (step + 1 < gp.max_tokens) {
        if (llama_decode(impl_->ctx, llama_batch_get_one(&out_tokens.back(), 1))) {
          err = "step=decode msg=\"decode failed at step " + std::to_string(step) + "\"";
          impl_->cache_valid = false;
          return false;
        }
      }
    }
  }
  r.decode_sec = now_sec() - decode_t0;

  if (!out_tokens.empty()) {
    int32_t need = 512;
    std::vector<char> buf(static_cast<std::size_t>(need));
    for (int attempt = 0; attempt < 4; ++attempt) {
      const int32_t n = llama_detokenize(impl_->vocab, out_tokens.data(),
                                         static_cast<int32_t>(out_tokens.size()), buf.data(),
                                         need, false, false);
      if (n >= 0) {
        r.text.assign(buf.data(), static_cast<std::size_t>(n));
        break;
      }
      need = -n;
      buf.resize(static_cast<std::size_t>(need));
      if (attempt == 3) PRESTO_LOG_WARN("llamacpp", "detokenize kept failing; text omitted");
    }
  }

  // memory now holds prompt+generated; publish it as reusable history
  impl_->cache_tokens = prompt;
  impl_->cache_tokens.insert(impl_->cache_tokens.end(), out_tokens.begin(),
                             out_tokens.end());
  impl_->cache_valid = true;

  r.tokens.assign(out_tokens.begin(), out_tokens.end());
  r.tok_per_sec =
      r.decode_sec > 0 ? static_cast<double>(out_tokens.size()) / r.decode_sec : 0.0;
  r.load_sec = load_sec_;
  return true;
}

std::unique_ptr<IBackend> make_llamacpp_backend(const std::string& path) {
  return std::make_unique<LlamaCppBackend>(path);
}

} // namespace presto
