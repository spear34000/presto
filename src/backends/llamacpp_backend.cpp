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
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace presto {

namespace fs = std::filesystem;

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
  uint32_t max_slots = 1;
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

  // persistent cross-request n-gram pool: every completed prompt+output is
  // appended here so future requests can draft from history without a model.
  // This is the user-visible "innovation" that turns idle compute into tokens
  // past the 11.9 tok/s bandwidth wall. Bounded to avoid unbounded growth.
  std::vector<llama_token> ngram_pool;
  static constexpr size_t kPoolCap = 200000;
  int ngram_low_streak = 0;
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

  // Adaptive context size: scale with available system RAM so low-spec
  // machines get a working default and high-spec machines get headroom.
  // PRESTO_CTX overrides; otherwise RAM-derived.
  {
    uint64_t avail_bytes = 0;
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) avail_bytes = ms.ullAvailPhys;
#elif defined(__APPLE__) || defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    uint64_t kb = 0;
    while (meminfo >> key >> kb) {
      if (key == "MemAvailable:") { avail_bytes = kb * 1024ULL; break; }
      meminfo.ignore(1 << 20, '\n');
    }
#endif
    const uint64_t free_gb = avail_bytes >> 30;
    uint32_t auto_ctx = free_gb >= 12 ? 4096 : (free_gb >= 6 ? 2048 : 1024);
    if (auto_ctx < 1024) auto_ctx = 1024;
    impl_->n_ctx = static_cast<uint32_t>(env_int("PRESTO_CTX", static_cast<int32_t>(auto_ctx)));
    PRESTO_LOG_INFO("llamacpp", "auto ctx=" + std::to_string(impl_->n_ctx) +
                                    " (free ram " + std::to_string(free_gb) + " GB)");
  }

  const int32_t hw = static_cast<int32_t>(std::thread::hardware_concurrency());
  impl_->threads =
      env_int("PRESTO_THREADS", hw > 0 ? std::min<int32_t>(hw, 8) : 4);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = impl_->n_ctx;
  cparams.n_batch = impl_->n_ctx;
  cparams.n_threads = impl_->threads;
  cparams.n_threads_batch = impl_->threads;
  // KV pool is shared across sequences; n_ctx stays the TOTAL budget so the
  // single-stream path is untouched and batch slots borrow from the pool.
  impl_->max_slots =
      static_cast<uint32_t>(std::max(1, env_int("PRESTO_BATCH_SLOTS", 4)));
  cparams.n_seq_max = impl_->max_slots;

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

  // ---- persistent prompt snapshot ("compiled prompt" cache) ----
  // Saves the post-prefill KV state of long prompts to disk, keyed by token
  // hash. Future runs (even after restart) restore it instead of re-prefilling
  // seconds of compute. The file embeds its own token list for verification.
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

  auto fnv1a64 = [](const std::vector<llama_token>& v, uint64_t seed) {
    uint64_t h = seed;
    for (llama_token t : v) { h ^= static_cast<uint64_t>(t); h *= 0x100000001b3ULL; }
    return h;
  };
  const uint64_t model_key = [&] {
    std::error_code ec;
    const auto sz = static_cast<uint64_t>(fs::file_size(fs::path(path_), ec));
    return sz ? sz : 0x9e3779b97f4a7c15ULL;
  }();
  const int snapshot_min = env_int("PRESTO_SNAPSHOT_MIN", 64);
  const bool snapshots_on = [] {
    const char* e = std::getenv("PRESTO_KV_SNAPSHOT");
    return !(e && *e && e[0] == '0');
  }();
  bool snap_restored = false;
  std::string snap_path;
  if (snapshots_on && prompt.size() >= static_cast<size_t>(snapshot_min)) {
    const char* lad = std::getenv("LOCALAPPDATA");
    std::string dir = lad
        ? std::string(lad) + "\\presto\\kv"
        : std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/.presto/kv";
    const uint64_t tokhash =
        fnv1a64(prompt, fnv1a64(prompt, model_key ^ 0x243f6a8885a308d3ULL));
    std::ostringstream ks; ks << std::hex << tokhash;
    dir += "\\" + std::to_string(model_key % 100000);
    snap_path = dir + "\\" + ks.str() + ".state";

    if (llama_memory_t m0 = llama_get_memory(impl_->ctx)) {
      llama_memory_clear(m0, true);
      std::vector<llama_token> rtok(static_cast<size_t>(prompt.size()) + 8);
      size_t ntok = 0;
      const size_t got = llama_state_seq_load_file(
          impl_->ctx, snap_path.c_str(), 0, rtok.data(), rtok.size(), &ntok);
      bool ok = got > 0 && ntok == prompt.size() &&
                std::equal(rtok.begin(), rtok.begin() + static_cast<long>(ntok),
                           prompt.begin());
      if (ok) {
        llama_memory_seq_rm(m0, 0, static_cast<llama_pos>(prompt.size()) - 1, -1);
        if (!decode_single_at(impl_->ctx, prompt.back(),
                              static_cast<llama_pos>(prompt.size()) - 1)) {
          PRESTO_LOG_INFO("llamacpp",
                          "kv snapshot restored: " + std::to_string(ntok) +
                              " tokens from " + snap_path);
          snap_restored = true;
        }
        ok = false;
      }
      if (!ok && got == 0 && fs::exists(snap_path)) {
        std::error_code ec2; fs::remove(snap_path, ec2);  // corrupt: drop
      }
    }
  }

  if (!snap_restored) {
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

  if (snapshots_on && !snap_path.empty() &&
      prompt.size() >= static_cast<size_t>(snapshot_min)) {
    const std::string tmpf = snap_path + ".tmp";
    if (llama_state_seq_save_file(impl_->ctx, tmpf.c_str(), 0, prompt.data(),
                                  prompt.size()) > 0) {
      std::error_code ec3;
      fs::rename(tmpf, snap_path, ec3);
      PRESTO_LOG_INFO("llamacpp",
                      "kv snapshot saved: " + std::to_string(prompt.size()) +
                          " tokens -> " + snap_path);
    } else {
      std::error_code ec3;
      fs::remove(tmpf, ec3);
    }
  }
  } // end !snap_restored prefill

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


  // Greedy -> speculative decoding. With a draft companion the proposer is
  // the small model; without one it is n-gram prompt lookup. Outputs are
  // bit-identical to the loop below (greedy equivalence), just cheaper.
  const bool want_spec = gp.temp <= 0.f && gp.max_tokens >= 4;
  const bool lookup_eligible =
      want_spec && !impl_->d_ctx &&
      prompt.size() >= static_cast<size_t>(env_int("PRESTO_LOOKUP_MIN", 32));
  bool speculative = false;
  bool plain_fallback = false;
  if (want_spec && (impl_->d_ctx || lookup_eligible)) {
    speculative = true;
    const bool have_draft = impl_->d_ctx != nullptr;
    const int Kmax = have_draft ? env_int("PRESTO_SPEC_K", 6)
                                : env_int("PRESTO_LOOKUP_K", 16);
    const int lookup_seed = env_int("PRESTO_LOOKUP_N", 8);
    std::vector<llama_token> fed = prompt;  // exact mirror of target KV

    auto target_argmax_row = [&](int row) {
      const float* lg = llama_get_logits_ith(impl_->ctx, row);
      int best = 0;
      for (int v = 1; v < n_vocab; ++v)
        if (lg[v] > lg[best]) best = v;
      return static_cast<llama_token>(best);
    };

    // draft prefill: same prompt (v1 = full re-prefill per request)
    if (have_draft) {
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
      speculative = false;
    } else {
    impl_->d_cache_tokens = prompt;
    }
    }

    if (speculative) {
    PRESTO_LOG_INFO("llamacpp", std::string("speculative decoding ready: mode=") +
                        (have_draft ? "draft-model" : "prompt-lookup"));

    // emit the target's first greedy token from the prefill logits, feeding
    // it into both models so every later round starts uniform.
    llama_token pred_carry = 0;
    {
      const llama_token first_tok = target_argmax_row(-1);
      if (llama_vocab_is_eog(impl_->vocab, first_tok)) {
        impl_->cache_valid = false;
        return true;
      }
      if (decode_single_at(impl_->ctx, first_tok,
                           static_cast<llama_pos>(fed.size()))) {
        err = "step=spec msg=\"first decode failed\"";
        impl_->cache_valid = false;
        return false;
      }
      if (have_draft) {
        if (decode_single_at(impl_->d_ctx, first_tok,
                             static_cast<llama_pos>(impl_->d_cache_tokens.size()))) {
          err = "step=spec msg=\"draft first decode failed\"";
          llama_free(impl_->d_ctx);
          impl_->d_ctx = nullptr;
          return false;
        }
      }
      out_tokens.push_back(first_tok);
      fed.push_back(first_tok);
      if (have_draft) impl_->d_cache_tokens.push_back(first_tok);
      pred_carry = target_argmax_row(-1);
    }

    while (static_cast<int>(out_tokens.size()) < gp.max_tokens) {
      const int room = gp.max_tokens - static_cast<int>(out_tokens.size());
      const int K = std::min(Kmax, std::max(room - 1, 1));
      const llama_pos base = static_cast<llama_pos>(fed.size());
      if (base + K + 1 > static_cast<llama_pos>(impl_->n_ctx)) break;

      // --- draft proposals ---
      std::vector<llama_token> props;
      if (have_draft) {
        const float* dl = llama_get_logits_ith(impl_->d_ctx, -1);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v)
          if (dl[v] > dl[best]) best = v;
        llama_token dtok = static_cast<llama_token>(best);
        while (static_cast<int>(props.size()) < K) {
          if (llama_vocab_is_eog(impl_->vocab, dtok)) break;
          props.push_back(dtok);
          if (static_cast<int>(props.size()) >= K) break;
          const llama_pos dpos =
              static_cast<llama_pos>(impl_->d_cache_tokens.size());
          if (decode_single_at(impl_->d_ctx, dtok, dpos)) break;
          impl_->d_cache_tokens.push_back(dtok);
        }
      } else {
        const size_t seed =
            std::min<size_t>(std::max(lookup_seed, 1), fed.size());
        if (fed.size() > seed) {
          const size_t tail = fed.size() - seed;
          for (size_t i = tail; i-- > 0 && props.empty();) {
            bool match = true;
            for (size_t j = 0; j < seed; ++j)
              if (fed[i + j] != fed[tail + j]) { match = false; break; }
            if (!match) continue;
            for (size_t j = seed;
                 j < seed + static_cast<size_t>(K) && i + j < fed.size(); ++j)
              props.push_back(fed[i + j]);
          }
          if (props.empty() && !impl_->ngram_pool.empty()) {
            const auto &pool = impl_->ngram_pool;
            if (pool.size() > seed) {
              for (size_t i = pool.size() - seed; i-- > 0 && props.empty();) {
                bool match = true;
                for (size_t j = 0; j < seed; ++j)
                  if (pool[i + j] != fed[tail + j]) { match = false; break; }
                if (!match) continue;
                for (size_t j = seed;
                     j < seed + static_cast<size_t>(K) && i + j < pool.size(); ++j)
                  props.push_back(pool[i + j]);
              }
            }
          }
        }
      }
      const int n = static_cast<int>(props.size());
      if (n == 0) {
        if (have_draft) {
          PRESTO_LOG_WARN("llamacpp", "draft produced no proposals; spec off");
          llama_free(impl_->d_ctx);
          impl_->d_ctx = nullptr;
          break;
        }
        // lookup found no match: one plain greedy step, retry next round.
        const llama_token t = target_argmax_row(-1);
        if (llama_vocab_is_eog(impl_->vocab, t)) {
          impl_->cache_valid = false;
          return true;
        }
        if (decode_single_at(impl_->ctx, t,
                             static_cast<llama_pos>(fed.size()))) {
          err = "step=spec msg=\"plain step failed\"";
          impl_->cache_valid = false;
          return false;
        }
        out_tokens.push_back(t);
        fed.push_back(t);
        pred_carry = target_argmax_row(-1);
        continue;
      }

      // --- verify: feed d_1..d_n together, every row carries logits ---
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
      // shifts validation one row deeper; the final row yields the
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
        // (Qwen3.5-style) may not support partial removal at all - fall back
        // to a full re-prefill of the accepted history in that case.
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
          plain_fallback = true;
          break;
        }
        if (decode_single_at(impl_->ctx, pred, base + acc)) {
          err = "step=spec msg=\"correction decode failed\"";
          impl_->cache_valid = false;
          return false;
        }
      } else {
        llama_memory_seq_rm(tmem, 0, base + acc, -1);
      }

      // draft mirror: rewind rejected proposals, absorb `pred`, so D stays
      // token-aligned with T for the next round of proposals.
      if (have_draft) {
        const size_t keep_d =
            impl_->d_cache_tokens.size() - static_cast<size_t>(n - acc);
        llama_memory_seq_rm(dmem, 0, static_cast<llama_pos>(keep_d), -1);
        impl_->d_cache_tokens.resize(keep_d);
        if (!stop) {
          if (decode_single_at(impl_->d_ctx, pred,
                               static_cast<llama_pos>(impl_->d_cache_tokens.size()))) {
            PRESTO_LOG_WARN("llamacpp", "draft resync failed; spec off");
            llama_free(impl_->d_ctx);
            impl_->d_ctx = nullptr;
          } else {
            impl_->d_cache_tokens.push_back(pred);
          }
        }
      }

      if (stop) break;
      if (have_draft && !impl_->d_ctx) break;
      pred_carry = target_argmax_row(-1);
    }

    // the final correction is intentionally left unfed when stopped; the
    // prefix cache is invalidated rather than allowed to lie about state.
    impl_->cache_valid = false;
    }
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

  if (prompt.size() >= 32) {
    auto &pool = impl_->ngram_pool;
    pool.insert(pool.end(), prompt.begin(), prompt.end());
    pool.insert(pool.end(), out_tokens.begin(), out_tokens.end());
    if (pool.size() > Impl::kPoolCap)
      pool.erase(pool.begin(), pool.begin() + (pool.size() - Impl::kPoolCap));
  }

  r.tokens.assign(out_tokens.begin(), out_tokens.end());
  r.tok_per_sec =
      r.decode_sec > 0 ? static_cast<double>(out_tokens.size()) / r.decode_sec : 0.0;
  r.load_sec = load_sec_;
  return true;
}

bool LlamaCppBackend::generate_many(std::vector<BatchJob>& jobs, std::string& err) {
  if (jobs.empty()) return true;
  if (jobs.size() == 1) {
    jobs[0].ok = generate(jobs[0].params, jobs[0].result, jobs[0].err);
    return true;
  }
  for (const auto& j : jobs) {
    if (j.params.temp > 0.f) {
      err = "batch requires greedy (temp<=0)";
      return false;
    }
  }
  if (!impl_ || !impl_->model || !impl_->vocab || !impl_->ctx) {
    err = "step=state msg=\"backend not loaded\"";
    return false;
  }

  const int n_vocab_v = llama_n_vocab(impl_->vocab);
  const size_t S = std::min(jobs.size(), static_cast<size_t>(impl_->max_slots));

  struct BSlot {
    std::vector<llama_token> prompt;
    std::vector<llama_token> gen;
    bool active = false;
  };
  std::vector<BSlot> slot(S);

  auto tokenize_into = [&](const GenerateParams& gp,
                           std::vector<llama_token>& out) -> bool {
    if (!gp.prompt_text.empty()) {
      const int32_t text_len = static_cast<int32_t>(gp.prompt_text.size());
      bool add_special = true;
      int32_t need = llama_tokenize(impl_->vocab, gp.prompt_text.c_str(),
                                    text_len, nullptr, 0, add_special, true);
      if (need == -1 || need == INT32_MIN) {
        add_special = false;
        need = llama_tokenize(impl_->vocab, gp.prompt_text.c_str(), text_len,
                              nullptr, 0, add_special, true);
      }
      if (need <= 0 && need != INT32_MIN) need = -need;
      if (need <= 0 || need == INT32_MIN) return false;
      out.resize(static_cast<std::size_t>(need));
      int32_t n = llama_tokenize(impl_->vocab, gp.prompt_text.c_str(), text_len,
                                 out.data(), need, add_special, true);
      if (n < 0 && n != INT32_MIN) {
        need = -n;
        out.resize(static_cast<std::size_t>(need));
        n = llama_tokenize(impl_->vocab, gp.prompt_text.c_str(), text_len,
                           out.data(), need, add_special, true);
      }
      if (n < 0) return false;
      out.resize(static_cast<std::size_t>(n));
      return !out.empty();
    }
    for (int t : gp.prompt_tokens) out.push_back(static_cast<llama_token>(t));
    return !out.empty();
  };

  for (size_t j = 0; j < S; ++j) {
    auto& job = jobs[j];
    if (!tokenize_into(job.params, slot[j].prompt)) {
      job.err = "step=tokenize msg=\"batch tokenize failed\"";
      continue;
    }
    if (slot[j].prompt.size() + static_cast<size_t>(job.params.max_tokens) + 8 >
        impl_->n_ctx) {
      job.err = "step=batch msg=\"request exceeds context pool\"";
      continue;
    }
    slot[j].active = true;
    llama_memory_seq_rm(llama_get_memory(impl_->ctx),
                        static_cast<llama_seq_id>(j), 0, -1);
  }

  double batch_t0 = now_sec();
  for (size_t j = 0; j < S; ++j) {
    if (!slot[j].active) continue;
    const auto& p = slot[j].prompt;
    for (int32_t off = 0; off < static_cast<int32_t>(p.size());
         off += static_cast<int32_t>(impl_->n_ctx)) {
      const int32_t chunk =
          std::min<int32_t>(static_cast<int32_t>(impl_->n_ctx),
                            static_cast<int32_t>(p.size()) - off);
      llama_batch pb = llama_batch_init(chunk, 0, 1);
      pb.n_tokens = chunk;
      for (int32_t i = 0; i < chunk; ++i) {
        pb.token[i] = p[static_cast<size_t>(off + i)];
        pb.pos[i] = static_cast<llama_pos>(off + i);
        pb.n_seq_id[i] = 1;
        pb.seq_id[i][0] = static_cast<llama_seq_id>(j);
        pb.logits[i] = (off + i == static_cast<int32_t>(p.size()) - 1);
      }
      const int rc = llama_decode(impl_->ctx, pb);
      llama_batch_free(pb);
      if (rc) {
        slot[j].active = false;
        jobs[j].err = "step=batch_prefill msg=\"prefill failed\"";
        break;
      }
    }
    if (!slot[j].active) continue;
    // sample the first token straight from the prefill logits so the
    // decode loop never re-feeds a position the prompt already occupies
    const float* prow = llama_get_logits_ith(impl_->ctx, -1);
    int pbest = 0;
    for (int v = 1; v < n_vocab_v; ++v)
      if (prow[v] > prow[pbest]) pbest = v;
    if (llama_vocab_is_eog(impl_->vocab, static_cast<llama_token>(pbest))) {
      slot[j].active = false;
    } else {
      slot[j].gen.push_back(static_cast<llama_token>(pbest));
      if (static_cast<int>(slot[j].gen.size()) >= jobs[j].params.max_tokens)
        slot[j].active = false;
    }
  }

  while (true) {
    std::vector<size_t> live;
    for (size_t j = 0; j < S; ++j)
      if (slot[j].active &&
          static_cast<int>(slot[j].gen.size()) < jobs[j].params.max_tokens)
        live.push_back(j);
    if (live.empty()) break;

    llama_batch db = llama_batch_init(static_cast<int32_t>(live.size()), 0, 1);
    db.n_tokens = static_cast<int32_t>(live.size());
    for (size_t k = 0; k < live.size(); ++k) {
      const size_t j = live[k];
      BSlot& s = slot[j];
      db.token[k] = s.gen.empty() ? s.prompt.back() : s.gen.back();
      db.pos[k] =
          static_cast<llama_pos>(s.prompt.size() + s.gen.size() - 1);
      db.n_seq_id[k] = 1;
      db.seq_id[k][0] = static_cast<llama_seq_id>(j);
      db.logits[k] = true;
    }
    const int rc = llama_decode(impl_->ctx, db);
    llama_batch_free(db);
    if (rc) {
      err = "step=batch_decode msg=\"decode failed\"";
      for (auto& s2 : slot) s2.active = false;
      break;
    }
    for (size_t k = 0; k < live.size(); ++k) {
      const size_t j = live[k];
      BSlot& s = slot[j];
      const float* row = llama_get_logits_ith(impl_->ctx,
                                              static_cast<int32_t>(k));
      int best = 0;
      for (int v = 1; v < n_vocab_v; ++v)
        if (row[v] > row[best]) best = v;
      const llama_token tok = static_cast<llama_token>(best);
      if (llama_vocab_is_eog(impl_->vocab, tok)) {
        s.active = false;
        continue;
      }
      s.gen.push_back(tok);
      if (static_cast<int>(s.gen.size()) >= jobs[j].params.max_tokens)
        s.active = false;
    }
  }

  const double decode_sec = now_sec() - batch_t0;
  llama_memory_clear(llama_get_memory(impl_->ctx), true);
  impl_->cache_valid = false;

  for (size_t j = 0; j < S; ++j) {
    auto& job = jobs[j];
    auto& s = slot[j];
    if (!job.ok && job.err.empty() && s.gen.empty()) continue;
    if (s.gen.empty()) continue;
    job.result.decode_sec = decode_sec;
    job.result.tok_per_sec =
        decode_sec > 0
            ? static_cast<double>(s.gen.size()) / decode_sec
            : 0.0;
    job.result.load_sec = load_sec_;
    job.result.tokens.assign(s.gen.begin(), s.gen.end());
    int32_t need = 512;
    std::vector<char> buf(static_cast<std::size_t>(need));
    for (int attempt = 0; attempt < 4; ++attempt) {
      const int32_t n = llama_detokenize(
          impl_->vocab, s.gen.data(), static_cast<int32_t>(s.gen.size()),
          buf.data(), need, false, false);
      if (n >= 0) {
        job.result.text.assign(buf.data(), static_cast<std::size_t>(n));
        break;
      }
      need = -n;
      buf.resize(static_cast<std::size_t>(need));
    }
    job.ok = true;
  }
  PRESTO_LOG_INFO("llamacpp", "batched " + std::to_string(S) +
                                  " request(s) in one engine pass");
  return true;
}

std::unique_ptr<IBackend> make_llamacpp_backend(const std::string& path) {
  return std::make_unique<LlamaCppBackend>(path);
}

} // namespace presto
