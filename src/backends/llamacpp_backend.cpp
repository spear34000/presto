// presto - llama.cpp GGUF backend implementation
#include "backends/llamacpp_backend.hpp"

#include "presto/log.hpp"

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace presto {

struct LlamaCppBackend::Impl {
  llama_model* model = nullptr;
  const llama_vocab* vocab = nullptr;
};

LlamaCppBackend::LlamaCppBackend(std::string path) : path_(std::move(path)) {}

LlamaCppBackend::~LlamaCppBackend() {
  if (impl_) {
    if (impl_->model) llama_model_free(impl_->model);
    llama_backend_free();
  }
}

bool LlamaCppBackend::load(std::string& err) {
  const auto t0 = std::chrono::steady_clock::now();
  llama_backend_init();
  impl_ = std::make_unique<Impl>();

  // Newer llama.cpp refuses to load a model until at least one ggml backend
  // is registered; this also picks up the statically linked CPU backend.
  if (ggml_backend_reg_count() == 0) {
    ggml_backend_load_all();
  }

  llama_model_params mparams = llama_model_default_params();
  impl_->model = llama_model_load_from_file(path_.c_str(), mparams);
  if (!impl_->model) {
    err = "step=model_load msg=\"llama_model_load_from_file failed for " + path_ + "\"";
    return false;
  }
  impl_->vocab = llama_model_get_vocab(impl_->model);
  if (!impl_->vocab) {
    err = "step=get_vocab msg=\"llama_model_get_vocab returned null\"";
    return false;
  }
  load_sec_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  PRESTO_LOG_INFO("llamacpp", "model loaded in " + std::to_string(load_sec_) + "s");
  return true;
}

namespace {

int32_t tokenize_prompt(const llama_vocab* vocab, const std::string& text,
                        std::vector<llama_token>& out, std::string& err) {
  // llama.h contract: a null buffer yields the negative required token count;
  // only INT32_MIN signals hard failure.
  int32_t need = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                nullptr, 0, true, true);
  if (need == INT32_MIN || need == -1) {
    err = "step=tokenize msg=\"sizing pass hard-failed (" + std::to_string(need) + ")\"";
    return -1;
  }
  if (need < 0) need = -need;

  for (int attempt = 0; attempt < 2; ++attempt) {
    out.assign(static_cast<std::size_t>(need), 0);
    const int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                     out.data(), need, true, true);
    if (n >= 0) {
      return n;
    }
    if (n == INT32_MIN) break;
    need = -n;
  }
  err = "step=tokenize msg=\"tokenization failed after sizing\"";
  return -1;
}

llama_token sample_token(const float* logits, int n_vocab, float temp, long long seed,
                         std::mt19937& rng) {
  if (temp <= 0.0f || n_vocab <= 0) {
    int best = 0;
    for (int i = 1; i < n_vocab; ++i)
      if (logits[i] > logits[best]) best = i;
    return static_cast<llama_token>(best);
  }
  // temperature softmax over a stable copy
  std::vector<double> probs(static_cast<std::size_t>(n_vocab));
  double maxl = -1e30;
  for (int i = 0; i < n_vocab; ++i) maxl = std::max(maxl, static_cast<double>(logits[i]));
  double sum = 0.0;
  for (int i = 0; i < n_vocab; ++i) {
    probs[static_cast<std::size_t>(i)] = std::exp((logits[i] - static_cast<float>(maxl)) / temp);
    sum += probs[static_cast<std::size_t>(i)];
  }
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return static_cast<llama_token>(dist(rng));
}

} // namespace

bool LlamaCppBackend::generate(const GenerateParams& gp, GenerateResult& r, std::string& err) {
  if (!impl_ || !impl_->model || !impl_->vocab) {
    err = "step=state msg=\"backend not loaded\"";
    return false;
  }

  std::vector<llama_token> prompt;
  if (!gp.prompt_text.empty()) {
    const int n = tokenize_prompt(impl_->vocab, gp.prompt_text, prompt, err);
    if (n < 0) return false;
    prompt.resize(static_cast<std::size_t>(n));
  } else {
    prompt.reserve(gp.prompt_tokens.size());
    for (int t : gp.prompt_tokens) prompt.push_back(static_cast<llama_token>(t));
  }
  if (prompt.empty()) {
    err = "step=prompt msg=\"empty prompt\"";
    return false;
  }

  const auto want_ctx = static_cast<std::uint32_t>(prompt.size() + gp.max_tokens + 8);
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = std::min<std::uint32_t>(8192, std::max<std::uint32_t>(512, want_ctx));
  cparams.n_batch = cparams.n_ctx;

  llama_context* ctx = llama_init_from_model(impl_->model, cparams);
  if (!ctx) {
    err = "step=create_context msg=\"llama_init_from_model failed\"";
    return false;
  }
  struct CtxGuard {
    llama_context* c;
    ~CtxGuard() { if (c) llama_free(c); }
  } ctx_guard{ctx};

  const int n_vocab = llama_n_vocab(impl_->vocab);
  if (n_vocab <= 0) {
    err = "step=vocab_size msg=\"non-positive vocab size\"";
    return false;
  }

  std::mt19937 rng(gp.seed >= 0 ? static_cast<unsigned>(gp.seed) : std::random_device{}());

  // evaluate the prompt (chunked by n_batch)
  const auto gen_t0 = std::chrono::steady_clock::now();
  int32_t processed = 0;
  while (processed < static_cast<int32_t>(prompt.size())) {
    const int32_t chunk =
        std::min<int32_t>(static_cast<int32_t>(cparams.n_batch),
                          static_cast<int32_t>(prompt.size()) - processed);
    llama_batch b = llama_batch_get_one(prompt.data() + processed, chunk);
    if (llama_decode(ctx, b)) {
      err = "step=prompt_decode msg=\"llama_decode failed at offset " +
            std::to_string(processed) + "\"";
      return false;
    }
    processed += chunk;
  }

  std::vector<llama_token> out_tokens;
  out_tokens.reserve(static_cast<std::size_t>(std::max(0, gp.max_tokens)));

  for (int step = 0; step < gp.max_tokens; ++step) {
    const float* logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
      err = "step=get_logits msg=\"null logits at step " + std::to_string(step) + "\"";
      return false;
    }
    const llama_token next = sample_token(logits, n_vocab, gp.temp, gp.seed, rng);
    if (llama_vocab_is_eog(impl_->vocab, next)) break;
    out_tokens.push_back(next);

    if (step + 1 < gp.max_tokens) {
      if (llama_decode(ctx, llama_batch_get_one(&out_tokens.back(), 1))) {
        err = "step=decode msg=\"llama_decode failed at step " + std::to_string(step) + "\"";
        return false;
      }
    }
  }

  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - gen_t0).count();

  // detokenize output tokens to text
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

  r.tokens.assign(out_tokens.begin(), out_tokens.end());
  r.tok_per_sec = secs > 0 ? static_cast<double>(out_tokens.size()) / secs : 0.0;
  r.load_sec = load_sec_;
  return true;
}

std::unique_ptr<IBackend> make_llamacpp_backend(const std::string& path) {
  return std::make_unique<LlamaCppBackend>(path);
}

} // namespace presto
