// presto - llama.cpp GGUF backend
#pragma once

#include "presto/backend.hpp"

#include <memory>
#include <string>

namespace presto {

// Optional speculative-decoding companion model (same tokenizer/vocab family).
// Set before LlamaCppBackend::load(); greedy-only acceleration whose outputs
// are bit-identical to the non-speculative path.
void set_spec_draft_path(const std::string& path);

// Persistent llama.cpp engine: model and context are created once at load()
// and reused across generate() calls. The KV memory is cleared between calls
// so every request starts from a clean state without reallocating buffers -
// this is what keeps server-mode latency stable.
class LlamaCppBackend : public IBackend {
public:
  explicit LlamaCppBackend(std::string model_path);
  ~LlamaCppBackend() override;

  const char* name() const override { return "llamacpp"; }
  bool load(std::string& err) override;
  bool generate(const GenerateParams& p, GenerateResult& r, std::string& err) override;
  bool generate_many(std::vector<BatchJob>& jobs, std::string& err) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string path_;
  double load_sec_ = 0.0;
  bool generate_speculative(const GenerateParams& p, GenerateResult& r, std::string& err);
};

std::unique_ptr<IBackend> make_llamacpp_backend(const std::string& path);

} // namespace presto
