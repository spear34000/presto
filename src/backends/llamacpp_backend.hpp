// presto - llama.cpp GGUF backend
#pragma once

#include "presto/backend.hpp"

#include <memory>

namespace presto {

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

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string path_;
  double load_sec_ = 0.0;
};

std::unique_ptr<IBackend> make_llamacpp_backend(const std::string& path);

} // namespace presto
