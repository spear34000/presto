// presto - MLX backend (Apple Silicon only)
// Minimal token-id-level Llama-family forward pass on mlx core ops.
#pragma once

#include "presto/backend.hpp"

#include <memory>

namespace presto {

class MlxBackend : public IBackend {
public:
  explicit MlxBackend(std::string model_dir);
  ~MlxBackend() override;

  const char* name() const override { return "mlx"; }
  bool load(std::string& err) override;
  bool generate(const GenerateParams& p, GenerateResult& r, std::string& err) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string dir_;
  double load_sec_ = 0.0;
};

std::unique_ptr<IBackend> make_mlx_backend(const std::string& path);

} // namespace presto
