// presto - engine facade + backend registry
#include "presto/engine.hpp"

#ifdef PRESTO_WITH_LLAMACPP
#include "backends/llamacpp_backend.hpp"
#endif
#ifdef PRESTO_WITH_MLX
#include "backends/mlx_backend.hpp"
#endif

#include <memory>

namespace presto {

BackendCaps backend_caps() {
  BackendCaps caps;
#ifdef PRESTO_WITH_LLAMACPP
  caps.llamacpp = true;
#endif
#ifdef PRESTO_WITH_MLX
  caps.mlx = true;
#endif
  return caps;
}

std::unique_ptr<IBackend> select_backend(const Detection& d, std::string& err) {
  switch (d.format) {
    case ModelFormat::GGUF: {
#ifdef PRESTO_WITH_LLAMACPP
      return make_llamacpp_backend(d.path);
#else
      err = "GGUF execution unavailable in this binary; rebuild with -DPRESTO_WITH_LLAMACPP=ON";
      return nullptr;
#endif
    }
    case ModelFormat::MLX_DIR: {
#ifdef PRESTO_WITH_MLX
      return make_mlx_backend(d.path);
#else
      err = "MLX execution requires an Apple Silicon build with -DPRESTO_WITH_MLX=ON "
            "(mlx-lm layout directories)";
      return nullptr;
#endif
    }
    default:
      err = "no native execution path for format '" + std::string(format_name(d.format)) +
            "' yet. Roadmap: convert to GGUF via llama.cpp convert_hf_to_gguf.py, "
            "or to MLX via mlx_lm.convert";
      return nullptr;
  }
}

} // namespace presto
