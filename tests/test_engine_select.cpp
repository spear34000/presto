// presto tests - backend routing logic
#include "test_util.hpp"

#include "presto/engine.hpp"

namespace presto::testing {

Detection det(presto::ModelFormat f) {
  Detection d;
  d.format = f;
  d.path = "/virtual/model";
  return d;
}

PRESTO_TEST(select_backend_routing) {
  const BackendCaps caps = backend_caps();

  std::string err;
  if (auto b = select_backend(det(presto::ModelFormat::GGUF), err)) {
    PRESTO_EXPECT(caps.llamacpp);
    PRESTO_EXPECT(std::string(b->name()) == "llamacpp");
  } else {
    PRESTO_EXPECT(!caps.llamacpp);
    PRESTO_EXPECT(err.find("PRESTO_WITH_LLAMACPP") != std::string::npos);
  }

  if (auto b = select_backend(det(presto::ModelFormat::MLX_DIR), err)) {
    PRESTO_EXPECT(caps.mlx);
    PRESTO_EXPECT(std::string(b->name()) == "mlx");
  } else {
    PRESTO_EXPECT(!caps.mlx);
    PRESTO_EXPECT(err.find("MLX") != std::string::npos);
  }

  // inspection-only formats must never select an executor
  for (const auto f : {presto::ModelFormat::SAFETENSORS_FILE,
                       presto::ModelFormat::PYTORCH_FILE,
                       presto::ModelFormat::HF_DIR_AWQ,
                       presto::ModelFormat::HF_DIR_GPTQ}) {
    err.clear();
    PRESTO_EXPECT(select_backend(det(f), err) == nullptr);
    PRESTO_EXPECT(err.find("convert") != std::string::npos);  // actionable hint present
  }
}

} // namespace presto::testing
