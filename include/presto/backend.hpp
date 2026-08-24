// presto - backend interface
#pragma once

#include <string>
#include <vector>

namespace presto {

struct GenerateParams {
  std::vector<int> prompt_tokens; // used when prompt_text empty (mlx path)
  std::string prompt_text;        // preferred for backends with a tokenizer
  int max_tokens = 32;
  float temp = 0.0f;              // <=0 => greedy
  long long seed = -1;
};

struct GenerateResult {
  std::vector<int> tokens;
  std::string text;
  double tok_per_sec = 0.0;
  double load_sec = 0.0;
  double prefill_sec = 0.0;  // prompt evaluation time
  double decode_sec = 0.0;   // token-by-token generation time
};

struct BatchJob {
  GenerateParams params;
  GenerateResult result;
  std::string err;
  bool ok = false;
};

class IBackend {
public:
  virtual ~IBackend() = default;
  virtual const char* name() const = 0;
  virtual bool load(std::string& err) = 0;
  virtual bool generate(const GenerateParams& p, GenerateResult& r, std::string& err) = 0;

  // Execute several requests concurrently. Backends that can interleave
  // sequences in one decode pass override this; the default runs the jobs
  // sequentially through generate(). Returning false means "cannot batch
  // this mix" - callers fall back to per-job generate().
  virtual bool generate_many(std::vector<BatchJob>& jobs, std::string& err) {
    for (auto& j : jobs) j.ok = generate(j.params, j.result, j.err);
    return true;
  }
};

} // namespace presto
