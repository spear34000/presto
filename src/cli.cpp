// presto - CLI entry point (hand-rolled argparse, zero external deps)
#include "presto/detector.hpp"
#include "presto/engine.hpp"
#include "presto/format.hpp"
#include "presto/log.hpp"
#include "presto/resolve.hpp"
#include "presto/server.hpp"

#ifdef PRESTO_WITH_LLAMACPP
#include "backends/llamacpp_backend.hpp"
#endif
#ifdef PRESTO_WITH_MLX
#include "backends/mlx_backend.hpp"
#endif

namespace presto {
namespace {
void note_draft_path(const std::string& p) {
#ifdef PRESTO_WITH_LLAMACPP
  set_spec_draft_path(p);
#else
  (void)p;
#endif
}
} // namespace
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace presto {
namespace {

constexpr const char* kVersion = "0.2.0";

enum ExitCode : int {
  kOk = 0,
  kGenericError = 1,
  kUsage = 2,
  kUnsupportedFormat = 3,
  kBackendUnavailable = 4,
  kInferenceFailure = 5,
};

void print_error_line(const std::string& step, const std::string& msg) {
  std::fprintf(stderr, "[presto-error] step=%s msg=\"%s\"\n", step.c_str(), msg.c_str());
}

void print_usage(std::FILE* out) {
  std::fprintf(out,
               "presto %s - unified LLM runtime (gguf | safetensors | pytorch | awq | gptq | mlx)\n"
               "\"presto\": Italian musical term - very fast (172-208 BPM). We take the marking seriously.\n\n"
               "USAGE:\n"
               "  presto info <model-path>\n"
               "  presto run <model-path> [--prompt \"...\"] [--prompt-tokens \"1,2,3\"]\n"
               "              [--max-tokens N] [--temp F] [--seed N]\n"
               "  presto bench <model-path> [--steps N] [--warmup N] [--runs N]\n"
               "              [--temp F]\n"
               "  presto serve <model-path> [--host H] [--port P]\n"
               "  presto version\n\n"
               "ENV:\n"
               "  PRESTO_LOG=debug|info|warn|error   log level (stderr)\n"
               "  PRESTO_SMOKE=1                     emit machine-readable success line on run\n"
               "  PRESTO_THREADS=N                   inference threads (default: hw threads, cap 8)\n"
               "  PRESTO_CTX=N                       context size for gguf models (default 4096)\n",
               kVersion);
}

bool parse_int(const char* s, long long& out) {
  char* e = nullptr;
  out = std::strtoll(s, &e, 10);
  return e && *e == '\0';
}

bool parse_float(const char* s, float& out) {
  char* e = nullptr;
  out = std::strtof(s, &e);
  return e && *e == '\0';
}

std::vector<int> parse_token_list(const std::string& s, bool& ok) {
  ok = true;
  std::vector<int> ids;
  std::size_t i = 0;
  while (i < s.size()) {
    const std::size_t j = s.find(',', i);
    const std::string part =
        s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    try {
      ids.push_back(std::stoi(part));
    } catch (...) {
      ok = false;
      break;
    }
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return ids;
}

int cmd_version() {
  const BackendCaps caps = backend_caps();
  std::printf("presto v%s - \"presto\": very fast (Italian, musical tempo; 172-208 BPM)\n",
              kVersion);
#if defined(_WIN32)
  std::printf("platform: windows\n");
#elif defined(__APPLE__)
  std::printf("platform: darwin (%s)\n", __aarch64__ ? "arm64" : "x86_64");
#else
  std::printf("platform: linux\n");
#endif
  std::printf("backends: llamacpp=%s mlx=%s\n", caps.llamacpp ? "YES" : "no",
              caps.mlx ? "YES" : "no");
  std::printf(
      "hw      : cuda=%s vulkan=%s hip=%s sycl=%s metal=%s cpu=yes\n",
#if defined(PRESTO_HW_CUDA)
      "yes",
#else
      "no",
#endif
#if defined(PRESTO_HW_VULKAN)
      "yes",
#else
      "no",
#endif
#if defined(PRESTO_HW_HIP)
      "yes",
#else
      "no",
#endif
#if defined(PRESTO_HW_SYCL)
      "yes",
#else
      "no",
#endif
#if defined(PRESTO_HW_METAL)
      "yes"
#else
      "no"
#endif
  );
  std::printf(
      "formats : gguf safetensors pytorch awq gptq mlx (info+inspection on all platforms)\n");
  std::printf("execute : %s\n",
              caps.llamacpp && caps.mlx     ? "gguf, mlx-dir"
              : caps.llamacpp               ? "gguf"
              : caps.mlx                    ? "mlx-dir"
                                            : "(none - inspection-only build)");
  return kOk;
}

int cmd_info(const std::vector<std::string>& args) {
  if (args.empty()) {
    print_usage(stderr);
    return kUsage;
  }
  const Detection d = detect_format(resolve_model_path(args[0]));
  std::printf("path   : %s\n", d.path.c_str());
  std::printf("format : %s\n", format_name(d.format));
  std::printf("summary: %s\n", d.summary.c_str());
  for (const auto& [k, v] : d.meta) std::printf("  %-18s %s\n", k.c_str(), v.c_str());
  return d.format == ModelFormat::UNKNOWN ? kUnsupportedFormat : kOk;
}

int cmd_run(const std::vector<std::string>& args) {
  if (args.empty()) {
    print_usage(stderr);
    return kUsage;
  }

  std::string model_path = args[0];
  std::string prompt_text;
  std::vector<int> prompt_tokens;
  int max_tokens = 32;
  float temp = 0.0f;
  long long seed = -1;
  std::string draft_path;

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    auto next_arg = [&](std::string& dst) -> bool {
      if (i + 1 >= args.size()) return false;
      dst = args[++i];
      return true;
    };
    if (a == "--prompt" && next_arg(prompt_text)) continue;
    if (a == "--max-tokens") {
      std::string v;
      long long n = 0;
      if (!next_arg(v) || !parse_int(v.c_str(), n)) {
        print_error_line("usage", "--max-tokens expects an integer");
        return kUsage;
      }
      max_tokens = static_cast<int>(n);
      continue;
    }
    if (a == "--temp") {
      std::string v;
      float f = 0.f;
      if (!next_arg(v) || !parse_float(v.c_str(), f)) {
        print_error_line("usage", "--temp expects a float");
        return kUsage;
      }
      temp = f;
      continue;
    }
    if (a == "--seed") {
      std::string v;
      if (!next_arg(v) || !parse_int(v.c_str(), seed)) {
        print_error_line("usage", "--seed expects an integer");
        return kUsage;
      }
      continue;
    }
    if (a == "--prompt-tokens") {
      std::string v;
      bool ok = false;
      if (!next_arg(v) || (prompt_tokens = parse_token_list(v, ok), !ok)) {
        print_error_line("usage", "--prompt-tokens expects comma-separated integers");
        return kUsage;
      }
      continue;
    }
    if (a == "--draft" && i + 1 < args.size()) {
      draft_path = args[++i];
      continue;
    }
    print_error_line("usage", "unknown option '" + a + "'");
    return kUsage;
  }

  if (!draft_path.empty()) note_draft_path(draft_path);

  const Detection d = detect_format(resolve_model_path(model_path));
  if (d.format == ModelFormat::UNKNOWN) {
    print_error_line("detect", d.summary);
    return kUnsupportedFormat;
  }

  std::string err;
  auto backend = select_backend(d, err);
  if (!backend) {
    print_error_line("select_backend", err);
    return (d.format == ModelFormat::GGUF || d.format == ModelFormat::MLX_DIR)
               ? kBackendUnavailable
               : kUnsupportedFormat;
  }

  GenerateParams p;
  p.prompt_text = prompt_text;
  p.prompt_tokens = prompt_tokens;
  p.max_tokens = max_tokens > 0 ? max_tokens : 32;
  p.temp = temp;
  p.seed = seed;

  if (!backend->load(err)) {
    print_error_line("load", err);
    return kInferenceFailure;
  }
  GenerateResult r;
  if (!backend->generate(p, r, err)) {
    print_error_line("generate", err);
    return kInferenceFailure;
  }

  if (!r.text.empty()) {
    std::printf("%s\n", r.text.c_str());
  }
  std::printf("--- generated %zu token(s)", r.tokens.size());
  if (!r.tokens.empty()) {
    std::printf(": ");
    for (std::size_t i = 0; i < r.tokens.size(); ++i)
      std::printf("%d%s", r.tokens[i], i + 1 < r.tokens.size() ? "," : "");
  }
  std::printf("\n");

  if (const char* smoke = std::getenv("PRESTO_SMOKE"); smoke && std::strcmp(smoke, "1") == 0) {
    std::printf("[presto-smoke] format=%s backend=%s generated_tokens=%zu tps=%.2f ok=true\n",
                format_name(d.format), backend->name(), r.tokens.size(), r.tok_per_sec);
  }
  return kOk;
}

const char* tempo_marking(double tok_per_sec) {
  if (tok_per_sec >= 1000) return "prestissimo";
  if (tok_per_sec >= 200) return "presto";
  if (tok_per_sec >= 80) return "allegro";
  if (tok_per_sec >= 40) return "moderato";
  return "andante";
}

int cmd_bench(const std::vector<std::string>& args) {
  if (args.empty()) {
    print_usage(stderr);
    return kUsage;
  }

  std::string model_path = args[0];
  std::string prompt = "The quick brown fox";
  int steps = 128, warmup = 8, runs = 5;
  float temp = 0.0f;

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    auto next_int = [&](int& dst) -> bool {
      if (i + 1 >= args.size()) return false;
      long long v = 0;
      if (!parse_int(args[++i].c_str(), v)) return false;
      dst = static_cast<int>(v);
      return true;
    };
    auto next_float = [&](float& dst) -> bool {
      if (i + 1 >= args.size()) return false;
      return parse_float(args[++i].c_str(), dst);
    };
    if (a == "--steps" && next_int(steps)) continue;
    if (a == "--warmup" && next_int(warmup)) continue;
    if (a == "--runs" && next_int(runs)) continue;
    if (a == "--temp" && next_float(temp)) continue;
    if (a == "--prompt" && i + 1 < args.size()) {
      prompt = args[++i];
      continue;
    }
    if (a == "--draft" && i + 1 < args.size()) {
      note_draft_path(args[++i]);
      continue;
    }
    print_error_line("usage", "unknown bench option '" + a + "'");
    return kUsage;
  }
  steps = std::max(1, std::min(steps, 4096));
  warmup = std::max(0, std::min(warmup, 64));
  runs = std::max(1, std::min(runs, 64));

  const Detection d = detect_format(resolve_model_path(model_path));
  if (d.format == ModelFormat::UNKNOWN) {
    print_error_line("detect", d.summary);
    return kUnsupportedFormat;
  }
  std::string err;
  auto backend = select_backend(d, err);
  if (!backend) {
    print_error_line("select_backend", err);
    return (d.format == ModelFormat::GGUF || d.format == ModelFormat::MLX_DIR)
               ? kBackendUnavailable
               : kUnsupportedFormat;
  }
  if (!backend->load(err)) {
    print_error_line("load", err);
    return kInferenceFailure;
  }

  GenerateParams p;
  p.prompt_text = prompt;
  p.max_tokens = steps;
  p.temp = temp;
  p.seed = 42;

  for (int i = 0; i < warmup; ++i) {
    GenerateResult w;
    p.max_tokens = std::min(steps, 16);
    if (!backend->generate(p, w, err)) {
      print_error_line("warmup", err);
      return kInferenceFailure;
    }
  }
  p.max_tokens = steps;

  std::vector<double> tps;
  double med_prefill = 0.0;
  tps.reserve(static_cast<std::size_t>(runs));
  for (int run = 0; run < runs; ++run) {
    GenerateResult r;
    if (!backend->generate(p, r, err)) {
      print_error_line("bench_run", err);
      return kInferenceFailure;
    }
    tps.push_back(r.tok_per_sec);
    med_prefill += r.prefill_sec;
  }
  med_prefill /= static_cast<double>(runs);

  std::sort(tps.begin(), tps.end());
  const double median =
      tps.size() % 2 == 1
          ? tps[tps.size() / 2]
          : 0.5 * (tps[tps.size() / 2 - 1] + tps[tps.size() / 2]);
  double var = 0.0;
  for (double v : tps) var += (v - median) * (v - median);
  const double sigma = std::sqrt(var / static_cast<double>(tps.size()));
  const double spread_pct = median > 0 ? 100.0 * sigma / median : 0.0;

  std::printf("=== Tempo Report: %s ===\n", model_path.c_str());
  std::printf("backend : %s   format: %s\n", backend->name(), format_name(d.format));
  std::printf("runs    : %d x %d tok (warmup %d, temp %.2f)\n", runs, steps, warmup, temp);
  std::printf("prefill : median %.4f s\n", med_prefill);
  std::printf("decode  : min/med/max = %.1f / %.1f / %.1f tok/s   sigma=%.2f (+-%.1f%%)\n",
              tps.front(), median, tps.back(), sigma, spread_pct);
  std::printf("tempo   : %.1f tok/s - marking \"%s\"\n", median,
              tempo_marking(median));
  if (const char* smoke = std::getenv("PRESTO_SMOKE"); smoke && std::strcmp(smoke, "1") == 0) {
    std::printf(
        "[presto-bench] backend=%s format=%s steps=%d med_tps=%.2f sigma=%.2f ok=true\n",
        backend->name(), format_name(d.format), steps, median, sigma);
  }
  return kOk;
}

} // namespace
} // namespace presto

#ifdef PRESTO_WITH_LLAMACPP
#endif

int main(int argc, char** argv) {
  using namespace presto;

  if (argc < 2) {
    print_usage(stderr);
    return kUsage;
  }
  const std::string cmd = argv[1];

  try {
    if (cmd == "version") return cmd_version();
    if (cmd == "models") {
      for (const auto& p : list_known_models()) std::printf("%s\n", p.c_str());
      return kOk;
    }
    if (cmd == "info") return cmd_info({argv + 2, argv + argc});
    if (cmd == "run") return cmd_run({argv + 2, argv + argc});
    if (cmd == "bench") return cmd_bench({argv + 2, argv + argc});
    if (cmd == "serve") {
      std::vector<std::string> args{argv + 2, argv + argc};
      if (args.empty()) {
        print_usage(stderr);
        return kUsage;
      }
      std::string host = "127.0.0.1";
      int port = 8000;
      for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--host" && i + 1 < args.size()) host = args[++i];
        else if (args[i] == "--port" && i + 1 < args.size()) {
          long long p = 0;
          if (!parse_int(args[++i].c_str(), p) || p <= 0 || p > 65535) {
            print_error_line("usage", "--port expects 1..65535");
            return kUsage;
          }
          port = static_cast<int>(p);
        } else {
          print_error_line("usage", "unknown serve option '" + args[i] + "'");
          return kUsage;
        }
      }
      const Detection d = detect_format(resolve_model_path(args[0]));
      if (d.format == ModelFormat::UNKNOWN) {
        print_error_line("detect", d.summary);
        return kUnsupportedFormat;
      }
      return run_openai_server(d, host, port);
    }
    print_usage(stderr);
    return kUsage;
  } catch (const std::exception& e) {
    print_error_line("fatal", e.what());
    return kGenericError;
  }
}
