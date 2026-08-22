// presto - CLI entry point (hand-rolled argparse, zero external deps)
#include "presto/detector.hpp"
#include "presto/engine.hpp"
#include "presto/format.hpp"
#include "presto/log.hpp"
#include "presto/server.hpp"

#ifdef PRESTO_WITH_LLAMACPP
#include "backends/llamacpp_backend.hpp"
#endif
#ifdef PRESTO_WITH_MLX
#include "backends/mlx_backend.hpp"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace presto {
namespace {

constexpr const char* kVersion = "0.1.0";

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
               "presto %s - unified LLM runtime (gguf | safetensors | pytorch | awq | gptq | mlx)\n\n"
               "USAGE:\n"
               "  presto info <model-path>\n"
               "  presto run <model-path> [--prompt \"...\"] [--prompt-tokens \"1,2,3\"]\n"
               "              [--max-tokens N] [--temp F] [--seed N]\n"
               "  presto serve <model-path> [--host H] [--port P]\n"
               "  presto version\n\n"
               "ENV:\n"
               "  PRESTO_LOG=debug|info|warn|error   log level (stderr)\n"
               "  PRESTO_SMOKE=1                     emit machine-readable success line on run\n",
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
  std::printf("presto v%s\n", kVersion);
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
  const Detection d = detect_format(args[0]);
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
    print_error_line("usage", "unknown option '" + a + "'");
    return kUsage;
  }

  const Detection d = detect_format(model_path);
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
    if (cmd == "info") return cmd_info({argv + 2, argv + argc});
    if (cmd == "run") return cmd_run({argv + 2, argv + argc});
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
      const Detection d = detect_format(args[0]);
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
