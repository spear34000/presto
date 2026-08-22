# presto

Unified LLM inference runtime in C++ - one binary that inspects **all** major model
formats and executes GGUF and MLX models natively.

Built by surveying the most popular open-source runtimes (stars verified 2026-08) and
absorbing the best idea of each:

| # | Runtime | Stars | Idea presto absorbs |
|---|---------|-------|---------------------|
| 1 | [ollama](https://github.com/ollama/ollama) | ~179k | one-command UX + OpenAI-compatible server |
| 2 | [llama.cpp](https://github.com/ggml-org/llama.cpp) | ~125k | native GGUF execution on every CPU/GPU |
| 3 | [vLLM](https://github.com/vllm-project/vllm) | ~90k | OpenAI `/v1/*` API surface |
| 4 | [SGLang](https://github.com/sgl-project/sglang) | ~32k | (roadmap: prefix caching) |
| 5 | [MLX](https://github.com/ml-explore/mlx) / mlx-lm | ~28k | Apple-Silicon-native weight format |
| 7 | [ktransformers](https://github.com/kvcache-ai/ktransformers) | ~19k | heterogeneous backend routing |

## Format support

| Format | Inspect (`presto info`) | Execute |
|---|---|---|
| GGUF (`.gguf`)            | full metadata parse (pure C++)      | YES - llama.cpp backend (Windows/macOS/Linux) |
| MLX dir (mlx-lm layout)   | config + quantization detection     | YES - mlx core backend (Apple Silicon), token-id level |
| SafeTensors (`.safetensors`) | tensor inventory + dtype histogram | roadmap (convert to GGUF/MLX) |
| PyTorch (`.pt`/`.pth`/`.ckpt`) | zip container detection        | roadmap (conversion pipeline) |
| AWQ dirs                  | bits / group_size extraction        | roadmap |
| GPTQ dirs                 | quant_method / bits extraction      | roadmap |

Inspection works everywhere with zero dependencies; execution backends are compiled in
when their platform is available.

## Build

```bash
# full build (llama.cpp backend; MLX auto-enables on Apple Silicon)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# core-only build (no network-heavy deps, fastest)
cmake -B build-core -DPRESTO_WITH_LLAMACPP=OFF
cmake --build build-core --config Release --parallel
```

Dependencies are fetched via CMake FetchContent and pinned:
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `b21e4de74...` (verified commit)
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) v0.15.3
- [MLX](https://github.com/ml-explore/mlx) main (macOS arm64 only)

Requires CMake >= 3.24 and a C++20 compiler (MSVC 2022, clang, gcc).

## Usage

```bash
./build/presto version                 # capability report
./build/presto info model.gguf         # deep inspection of ANY supported format
./build/presto info ./mlx-model-dir/

./build/presto run model.gguf \
    --prompt "Once upon a time" --max-tokens 64 --temp 0.8 --seed 42

./build/presto run ./tiny_mlx --prompt-tokens "1,2,3" --max-tokens 8   # token-id level (MLX)

./build/presto serve model.gguf --port 8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Exit codes: `0` ok · `1` generic · `2` usage · `3` unsupported format ·
`4` backend unavailable · `5` inference failure.
Set `PRESTO_SMOKE=1` to emit `[presto-smoke] ... ok=true` for scripting.
Set `PRESTO_LOG=debug` for verbose logging.

## Tests

Zero-dependency unit tests cover the JSON parser, GGUF header parser,
safetensors parser, format detection and backend routing with synthetic
fixtures:

```bash
cmake --build build --config Release
./build/presto_tests          # prints ALL TESTS PASSED
```

## CI

GitHub Actions runs three jobs on every push:

| Job | Runner | Verifies |
|-----|--------|----------|
| `ubuntu-core` | ubuntu-latest | core-only build + unit tests (fast gate) |
| `windows-gguf` | windows-latest | MSVC + llama.cpp static link + **real GGUF token generation** |
| `macos-mlx` | macos-15 (arm64) | clang + MLX v0.32.1 link + **real MLX token generation** + Metal GGUF |

Failures always upload a `logs-*` artifact containing configure/build/test/smoke logs
and environment info; smoke scripts print `[presto-smoke]` markers so a single log
line tells you exactly which stage produced tokens and at what throughput.

## License

MIT
