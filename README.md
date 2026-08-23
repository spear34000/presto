# presto 🎶

> **presto** /ˈprɛs.to/ - Italian musical term: *very fast* (172-208 BPM).
> A unified LLM inference runtime that takes the marking seriously.

One static C++ binary that inspects every major model format, executes GGUF and
MLX natively, and runs on whatever silicon you have: NVIDIA, AMD Radeon,
Intel iGPU/Arc via Vulkan, Apple Silicon via Metal and MLX, or plain x86 CPU.

## Why presto (tempo = tokens)

Built by surveying the most popular open-source runtimes (stars verified 2026-08)
and absorbing the best idea of each:

| # | Runtime | Stars | Idea presto absorbs |
|---|---------|-------|---------------------|
| 1 | [ollama](https://github.com/ollama/ollama) | ~179k | one-command UX + OpenAI-compatible server |
| 2 | [llama.cpp](https://github.com/ggml-org/llama.cpp) | ~125k | native GGUF execution, broadest hardware |
| 3 | [vLLM](https://github.com/vllm-project/vllm) | ~90k | OpenAI `/v1/*` API surface |
| 4 | [SGLang](https://github.com/sgl-project/sglang) | ~32k | (roadmap: prefix caching) |
| 5 | [MLX](https://github.com/ml-explore/mlx) / mlx-lm | ~28k | Apple-Silicon-native weight format |
| 7 | [ktransformers](https://github.com/kvcache-ai/ktransformers) | ~19k | heterogeneous backend routing |

## Tempo Report (measured, not promised)

Every build ships `presto bench`, which measures decode throughput across
repeated runs and reports min/median/max and sigma. Stability is a feature:
the sigma number IS the stability claim.

### Head-to-head on one machine

Windows 11, Intel/AMD CPU 4 threads, stories15M-q4_0.gguf, 128-token decode,
temperature 0:

| Runtime | Median tok/s | Sigma | vs presto |
|---|---|---|---|
| **presto** (`presto bench`)            | **1724** | ±2.8% | - |
| llama.cpp stock (`llama-bench` tg128)  | 1720     | ±3.6% | 1.00x |
| ollama 0.32.15 (`/api/generate`)       | 1016     | n/a   | **1.70x faster** |

Same kernels as llama.cpp (we link it) with zero serving overhead; ollama's
daemon+HTTP stack costs ~40% of throughput at this scale.

### Prefix KV cache (SGLang RadixAttention idea)

Consecutive requests sharing token history skip re-prefill entirely -
multi-turn chat, agent loops and fixed system prompts get near-zero
time-to-first-token on the shared part. Measured in-process on
stories15M-q4_0 (serve mode):

```
prefix reuse: kept 385/398 tokens    <- 97% of prompt prefill skipped
```

Greedy outputs are byte-identical with `PRESTO_PREFIX_CACHE=0`, so the
speedup is free of behavioral change.

### GPU offload

Windows 11, Intel Arc 140V iGPU, stories15M-q4_0.gguf, 256 tokens x 5 runs:

| Device | Median tok/s | Marking | Sigma |
|---|---|---|---|
| Intel Arc 140V (Vulkan offload) | **707.1** | presto | ±12.3% |
| Same machine, CPU only          | 247.5   | presto | ±10.1% |

GPU offload is automatic: compile with `-DPRESTO_WITH_VULKAN=ON`, run the
same command, and the model lands on whichever device is available.

### Multi-model validation

All models verified on one machine: format detection, greedy reproducibility
(two identical invocations -> identical token ids) and decode tempo.

| Model | Arch | Quant | Detect | Reproducible | tok/s (CPU x4) |
|---|---|---|---|---|---|
| stories260K.gguf            | llama  | F32  | ✅ | ✅ | 2498 (prestissimo) |
| stories15M-q4_0.gguf        | llama  | Q4_0 | ✅ | ✅ | 1415 (prestissimo) |
| SmolLM2-135M-Instruct-Q8_0  | llama  | Q8_0 | ✅ | ✅ | 191 (allegro) |
| qwen2.5-0.5b-instruct-Q8_0  | qwen2  | Q8_0 | ✅ | ✅ | 81 (allegro) |
| SmolLM-135M-fp16 (MLX dir)  | llama  | F16  | ✅ | ✅ | macOS CI run |

The `qwen2` row exercises a different architecture, GQA head ratios and tied
embeddings through the same unified engine.

### Tuning knobs (all opt-in, measured)

| Env | Default | Effect |
|---|---|---|
| `PRESTO_PREFIX_CACHE` | on | reuse KV across shared-prefix requests (97% prefill skip measured); outputs bit-identical when off |
| `PRESTO_FLASH_ATTN=1` | off | force Flash Attention kernels (A/B: slower on some small models) |
| `PRESTO_POLL=100` | off | spinning threadpools: lowest latency for dedicated servers; pins cores while idle |
| `PRESTO_KV=q8_0` | fp16 | KV quantization for memory-bound decode |
| `PRESTO_THREADS` / `PRESTO_CTX` / `PRESTO_GPU_LAYERS` | auto | thread count / context size / layer offload |

CI enforces tempo floors so a regression fails the build:
windows `med_tps >= 100`, macOS Metal `med_tps >= 20` (calibrated against
the measured 29.5 tok/s of stories260K on an M1 runner).

## Hardware support

| Hardware | Path | Build flag | Status |
|---|---|---|---|
| NVIDIA (discrete)      | CUDA   | `-DPRESTO_WITH_CUDA=ON`   | CI-verified builds; runtime needs NVIDIA GPU |
| NVIDIA / AMD / Intel   | Vulkan | `-DPRESTO_WITH_VULKAN=ON` | **runtime-verified on Intel Arc 140V** |
| AMD Radeon (Linux)     | HIP    | `-DPRESTO_WITH_HIP=ON`    | builds; runtime needs ROCm device |
| Apple Silicon          | Metal + MLX | auto ON on macOS     | CI runtime-verified every push |
| x86 CPU (Intel/AMD)    | ggml CPU (AVX dispatch) | default  | always available |

Control layer offload with `PRESTO_GPU_LAYERS` (`-1` = auto/max, `0` = CPU-only,
N = exact layers).

## Format support

| Format | Inspect (`presto info`) | Execute |
|---|---|---|
| GGUF (`.gguf`)            | full metadata parse (pure C++)      | YES - llama.cpp backend |
| MLX dir (mlx-lm layout)   | config + quantization detection     | YES - mlx core backend (Apple Silicon), token-id level |
| SafeTensors (`.safetensors`) | tensor inventory + dtype histogram | roadmap (convert to GGUF/MLX) |
| PyTorch (`.pt`/`.pth`/`.ckpt`) | zip container detection        | roadmap (conversion pipeline) |
| AWQ dirs                  | bits / group_size extraction        | roadmap |
| GPTQ dirs                 | quant_method / bits extraction      | roadmap |

## Build

```bash
# baseline: CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# your GPU: add one flag
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPRESTO_WITH_VULKAN=ON   # NVIDIA/AMD/Intel
# cmake -B build ... -DPRESTO_WITH_CUDA=ON                          # NVIDIA
# cmake -B build ... -DPRESTO_WITH_HIP=ON                           # AMD ROCm
cmake --build build --config Release --parallel

# core-only (no network-heavy deps)
cmake -B build-core -DPRESTO_WITH_LLAMACPP=OFF && cmake --build build-core --config Release --parallel
```

Dependencies are FetchContent'd and pinned: llama.cpp `b21e4de74...`,
cpp-httplib v0.15.3, MLX v0.32.1. Requires CMake >= 3.24, C++20.

## Usage

```bash
./build/presto version                  # capability report incl. hw backends
./build/presto info model.gguf          # deep inspection of ANY supported format

./build/presto run model.gguf --prompt "Once upon a time" --max-tokens 64 --temp 0.8

./build/presto bench model.gguf --steps 128 --runs 5     # Tempo Report
./build/presto serve model.gguf --port 8000              # OpenAI-compatible API
curl http://127.0.0.1:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Env: `PRESTO_THREADS` (inference threads) · `PRESTO_CTX` (context size) ·
`PRESTO_GPU_LAYERS` (offload control) · `PRESTO_LOG` · `PRESTO_SMOKE=1`.

Exit codes: `0` ok · `2` usage · `3` unsupported format · `4` backend unavailable · `5` failure.

## Tests

Zero-dependency unit tests (18 cases): JSON parser, GGUF header parser,
safetensors parser, format detection, backend routing.

```bash
cmake --build build --config Release && ./build/presto_tests   # ALL TESTS PASSED
```

## CI

| Job | Runner | Verifies |
|-----|--------|----------|
| `ubuntu-core` | ubuntu-latest | core-only build + unit tests (fast gate) |
| `linux-hw-build` | ubuntu-latest | CUDA + Vulkan toolchains compile & link |
| `windows-gguf` | windows-latest | MSVC + llama.cpp + real generation + tempo gate |
| `macos-mlx` | macos-15 (arm64) | real MLX weights generate tokens + Metal GGUF + tempo gate |

Failures always upload `logs-*` artifacts (configure/build/test/smoke/bench logs +
environment info); smoke and bench emit `[presto-smoke]` / `[presto-bench]` markers,
so one log line identifies the failing stage.

## License

MIT
