# presto architecture

presto (v0.3.1) is a unified LLM inference runtime in C++20 that ships as one
static binary. It inspects every major open-source model format, executes
GGUF and MLX natively, speaks the OpenAI API, and adapts itself to whatever
silicon it finds: NVIDIA, AMD Radeon, Intel Arc or iGPU, Apple Silicon, or
plain x86. This document explains how the system is put together, why each
piece exists, and how the three techniques that break the decode
bandwidth wall fit into the design.

## Design goals

The project started as a survey of the most popular open-source runtimes
(star counts verified 2026-08): ollama (~179k), llama.cpp (~125k), vLLM
(~90k), SGLang (~32k), MLX (~28k), and ktransformers (~19k). Rather than
compete with any of them head-on, presto absorbed the best idea of each:

| Runtime | Idea absorbed | Where it lives now |
|---|---|---|
| ollama | pull-and-run UX, name-based model resolution | `src/resolve.cpp`, `presto models` |
| llama.cpp | native GGUF execution, broadest hardware reach | `src/backends/llamacpp_backend.cpp` |
| vLLM | `/v1/*` API surface | `src/server_http.cpp` |
| SGLang | prefix KV-cache reuse (RadixAttention) | prefix cache in the llama.cpp backend |
| MLX | Apple-Silicon-native weight format | `src/backends/mlx_backend.cpp` |
| ktransformers | heterogeneous backend routing | `src/engine.cpp` selector |

Three principles fall out of that lineage. First, zero configuration: the
binary probes the machine at load and tunes itself, and every automatic
choice stays overridable through environment variables. Second, correctness
is never traded for speed: every accelerated path (speculation, batching,
prefix reuse) produces output bit-identical to the plain path or falls back
honestly. Third, no heavy dependencies in the core: parsers are pure C++,
the server rides on header-only cpp-httplib, and a core-only build needs
nothing beyond a C++20 compiler.

## System overview

```
                        +-------------------------------------------------+
                        |                   presto CLI                    |
                        |   run | chat | bench | serve | info | models    |
                        +-----------+--------------------------+----------+
                                    |                          |
                     model ref/path |                          | serve mode
                                    v                          |
                         +---------------------+               |
                         |      Detector       |  gguf_meta / st_meta
                         |  (format sniffing)  |--+            |
                         +----------+----------+  |            |
                                    |             |            |
                       Detection    |             |  full metadata parse:
                       {format,     |             |  GGUF kv map, safetensors
                        path, meta} |             |  dtype histogram
                                    v             |
                         +---------------------+ |
                         |   Backend Selector  | |  info / inspect path
                         |    (engine.cpp)     |<+
                         +----+-----------+----+
                              |           |
              GGUF (.gguf)    |           |   MLX dir (Apple Silicon)
                              v           v
        +------------------------------------------------------+
        |                 IBackend interface                   |
        |   load()   generate()   generate_many()              |
        +--------+--------------------------------+------------+
                  |                               |               |
                  v                               v               v
    +-----------------------------+   +-------------------------+  +------------------+
    |     LlamaCpp backend        |   |      MLX backend        |  |   Own backend    |
    |  - adaptive ctx/threads     |   |  mlx core ops,          |  |  ggml-free,      |
    |  - prefix KV cache          |   |  Llama-family forward   |  |  --engine own   |
    |  - speculative decoding     |   |  pass at token-id level |  |  full-recompute|
    |    (draft model + n-gram    |   +-------------------------+  |  (no KV cache) |
    |     pool)                   |                                +------------------+
    |  - continuous batching      |
    |  - ggml devices: CPU, CUDA, |
    |    Vulkan, HIP, SYCL, Metal |
    +--------------+--------------+
                  ^  generate_many(jobs)
                  |
   +--------------+--------------+         +--------------------------+
   |     Server (serve mode)     |<--------|     HTTP clients         |
   |  cpp-httplib endpoints:     |  /v1/*  |  OpenAI SDKs, curl, apps |
   |    GET  /health             |         +--------------------------+
   |    GET  /v1/models          |
   |    POST /v1/completions     |
   |    POST /v1/chat/completions|
   |  ---------------------------|
   |  request queue (mutex+cv)   |
   |  [50 ms coalescing window]  |  gathers arrivals into one batch
   |  batching worker thread     |  exclusively owns the engine
   +-----------------------------+
```

Everything above the `IBackend` line is dependency-free and compiles in
every configuration. The two backend boxes are conditionally compiled:
llama.cpp when `PRESTO_WITH_LLAMACPP=ON` (the default), MLX only on Apple
Silicon.

The llama.cpp execution box is now a migration adapter rather than the
long-term core. `presto_runtime_core` owns provider-neutral status, tensor,
buffer, device, and registry contracts under `include/presto/runtime/`.
Those headers expose no llama.cpp, ggml, BLAS-provider, or GPU-SDK types and
build when every inference backend is disabled. Model ingestion, operators,
sessions, and GPU providers will move behind this boundary family by family;
the adapter is removed after parity gates pass for the supported matrix.

## Format detection

`detect_format()` in `src/detector.cpp` classifies any filesystem path and
never throws. Malformed input yields `UNKNOWN` plus an explanatory summary,
which the CLI turns into exit code 3. The detectors underneath are honest
parsers rather than extension sniffers:

- `src/gguf_meta.cpp` reads the GGUF magic, version, tensor inventory, and
  the full key/value metadata map in pure C++, summarizing arrays instead of
  loading them.
- `src/st_meta.cpp` reads only the JSON header of a safetensors archive and
  builds a tensor inventory with a dtype histogram.
- PyTorch `.pt/.pth/.ckpt` files are recognized by their zip container;
  AWQ and GPTQ directories expose bits/group_size and quant_method metadata.

That split gives `presto info` its promise: deep inspection on all six
format families on every platform, while execution stays limited to formats
with a compiled backend.

## Backend abstraction and routing

`include/presto/backend.hpp` defines the whole contract between the frontend
and inference engines: `GenerateParams` (prompt text or raw token ids,
max_tokens, temperature, seed), `GenerateResult` (tokens, text, tok/s, and
separate load/prefill/decode timings), `BatchJob`, and `IBackend`. The
interface has exactly three virtuals that matter:

- `load()` prepares weights and context.
- `generate()` runs one request.
- `generate_many()` runs several concurrently. The default implementation
  loops over `generate()`; backends capable of interleaving sequences in one
  decode pass override it. Returning false means "cannot batch this mix",
  and callers fall back to per-job generation.

`select_backend()` in `src/engine.cpp` routes on the detected format: GGUF
goes to the llama.cpp backend, MLX directories go to the MLX backend, and
anything else returns nullptr with an actionable error message that names
the conversion tool (`convert_hf_to_gguf.py` or `mlx_lm.convert`). A
compile-time capability report (`backend_caps()`) feeds `presto version`,
so users can see exactly what a given binary can execute.

## The llama.cpp execution engine

`src/backends/llamacpp_backend.cpp` carries most of the system's runtime
intelligence. Its `Impl` struct holds the llama model, vocab, context, the
prefix-cache token mirror, an optional draft-model triple (model, vocab,
context plus its own token mirror), dual spinning threadpools, and the
persistent n-gram pool bounded at 200,000 tokens.

### Load-time adaptation

`load()` probes the machine before creating a single tensor buffer:

1. Device enumeration. Every registered ggml compute device is logged, so
   the record shows exactly where the model will run (CPU, CUDA, Vulkan,
   HIP, Metal).
2. Context size from free RAM. Available memory is read per OS
   (`GlobalMemoryStatusEx` on Windows, `/proc/meminfo` elsewhere) and mapped
   to tiers: 4096 tokens at 12 GB free or more, 2048 at 6 GB or more,
   otherwise 1024. `PRESTO_CTX` overrides.
3. Size-aware thread tuning. Files under 1 GB get `min(hw_threads, 4)`
   threads; larger models get `min(hw_threads, 8)`. The reasoning is
   measured, not guessed: a cache-resident model like SmolLM2 is
   latency-bound, and extra threads only add synchronization overhead (228
   tok/s at 4 threads versus 136 at 8). Large models stream weights from
   memory and do scale with cores. `PRESTO_THREADS` overrides both.
4. GPU layer offload. `PRESTO_GPU_LAYERS` defaults to -1, meaning llama.cpp
   offloads whatever fits on available devices; 0 forces CPU-only.
5. Optional kernel levers, all off by default because they measured slower
   on some workloads: flash attention via `PRESTO_FLASH_ATTN=1` (with an
   automatic retry at AUTO if context creation rejects it), KV-cache
   quantization via `PRESTO_KV=q8_0|q4_0` (V quantization requires flash
   attention), and spinning threadpools via `PRESTO_POLL`.
6. Draft model. If `--draft` was supplied, the companion loads with a vocab
   size check against the target; any failure downgrades to lookup
   speculation with a warning instead of aborting.
7. Warmup decode. One BOS token decodes once so weights and compute buffers
   page in before the first real request, and the BOS stays resident to seed
   the prefix cache.

### Request lifecycle

Each `generate()` call runs a pipeline that reuses as much prior work as
possible:

- Tokenization uses a sizing pass with a retry that disables special tokens
  for BPE vocabs whose BOS handling returns errors.
- Context growth compares the request with `n_ctx / n_seq_max`, the logical
  per-sequence capacity of llama.cpp's shared KV pool. It doubles the total
  pool until that logical capacity fits, preserves the configured slot count,
  then invalidates the prefix cache since fresh memory holds nothing reusable.
- The persistent prompt snapshot ("compiled prompt" cache) saves post-prefill
  KV state to disk for prompts of 64+ tokens (`PRESTO_SNAPSHOT_MIN`),
  keyed by an FNV-1a hash of the token list plus the model file size, under
  `LOCALAPPDATA\presto\kv` or `~/.presto/kv`. A future run restores the
  snapshot after verifying its embedded token list; corrupt files are
  dropped. `PRESTO_KV_SNAPSHOT=0` disables the feature.
- Prefix-cache-aware prefill computes the longest common token prefix with
  the previous request, removes everything after it, and feeds only the
  remainder. An identical prompt drops back one position to force at least
  one evaluation. Measured multi-turn traffic skips 97% of prompt tokens
  this way, with outputs bit-identical to the uncached path.
- After generation, the full prompt-plus-output sequence publishes as the
  new reusable history, and qualifying exchanges append to the n-gram pool.

### Speculative decoding

Greedy requests with at least four tokens to produce automatically enter
speculative decoding. Two proposers exist, and both preserve a hard
guarantee: accepted output is bit-identical to plain greedy decoding,
because every proposal is verified against target-model logits before being
committed.

With `--draft small.gguf`, the small model proposes K tokens per round
(`PRESTO_SPEC_K`, default 6). Both models prefill the same prompt; the
target emits its first greedy token from prefill logits; each round drafts
K proposals with the draft model, verifies them in one target batch decode,
then walks the acceptance chain using a carried prediction so validation
shifts one row deeper per acceptance. The final row yields the correction
token that follows the longest accepted chain. The draft's KV mirror rewinds
rejected proposals and absorbs the correction so it stays token-aligned for
the next round.

Without a draft model, prompt lookup takes over when the prompt has at least
`PRESTO_LOOKUP_MIN` tokens (default 32). The last `PRESTO_LOOKUP_N` tokens
(default 8) form a seed; the generator scans backward through the current
sequence for an earlier occurrence and copies up to `PRESTO_LOOKUP_K`
(default 16) following tokens as proposals. This is where copy-heavy
workloads win big: summarize, RAG, and code-edit requests echo long spans
of their own input, measured at +27-51% tok/s on Qwen2.5-0.5B.

When neither source finds a match, the loop takes one plain greedy step and
retries the next round, so speculation never stalls progress. Architectures
whose KV cache cannot roll back proposals (Qwen3.5-style hybrid attention)
trigger a mid-request fallback: the accepted history is fully re-prefilled
once, then the request finishes on the plain path instead of failing.

### Persistent n-gram pool

Prompt lookup alone is bounded by what the current prompt contains. The
persistent pool removes that bound. Every completed exchange with a prompt
of 32 or more tokens appends prompt and output tokens to
`Impl::ngram_pool`, trimmed from the front at the 200,000-token cap. Lookup
scans this pool as a second source when the local sequence has no match, so
a repeated request drafts from the first request's own output without a
single forward pass through the model. On R1-Distill-Qwen-32B with a
repetitive reasoning prompt, the second identical request fell from 23.2 s
to 11.9 s, a 1.95x speedup, turning the 11.9 tok/s bandwidth wall into an
effective 11.7 tok/s per user on the repeat hit.

### Continuous batching

`generate_many()` implements batched decoding across sequences. It refuses
sampled jobs (any temperature above zero fails the whole batch, and the
server then falls back to sequential generation with identical semantics).
Up to `PRESTO_BATCH_SLOTS` slots (default 4) tokenize, prefill each under
its own sequence id, sample the first token straight from prefill logits,
and then enter a joint loop: one `llama_decode` carries one token position
for every live slot, and each slot samples greedily from its logits row
until EOG or its token budget. The shared KV pool is cleared afterward and
the prefix cache invalidated honestly. Because one weight-streaming pass now
produces one token for each waiting user, aggregate throughput measured
1.6x sequential serving on CPU and 1.8x on the Arc iGPU with 4 users, and
up to 35-36 tok/s aggregate on the iGPU for Gemma-4-26B-A4B and GPT-OSS-20B
at batch 8.

## Own inference engine (ggml-free)

`src/own/` is the first milestone toward complete independence from
llama.cpp/ggml. It has zero external dependencies and builds even with
`PRESTO_WITH_LLAMACPP=OFF`:

- `src/own/gguf_load.cpp` - GGUF v2/v3 header, key/value map (all scalar
  types, string and string-array handling with byte-accurate string
  boundaries, v1 narrow counts), tensor table, alignment, and dequantization
  to f32 for F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q6_K. Q4_0 uses the split-nibble
  layout where byte j holds element j in the low nibble and element j+16 in
  the high nibble (verified against `ggml-quants.c:quantize_row_q4_0_ref`
  and `gguf-py`'s official `dequantize`).
- `src/own/tokenizer.cpp` - SentencePiece BPE rebuilt purely from the
  GGUF-embedded `tokenizer.ggml.tokens/scores/token_type` arrays, with byte
  fallback (`<0xNN>`) and `tokenizer.ggml.add_bos_token` handling.
- `src/own/llama_graph.cpp` - Llama-family forward pass: token gathering,
  per-layer RMSNorm, Q/K/V projections, GGML-convention interleaved RoPE
  (pairs `(2i, 2i+1)`), causal GQA attention, SwiGLU FFN, and the final
  RMSNorm plus `output.weight` (falling back to `token_embd.weight` when
  tied). Full recompute per call (no KV cache) in v0 - correctness over
  speed. Selected with `presto run <model> --engine own`.

Validation: F32 models (stories260K) match the llama.cpp backend
bit-identically on greedy decoding; stories15M Q4_0 does as well after the
Q4_0 layout fix. Quantized models diverge only where ggml's block-dot
kernels (per-block integer accumulation then scaling) differ numerically from
f32 dequantization - the next milestone ports those exact kernels.

## The MLX backend

On Apple Silicon, `src/backends/mlx_backend.cpp` implements a minimal
token-id-level Llama-family forward pass directly on mlx core ops, loading
mlx-lm directory layouts with config and quantization detection handled by
the detector. It exists so the same binary speaks the Apple-native weight
format alongside GGUF, validated in CI on macos-15 with real published MLX
weights on every push.

## Serving layer

`run_openai_server()` in `src/server_http.cpp` exposes four endpoints on
cpp-httplib: `GET /health`, `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions`. Request parsing uses the in-house RFC 8259
subset parser (`src/json_mini.hpp`); malformed bodies get HTTP 400 with a
JSON error while the server stays alive. Chat messages flatten through a
naive template ("role: content" per line).

Concurrency follows a single-owner design. One worker thread owns the
backend exclusively; HTTP handler threads only enqueue shared-pointer jobs
and wait on a condition variable. The worker waits for arrivals, sleeps
through a 50 ms coalescing window so simultaneous requests land in the same
cycle, gathers up to `PRESTO_BATCH_SLOTS` jobs, logs the gather size, and
dispatches them through `generate_many()`. If the backend declines the mix,
each job reruns sequentially. A generation mutex added in v0.2.0 keeps
model state safe regardless of handler timing. This architecture is what
lets the bandwidth wall be paid once per cycle instead of once per user.

## Model resolution

`resolve_model_path()` accepts whatever the user types. An existing literal
path wins immediately. Otherwise the resolver walks library roots in order:
`./models`, `%LOCALAPPDATA%\presto\models`, the LM Studio library
(`~/.lmstudio/models`), and `~/.presto/models`. Within each root it tries a
direct `root/<ref>` hit, then a recursive scan matching exact filename
(case-insensitive), then remembers the first unique prefix match. That is
why `presto run phi-4` finds `phi-4-Q4_K_M.gguf` anywhere in the library.
`list_known_models()` powers `presto models`, which prints every discovered
file with its size.

## Hardware adaptation

Hardware routes are compile-time selections over ggml backends, exposed as
CMake flags: `-DPRESTO_WITH_CUDA=ON` (NVIDIA, CI build gate),
`-DPRESTO_WITH_VULKAN=ON` (NVIDIA/AMD/Intel, runtime-verified on Arc 140V),
`-DPRESTO_WITH_HIP=ON` (AMD Radeon on Linux), `-DPRESTO_WITH_SYCL=ON`
(Intel XMX via oneAPI, measured 11.9 tok/s on Qwen3.5-9B versus 6.1 on
CPU), and Metal, which switches on automatically with llama.cpp on macOS.
The x86 CPU route always exists. Everything links statically
(`BUILD_SHARED_LIBS=OFF`) so the result is one self-contained executable;
shared ggml builds scatter DLLs that dynamic backend discovery cannot
reliably locate. Runtime behavior adapts along the axes described earlier:
threads by model size, context by free RAM, GPU layers by device memory,
with UMA spill handling letting 30B-class quantized models fully offload
(65/65 layers) on a 16 GB shared-memory iGPU.

### Breaking the bandwidth wall

Decode on consumer silicon is bandwidth-bound, and large models hit a wall:
R1-Distill-Qwen-32B streams weights at roughly 11.9 tok/s no matter how
idle the compute units look. Three layers attack that ceiling, and they
compose:

1. Persistent n-gram memory. The 200k-token cross-request pool lets repeat
   requests skip the model entirely for drafted spans: 23.2 s to 11.9 s
   (1.95x) on the R1-32B repeat benchmark, an effective 11.7 tok/s per user
   on the second hit.
2. Continuous batching. Sharing one weight-streaming pass across users
   divides the wall among them: 1.6x aggregate on CPU and 1.8x on the iGPU
   at 4 users, rising to 35.0 tok/s (Gemma-4-26B-A4B) and 36.0 tok/s
   (GPT-OSS-20B) at batch 8.
3. Quantization pipeline. Dropping R1-Distill-Qwen-32B from a Q4_K_M build
   that overflowed the 16 GB budget to Q3_K_M (14.84 GB, 4.8 tok/s solo)
   and then Q2_K (11.47 GB, 6.0 tok/s solo, 9.0 aggregate) restored full
   65/65 GPU offload, making 30B-class serving possible on an iGPU at all.

Stacked together, the pool plus batching turn a solo 11.9 tok/s stream into
roughly 17 tok/s combined effective throughput on repeated workloads.

## Quality gates

Three rings of verification protect the design. Unit tests
(`tests/`, 18 checks across six suites) cover the detector, engine
selection, GGUF and safetensors parsing, and the JSON parser with zero
external dependencies. The release-gate suite (`scripts/verify_release.ps1`)
runs 15 end-to-end checks: robustness against corrupt inputs, cross-process
greedy determinism, server defense, a 40-completion soak, RSS growth within
60 MB (measured 0.9 MB), and parallel clients. CI runs four jobs on every
push: ubuntu-core (build plus unit tests), linux-hw-build (CUDA and Vulkan
toolchains compile and link), windows-gguf (MSVC, real generation, tempo
gate with a floor of 100 med_tps on stories260K), and macos-mlx (real MLX
weights, Metal GGUF, tempo gate floored at 20 against the calibrated 29.5
tok/s M1 baseline). Failures always upload full stage logs as artifacts.

## Source map

| File | Role |
|---|---|
| `src/cli.cpp` | Entry point, hand-rolled argparse, commands (run/chat/bench/serve/info/models/version), Tempo Report, exit codes, version constant |
| `src/detector.cpp` | Format classification for any path; never throws, UNKNOWN plus summary on bad input |
| `src/engine.cpp` | Backend registry and routing; compile-time capability report; actionable errors for unexecutable formats |
| `src/gguf_meta.cpp` | Pure C++ GGUF header/metadata parser (header and kv map only, never tensor payloads) |
| `src/st_meta.cpp` | Safetensors JSON-header parser; tensor inventory and dtype histogram |
| `src/resolve.cpp` | Ollama-style model name resolution over ./models, presto dir, LM Studio library; `presto models` listing |
| `src/server_http.cpp` | OpenAI-compatible server: endpoints, request queue, 50 ms coalescing window, batching worker thread |
| `src/json_mini.hpp` | Dependency-free RFC 8259 subset JSON parser; depth-bounded, no exceptions |
| `src/backends/llamacpp_backend.cpp` | GGUF execution engine: adaptive tuning, prefix KV cache, KV snapshots, speculative decoding (draft + n-gram pool), continuous batching |
| `src/backends/llamacpp_backend.hpp` | LlamaCpp backend class declaration |
| `src/backends/mlx_backend.cpp` | MLX execution engine: Llama-family forward pass on mlx core ops (Apple Silicon) |
| `src/backends/mlx_backend.hpp` | MLX backend class declaration |
| `include/presto/backend.hpp` | IBackend interface, GenerateParams/Result, BatchJob, default sequential generate_many |
| `include/presto/detector.hpp` | detect_format API |
| `include/presto/engine.hpp` | select_backend and capability report API |
| `include/presto/format.hpp` | Core format types (ModelFormat enum, Detection struct, format_name) |
| `include/presto/gguf_meta.hpp` | GGUF parser API |
| `include/presto/st_meta.hpp` | Safetensors parser API |
| `include/presto/log.hpp` | Leveled header-only logger; PRESTO_LOG override |
| `include/presto/resolve.hpp` | Model resolution API |
| `include/presto/server.hpp` | run_openai_server API |
| `include/presto/runtime/status.hpp` | Provider-neutral status and result contracts |
| `include/presto/runtime/tensor.hpp` | Native dtype, shape, stride, and checked-size contracts |
| `include/presto/runtime/buffer.hpp` | Device buffer interface, aligned host storage, checked tensor views |
| `include/presto/runtime/context.hpp` | Shared-pool and logical per-sequence context capacity contracts |
| `include/presto/runtime/gguf.hpp` | Native GGUF type traits, tensor index, mapped weights, and zero-copy spans |
| `include/presto/runtime/device.hpp` | Provider-neutral device capabilities and host device |
| `include/presto/runtime/registry.hpp` | Deterministic device registration and lookup |
| `src/runtime/tensor.cpp` | Overflow-safe scalar and quantized tensor layout arithmetic |
| `src/runtime/buffer.cpp` | Host allocation and tensor-view boundary validation |
| `src/runtime/context.cpp` | Overflow-safe context-pool capacity and growth calculation |
| `src/runtime/gguf.cpp` | Complete GGUF metadata/tensor descriptor parsing and extent validation |
| `src/runtime/mapped_file.cpp` | Win32/POSIX read-only file mapping for tensor payloads |
| `src/runtime/registry.cpp` | Host provider and device registry implementation |
| `src/own/own.hpp` | Own engine contract: Tensor, GGufModel, llama_forward |
| `src/own/gguf_load.cpp` | GGUF v2/v3 parser and F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q6_K dequantization |
| `src/own/tokenizer.cpp` | SentencePiece BPE from GGUF-embedded vocab |
| `src/own/llama_graph.cpp` | Llama forward graph (RMSNorm, RoPE, GQA, SwiGLU) |
| `tests/test_main.cpp` | Zero-dependency test harness entry |
| `tests/test_detector.cpp` | Detector tests across all supported formats |
| `tests/test_engine_select.cpp` | Backend routing tests |
| `tests/test_gguf_meta.cpp` | GGUF parser tests |
| `tests/test_st_meta.cpp` | Safetensors parser tests |
| `tests/test_json_mini.cpp` | JSON parser tests |
| `tests/test_util.hpp` | Shared test helpers |
| `CMakeLists.txt` | Build: C++20, FetchContent pins (llama.cpp b21e4de74, cpp-httplib v0.15.3, MLX v0.32.1), hardware flags, static linking, test target |
| `.github/workflows/ci.yml` | Four-job CI with hardware build gates, real-generation smokes, and calibrated tempo gates |
| `scripts/smoke_gguf.ps1` / `smoke_gguf.sh` | Real GGUF generation smokes (Windows CI, macOS Metal) |
| `scripts/smoke_mlx.sh` | Real MLX weights generation smoke (macOS CI) |
| `scripts/verify_native_gguf.ps1` / `verify_native_gguf.sh` | Native tensor-index verification across local GGUF files |
| `scripts/verify_release.ps1` | Release-gate suite: 15 robustness/determinism/server/soak/concurrency checks |
| `scripts/bench_hf.py` | HuggingFace Transformers reference benchmarks for head-to-head comparisons |
| `scripts/bench_matrix.py` | Multi-model bench matrix driver |
| `scripts/route_probe.ps1` | Probes which hardware route a build reports |
| `scripts/bigmodel_pipeline.py` | Download-quantize-serve pipeline for 30B-class models |
| `scripts/build_sycl.bat` | Intel SYCL build helper (oneAPI environment on Windows) |
| `scripts/make_demo_gif.py` | README demo GIF renderer |
| `scripts/dl120.bat` / `dl_phi.bat` | Model downloaders |

## Environment reference

| Variable | Effect | Default |
|---|---|---|
| `PRESTO_THREADS` | Inference thread count override | auto: 4 for models < 1 GB, else min(hw, 8) |
| `PRESTO_CTX` | Context size override | RAM tiers: 4096 / 2048 / 1024 |
| `PRESTO_GPU_LAYERS` | Layer offload count | -1 (auto/max); 0 forces CPU-only |
| `PRESTO_BATCH_SLOTS` | Max requests gathered per engine pass | 4 |
| `PRESTO_PREFIX_CACHE` | Set `0` to disable prefix KV reuse | on |
| `PRESTO_FLASH_ATTN` | `1` forces flash attention (auto-retry if rejected) | off |
| `PRESTO_KV` | `q8_0` or `q4_0` KV cache quantization (V needs FA) | f16/f16 |
| `PRESTO_POLL` | Spinning threadpool poll interval; > 0 enables | off |
| `PRESTO_SPEC_K` | Draft-model proposals per round | 6 |
| `PRESTO_LOOKUP_K` | Prompt-lookup proposals per round | 16 |
| `PRESTO_LOOKUP_N` | Prompt-lookup seed length | 8 |
| `PRESTO_LOOKUP_MIN` | Minimum prompt tokens for lookup speculation | 32 |
| `PRESTO_KV_SNAPSHOT` | `0` disables disk KV snapshots | on |
| `PRESTO_SNAPSHOT_MIN` | Minimum prompt tokens to snapshot | 64 |
| `PRESTO_LOG` | Log level: debug, info, warn, error | info |
| `PRESTO_SMOKE` | `1` emits machine-readable smoke markers for CI | off |
