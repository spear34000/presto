# Changelog

All notable changes to **presto** are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/).

Every performance number below comes from the built-in `presto bench`
(Tempo Report: min/median/max with sigma) or from scripts checked into the
repository. The reference machine is Windows 11 on Lunar Lake (Intel Arc
140V iGPU, 31.5 GB RAM) unless a note says otherwise; decode is greedy
unless stated.

## [Unreleased]

### Added

- The first dependency-free native runtime layer: structured status/results,
  overflow-checked tensor metadata, aligned host buffers and views, device
  contracts, and deterministic provider registration. It builds and tests
  with `PRESTO_WITH_LLAMACPP=OFF` as the foundation for replacing ggml model
  execution family by family.
- Native GGUF ingestion now indexes every tensor without llama.cpp, validates
  shapes, quantization block sizes, alignment, overlap, and file bounds, and
  exposes exact tensor payloads through read-only Windows/POSIX memory maps.
  `presto info` reports the native tensor data offset, byte total, and type
  inventory.
- `presto chat`: an interactive multi-turn REPL. Each message is appended to
  the running conversation, replies print with per-turn throughput, and
  `exit`, `quit`, `/q`, or Ctrl-Z ends the session. Options mirror `run`
  (`--temp`, `--max-tokens`) (`4613f53`).
- A friendlier command surface: `presto models` now prints each discovered
  model with its size in MB, the top-level help leads with a quick-start
  block, and referencing a model that cannot be found prints a hint pointing
  at `presto models` (`4613f53`).

These additions build on the size-aware CPU thread auto-tuning that landed
in v0.3.1: cache-resident models (under 1 GB) run with four inference
threads instead of the naive core count, which measured +43% decode
throughput on SmolLM2-135M.

### Fixed

- Windows command arguments and console output now remain UTF-8, and chat/run
  requests use the model's embedded GGUF chat template instead of feeding raw
  user text. This fixes Korean prompts producing mojibake and repeated garbage
  tokens.
- Context growth now checks capacity per logical sequence. With four batch
  slots, a shared `n_ctx=2048` pool exposes 512 tokens per sequence; requests
  that exceed that limit grow the pool before decode instead of failing near
  token 498 with `failed to find a memory slot`.

## [v0.3.1] - 2026-08-25

### Performance

- Size-aware CPU thread auto-tuning (`7adc905`). Small models live in cache
  and are latency-bound, so extra threads only add synchronization overhead.
  presto now reads the model file size at load and caps inference threads at
  4 for models under 1 GB while letting large weight-streaming models scale
  to 8 threads. SmolLM2-135M measured 228 tok/s at 4 threads versus 136 at
  8, a +43% gain over the previous heuristic.
- Auto-tuned CPU route on the reference machine: SmolLM2-135M Q8_0 reaches
  194 tok/s solo and 387 tok/s aggregate across 8 concurrent users;
  Qwen2.5-0.5B Q8_0 holds 82 tok/s solo; Qwen3.5-9B Q4_K_M decodes at
  8.2 tok/s.

### Verification

- Release-gate report refreshed against the v0.3.0 binaries (commit
  `4c31a4e`): 15/15 checks pass (`06bbcac`). Coverage includes truncated,
  random-byte, zero-byte, missing, and directory inputs failing cleanly with
  the right exit codes; cross-process greedy determinism on stories260K,
  stories15M, and SmolLM2-135M; malformed JSON rejected with HTTP 400 while
  the server stays alive; a 40-completion soak with zero failures (median
  165 ms); RSS growth of 0.9 MB against a 60 MB budget; and 6 parallel
  clients all returning HTTP 200.

### Changed

- Version bumped to 0.3.1 (`a74d9fc`).

## [v0.3.0] - 2026-08-25

Tagged "persistent memory breaks the wall" (`e13bd9d`). This cycle turned a
single-request runtime into a serving stack and attacked the
memory-bandwidth ceiling from three directions at once.

### Added

- Persistent cross-request n-gram pool (`4c31a4e`). Every completed exchange
  whose prompt has at least 32 tokens appends its prompt and output tokens
  to a pool capped at 200k tokens that survives for the server's lifetime.
  Prompt-lookup speculation drafts from this history when the current prompt
  alone has no match, so a repeated request reuses the first request's own
  output without touching the model weights. Measured on
  R1-Distill-Qwen-32B with a repetitive reasoning prompt: 23.2 s down to
  11.9 s, a 1.95x speedup.
- Continuous batching server (`a5ce614`). One worker thread owns the
  engine; concurrent HTTP requests are gathered so a single
  weight-streaming decode pass produces one token for every waiting user
  and the bandwidth wall is paid once, shared. Aggregate throughput
  measured 1.6x sequential serving on CPU and 1.8x on the Arc iGPU with
  4 users. Greedy requests batch automatically; sampled (temperature > 0)
  requests keep sequential semantics.
- 50 ms coalescing window in the server queue plus gather-size logging
  (`2df6a59`), giving simultaneous arrivals time to land in one engine pass.
- Zero-config prompt-lookup speculation (`99af242`). Greedy decoding
  automatically drafts n-gram spans from the prompt and verifies them in one
  target pass. Outputs are bit-identical to the plain path on any model,
  with no draft download and no flags. Copy-heavy workloads (summarize,
  RAG, code-edit) measured +27-51% tok/s on Qwen2.5-0.5B.
- Ollama-style model resolution and the `presto models` command (`d2c566c`).
  Bare names such as `phi-4` resolve against `./models`, the presto library
  directory, and the LM Studio library through exact-path, exact-filename,
  then unique-prefix matching.
- Adaptive auto-tuning plus persistent prompt snapshot (`3d4281d`). Context
  size derives from available RAM (1024/2048/4096 tiers), GPU layer offload
  follows device memory, and long prompts save their post-prefill KV state
  to disk keyed by token hash so future runs skip seconds of prefill even
  after a restart.
- Tooling: multi-model bench matrix, demo GIF renderer, and model
  downloaders (`90fa7cb`); route probe and big-model pipeline scripts
  (`15c757e`).
- Branding: official logo and README polish (`c0aa519`, `d7e08a3`,
  `3d94bce`).

### Performance

- Large-model serving on the 16 GB Arc iGPU, full 65/65 layer offload with
  UMA spill handled:

  | Model | Quant | Size | Solo tok/s | Batch x8 aggregate |
  |---|---|---|---|---|
  | R1-Distill-Qwen-32B | Q3_K_M | 14.84 GB | 4.8 | 7.2 |
  | R1-Distill-Qwen-32B | Q2_K | 11.47 GB | 6.0 | 9.0 |
  | Gemma-4-26B-A4B | Q3_K_M | - | 14.2 | 35.0 |
  | GPT-OSS-20B | Q4_K_M | 11.62 GB | 21.7 | 36.0 |

- Stacked effects: the persistent pool turns the 11.9 tok/s bandwidth wall
  into an effective 11.7 tok/s per user on a repeat hit, and roughly
  17 tok/s combined when batching is layered on top.

### Security

- Server generation mutex protecting model state across concurrent HTTP
  handlers, introduced together with the release-gate verification suite
  scoring 15/15 (`a30a123`).

## [v0.2.0] - 2026-08-23

Headline commit "presto takes its tempo seriously" (`24f0083`). The focus
shifted from working to fast, with every claim backed by a measurement.

### Added

- Prefix KV-cache reuse (`2028b3c`), absorbing SGLang's RadixAttention idea:
  consecutive requests sharing token history skip re-prefill for the shared
  part, with outputs bit-identical to the uncached path. Measured multi-turn
  scenario skipped 97% of prompt tokens.
- Draft-model speculative decoding (`fbd19f7`): `--draft small.gguf` adds a
  small companion model as the proposer for greedy decoding with a
  bit-identical output guarantee, including graceful mid-request fallback on
  architectures whose KV cache cannot roll back proposals (Qwen3.5 hybrid
  attention).
- Opt-in kernel levers and a multi-model validation suite (`1bc8364`):
  flash attention, KV-cache quantization, and spinning threadpools, all
  behind environment variables and off by default.

### Performance

- Head-to-head on stories15M Q4_0 (`32883fd`): presto 1724 ± 2.8% tok/s
  versus llama.cpp stock `llama-bench` at 1720 ± 3.6% (statistical parity)
  and 1.70x faster than ollama 0.32.15 at 1016 tok/s.
- Against HuggingFace Transformers fp32 CPU defaults (`b0e62f4`):
  SmolLM2-135M-Instruct 191 vs 30.9 tok/s (6.2x faster);
  Qwen2.5-0.5B-Instruct 81.3 vs 18.0 tok/s (4.5x faster).

### CI

- Linux hardware build gate extended to CUDA and Vulkan (`5dd6e98`,
  `4ffef8e`): nvcc installed on the runner, a single CUDA architecture
  pinned for speed, and build parallelism capped at 2 after nvcc plus
  shader generation OOM'd the 7 GB runner.
- Metal tempo gate calibrated to the measured 29.5 tok/s stories260K
  baseline on M1 (`8f5b40a`).

### Docs

- README rebuilt llama.cpp-style with badges, table of contents, and
  measured tables throughout (`3d94bce`).

### Security

- Generation mutex guarding server-side model state, plus the release-gate
  verification suite covering robustness, determinism, server defense, soak,
  memory growth, and concurrency (`a30a123`).

## [v0.1.0] - 2026-08-23

Initial release (`c7e4fea`): a unified C++ LLM runtime that inspects every
major open-source model format and executes GGUF and MLX natively.

### Added

- Format detection for GGUF files, mlx-lm directories, SafeTensors archives,
  PyTorch checkpoints (zip container), AWQ directories, and GPTQ
  directories. Malformed input classifies as UNKNOWN with an explanatory
  summary instead of throwing.
- Pure C++ GGUF header/metadata parser and a safetensors JSON-header parser;
  neither reads tensor payloads.
- Execution backends: llama.cpp for GGUF (pinned to a locally verified
  commit) and an MLX core-op backend for Apple Silicon.
- OpenAI-compatible HTTP server exposing `/v1/chat/completions` and
  `/v1/completions` on cpp-httplib.
- CLI commands `run`, `info`, `bench`, `serve`, and `version`, with stable
  exit codes (0 ok, 2 usage, 3 unsupported format, 4 backend unavailable,
  5 failure).
- Zero-dependency unit tests covering the detector, engine selection, GGUF
  metadata, safetensors metadata, and the mini JSON parser.

### Fixed

- Seven MLX bring-up defects found while aligning with mlx v0.32.1
  (`02d2ce2`, `cd63cc4`, `8c83127`, `21142ef`, `03a6367`, `c4d69ca`,
  `8b94022`): C++ API alignment, umbrella header name (`mlx.h`, no
  `core.h`), header include prefix (`mlx/mlx.h`), projection-weight
  transpose to [in,out] before matmul, non-default-constructible
  `mlx::array`, attention score axis ([nh,Tq,Tk], not [T,nh,nh]), and
  tied-embedding lm_head needing a transposed [H,V] view.
- Detector now classifies unquantized MLX-layout directories as MLX_DIR
  (`e6ef57b`).

### CI

- macOS MLX job pinned to mlx v0.32.1 on macos-15 runners (Xcode 16+),
  switched to real published MLX weights, and made smoke failures
  unmaskable (`fbb90d6`, `a6d8957`).

---

Release notes by version. v0.1.0 and v0.2.0 predate tagging; their links
point at the release commits.

[Unreleased]: https://github.com/spear34000/presto/compare/v0.3.1...HEAD
[v0.3.1]: https://github.com/spear34000/presto/compare/v0.3.0...v0.3.1
[v0.3.0]: https://github.com/spear34000/presto/tree/v0.3.0
[v0.2.0]: https://github.com/spear34000/presto/tree/24f0083
[v0.1.0]: https://github.com/spear34000/presto/tree/c7e4fea
