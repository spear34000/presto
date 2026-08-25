# presto verification report

- date: 2026-08-25 11:09 +09:00
- host: Windows 11, Lunar Lake CPU 8 threads, Intel Arc 140V iGPU, 31.5GB RAM
- binaries: build-full (CPU) / build-vulkan (Vulkan), commit 4c31a4e

| Area | Check | Result | Detail |
|---|---|---|---|
| robustness | truncated gguf -> run fails cleanly | PASS | exit=5 |
| robustness | random bytes .gguf | PASS | exit=3 classified unknown |
| robustness | zero-byte model file | PASS | exit=3 |
| robustness | nonexistent path (run) | PASS | exit=3 (3=unsupported/missing) |
| robustness | directory passed to run | PASS | exit=3 |
| determinism | stories260K.gguf cross-process greedy | PASS | 20 token(s): 423,419,426,346,397,354,419,267,262,411,411,261,370,268,414,444,426,346,397,354 |
| determinism | stories15M-q4_0.gguf cross-process greedy | PASS | 20 token(s): 271,29889,940,471,263,1407,9796,322,9045,29891,767,29889,940,750,263,4802,17819,373,670,3700 |
| determinism | SmolLM2-135M-Instruct-Q8_0.gguf cross-process greedy | PASS | 20 token(s): 30,198,198,504,1796,1341,314,288,722,260,476,48752,18106,18,365,98,25,284,476,1881 |
| server | startup health | PASS |  |
| server-defense | malformed JSON -> 400, server alive | PASS | http=400 |
| server-defense | empty object rejected | PASS | http=400 |
| soak | 40 sequential chat completions | PASS | fails=0 med=165ms |
| safety | RSS growth over soak <= 60MB | PASS | growth=0.9MB |
| concurrency | 6 parallel clients all 200 | PASS | codes=200,200,200,200,200,200 |
| gpu-vulkan | capability reports vulkan=yes | PASS |  |

**15 / 15 checks passed**
