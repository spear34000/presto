#!/usr/bin/env bash
# presto - MLX smoke test for macOS Apple Silicon CI
# Converts a tiny public HF model to MLX layout, then performs REAL token
# generation through the presto MLX backend.
set -uo pipefail

HF_MODEL="${1:-hf-internal-testing/tiny-random-LlamaForCausalLM}"
OUT_DIR="${2:-./tiny_mlx}"
MAX_TOKENS="${3:-8}"
BIN="${PRESTO_BIN:-./build/presto}"

echo "::group::[smoke-mlx] environment"
uname -a
python3 --version
python3 -m pip install --quiet --upgrade mlx-lm huggingface_hub || exit 1
python3 -c "import mlx.core as mx; print('mlx', mx.__version__ if hasattr(mx,'__version__') else 'ok')"
echo "::endgroup::"

echo "::group::[smoke-mlx] convert $HF_MODEL -> $OUT_DIR"
rm -rf "$OUT_DIR"
if ! python3 -m mlx_lm convert --hf-path "$HF_MODEL" --mlx-path "$OUT_DIR"; then
    echo "[smoke-mlx] python module form failed; trying API form"
    python3 -c "
from mlx_lm import convert
convert('$HF_MODEL', '$OUT_DIR')
print('converted ok')
" || { echo "::endgroup::"; echo "[smoke-mlx] FAILED: conversion"; exit 1; }
fi
ls -la "$OUT_DIR"
cat "$OUT_DIR/config.json" || true
echo "::endgroup::"

echo "::group::[smoke-mlx] locate binary"
if [ ! -x "$BIN" ]; then
    BIN=$(find . -type f -name presto -perm -u+x 2>/dev/null | head -n1)
fi
if [ -z "${BIN}" ] || [ ! -x "$BIN" ]; then
    echo "::endgroup::"
    echo "[smoke-mlx] FAILED: presto binary not found"
    exit 1
fi
"$BIN" info "$OUT_DIR" || true
echo "::endgroup::"

echo "::group::[smoke-mlx] generate $MAX_TOKENS tokens (token-id level)"
set +e
PRESTO_SMOKE=1 "$BIN" run "$OUT_DIR" --prompt-tokens "1,2,3" --max-tokens "$MAX_TOKENS" 2>&1 |
    tee smoke-mlx.log
STATUS=${PIPESTATUS[0]}
set -e
echo "::endgroup::"

if ! grep -q '\[presto-smoke\].*ok=true' smoke-mlx.log; then
    echo "[smoke-mlx] FAILED: success marker missing"
    tail -50 smoke-mlx.log || true
    exit 1
fi
echo "[smoke-mlx] SUCCESS"
exit "$STATUS"
