#!/usr/bin/env bash
# presto - GGUF smoke test for unix CI
# Downloads a tiny public GGUF and performs real token generation.
set -uo pipefail

MODEL_URL="${1:-https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories260K.gguf}"
OUT_PATH="${2:-models/stories260K.gguf}"
MAX_TOKENS="${3:-16}"
PROMPT="${4:-Once upon a time}"
BIN="${PRESTO_BIN:-./build/presto}"

echo "::group::[smoke-gguf] download model ($MODEL_URL)"
mkdir -p "$(dirname "$OUT_PATH")"
if ! curl -L --fail --retry 3 --silent --show-error -o "$OUT_PATH" "$MODEL_URL"; then
    echo "::endgroup::"
    echo "[smoke-gguf] FAILED: download"
    exit 1
fi
SIZE=$(wc -c <"$OUT_PATH")
if [ "${SIZE:-0}" -lt 1000000 ]; then
    echo "::endgroup::"
    echo "[smoke-gguf] FAILED: model suspiciously small ($SIZE bytes)"
    exit 1
fi
echo "[smoke-gguf] model size: $SIZE bytes"
echo "::endgroup::"

echo "::group::[smoke-gguf] locate binary"
if [ ! -x "$BIN" ]; then
    BIN=$(find . -type f -name presto -perm -u+x 2>/dev/null | head -n1)
fi
if [ -z "${BIN}" ] || [ ! -x "$BIN" ]; then
    echo "::endgroup::"
    echo "[smoke-gguf] FAILED: presto binary not found"
    exit 1
fi
echo "[smoke-gguf] using $BIN"
"$BIN" version || true
echo "::endgroup::"

echo "::group::[smoke-gguf] generate $MAX_TOKENS tokens"
set +e
PRESTO_SMOKE=1 "$BIN" run "$OUT_PATH" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" 2>&1 |
    tee smoke-gguf.log
STATUS=${PIPESTATUS[0]}
set -e
echo "::endgroup::"

if ! grep -q '\[presto-smoke\].*ok=true' smoke-gguf.log; then
    echo "[smoke-gguf] FAILED: success marker missing"
    tail -50 smoke-gguf.log || true
    exit 1
fi
echo "[smoke-gguf] SUCCESS"
exit "$STATUS"
