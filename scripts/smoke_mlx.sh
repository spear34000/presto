#!/usr/bin/env bash
# presto - MLX smoke test for macOS Apple Silicon CI
#
# Stage A (preferred): download a real published non-quantized MLX model
#                      (mlx-community/SmolLM-135M-fp16, llama architecture)
#                      and generate tokens through the presto MLX backend.
# Stage B (fallback) : synthesize a small deterministic MLX-layout directory
#                      (same layout unquantized mlx-lm produces) so a hub
#                      outage cannot mask backend regressions.
set -uo pipefail

REAL_MODEL="mlx-community/SmolLM-135M-fp16"
FALLBACK_DIR="${1:-./tiny_mlx_synth}"
MAX_TOKENS="${2:-8}"
BIN="${PRESTO_BIN:-./build/presto}"

echo "::group::[smoke-mlx] environment"
uname -a
python3 --version
python3 -m pip install --quiet --upgrade huggingface_hub numpy safetensors || exit 1
echo "::endgroup::"

MODEL_DIR=""

echo "::group::[smoke-mlx] stage A: real model $REAL_MODEL"
# Single snapshot call with explicit allow_patterns: requesting only what we
# need keeps huggingface_hub's completeness check satisfied.
MODEL_DIR=$(python3 - "$REAL_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(
        sys.argv[1],
        allow_patterns=["*.json", "*.safetensors"],
    )
    print(p)
except Exception as e:
    print(f"download failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
)
if [ -n "$MODEL_DIR" ]; then
    echo "[smoke-mlx] stage A model dir: $MODEL_DIR"
else
    echo "[smoke-mlx] stage A download failed; will use fallback"
fi
echo "::endgroup::"

if [ -z "$MODEL_DIR" ]; then
    echo "::group::[smoke-mlx] stage B: synthetic MLX-layout fixture"
    if ! python3 - "$FALLBACK_DIR" <<'PY'
import json, sys, os
import numpy as np
from safetensors.numpy import save_file

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
H, L, NH, NKV, FF, V = 32, 2, 4, 2, 96, 128
DH = H // NH
rng = np.random.default_rng(42)
t = {"model.embed_tokens.weight": rng.standard_normal((V, H)).astype(np.float32)}
for l in range(L):
    p = f"model.layers.{l}."
    t[p + "input_layernorm.weight"] = np.ones(H, np.float32)
    t[p + "post_attention_layernorm.weight"] = np.ones(H, np.float32)
    t[p + "self_attn.q_proj.weight"] = rng.standard_normal((NH * DH, H)).astype(np.float32)
    t[p + "self_attn.k_proj.weight"] = rng.standard_normal((NKV * DH, H)).astype(np.float32)
    t[p + "self_attn.v_proj.weight"] = rng.standard_normal((NKV * DH, H)).astype(np.float32)
    t[p + "self_attn.o_proj.weight"] = rng.standard_normal((H, NH * DH)).astype(np.float32)
    t[p + "mlp.gate_proj.weight"] = rng.standard_normal((FF, H)).astype(np.float32)
    t[p + "mlp.up_proj.weight"] = rng.standard_normal((FF, H)).astype(np.float32)
    t[p + "mlp.down_proj.weight"] = rng.standard_normal((H, FF)).astype(np.float32)
t["model.norm.weight"] = np.ones(H, np.float32)
t["lm_head.weight"] = rng.standard_normal((V, H)).astype(np.float32)
save_file(t, os.path.join(out, "model.safetensors"))
cfg = {
    "model_type": "llama",
    "hidden_size": H,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": NKV,
    "intermediate_size": FF,
    "vocab_size": V,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f)
print("fixture written to", out)
PY
    then
        echo "::endgroup::"
        echo "[smoke-mlx] FAILED: could not prepare any model"
        exit 1
    fi
    MODEL_DIR="$FALLBACK_DIR"
    STAGE="B-synthetic-fixture"
else
    STAGE="A-real-model($REAL_MODEL)"
fi
echo "[smoke-mlx] testing stage=$STAGE dir=$MODEL_DIR"
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
"$BIN" version || true
"$BIN" info "$MODEL_DIR" || true
echo "::endgroup::"

echo "::group::[smoke-mlx] generate $MAX_TOKENS tokens via MLX backend ($STAGE)"
set +e
PRESTO_SMOKE=1 "$BIN" run "$MODEL_DIR" --prompt-tokens "1,2,3" --max-tokens "$MAX_TOKENS" 2>&1 |
    tee smoke-mlx.log
STATUS=${PIPESTATUS[0]}
set -e
echo "::endgroup::"

if ! grep -q '\[presto-smoke\].*ok=true' smoke-mlx.log; then
    echo "[smoke-mlx] FAILED: success marker missing (stage=$STAGE)"
    tail -50 smoke-mlx.log || true
    exit 1
fi
echo "[smoke-mlx] SUCCESS stage=$STAGE"
exit "$STATUS"
