#!/usr/bin/env bash
set -euo pipefail

presto_path="${1:-./build/presto}"
models_path="${2:-./models}"
mapfile -d '' files < <(find "$models_path" -maxdepth 1 -type f -name '*.gguf' -print0 | sort -z)
if (( ${#files[@]} == 0 )); then
  echo "No GGUF files found in $models_path" >&2
  exit 1
fi

for file in "${files[@]}"; do
  "$presto_path" info "$file"
done
echo "Native GGUF inspection passed: ${#files[@]} file(s)"
