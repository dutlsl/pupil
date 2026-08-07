#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CC_BIN="${CC:-cc}"

"${CC_BIN}" -O3 -shared -fPIC \
  "${SCRIPT_DIR}/_native_binarep.c" \
  -o "${SCRIPT_DIR}/_native_binarep.so"

echo "Built ${SCRIPT_DIR}/_native_binarep.so"
