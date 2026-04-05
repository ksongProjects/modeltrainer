#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${QP_ROCM_VENV:-$HOME/.venvs/quant-platform-rocm}"
ROCDXG_SRC_DIR="${ROCDXG_SRC_DIR:-$HOME/.cache/librocdxg}"
ROCDXG_ENV_FILE="${ROCDXG_ENV_FILE:-$ROCDXG_SRC_DIR/wsl-rocdxg.env}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "ROCm app environment not found at ${VENV_PATH}."
  exit 1
fi

source "${VENV_PATH}/bin/activate"
if [[ -f "${ROCDXG_ENV_FILE}" ]]; then
  source "${ROCDXG_ENV_FILE}"
else
  export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
fi
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "== rocminfo =="
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | sed -n '1,120p'
else
  echo "rocminfo not found. Install the ROCm WSL stack first."
fi

echo
echo "== rocdxg =="
echo "HSA_ENABLE_DXG_DETECTION=${HSA_ENABLE_DXG_DETECTION:-unset}"
if command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p | grep librocdxg || echo "librocdxg not found in ldconfig output."
fi

echo
echo "== torch runtime =="
python - <<'PY'
import json
import torch
from quant_platform.runtime_profiles import runtime_capabilities, runtime_self_check

print(json.dumps(runtime_capabilities(), indent=2))
print(json.dumps(runtime_self_check({"compute_target": "cuda", "precision_mode": "auto", "batch_size": 16, "sequence_length": 12}, model_kind="gru", input_dim=8), indent=2))
PY
