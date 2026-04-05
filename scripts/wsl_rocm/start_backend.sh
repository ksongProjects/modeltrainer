#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${QP_ROCM_VENV:-$HOME/.venvs/quant-platform-rocm}"
HOST="${QUANT_PLATFORM_HOST:-0.0.0.0}"
PORT="${QUANT_PLATFORM_PORT:-8000}"
RELOAD="${QUANT_PLATFORM_RELOAD:-false}"
ROCDXG_SRC_DIR="${ROCDXG_SRC_DIR:-$HOME/.cache/librocdxg}"
ROCDXG_ENV_FILE="${ROCDXG_ENV_FILE:-$ROCDXG_SRC_DIR/wsl-rocdxg.env}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "ROCm app environment not found at ${VENV_PATH}."
  echo "Run scripts/wsl_rocm/setup_app_env.sh inside WSL first."
  exit 1
fi

cd "${ROOT_DIR}"
source "${VENV_PATH}/bin/activate"
if [[ -f "${ROCDXG_ENV_FILE}" ]]; then
  source "${ROCDXG_ENV_FILE}"
else
  export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
fi
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export QUANT_PLATFORM_HOST="${HOST}"
export QUANT_PLATFORM_PORT="${PORT}"
export QUANT_PLATFORM_RELOAD="${RELOAD}"

echo "Starting quant backend from ${ROOT_DIR}"
echo "Using ROCm env ${VENV_PATH}"
echo "Binding backend to ${HOST}:${PORT}"
echo "HSA_ENABLE_DXG_DETECTION=${HSA_ENABLE_DXG_DETECTION}"

python -m quant_platform.main
