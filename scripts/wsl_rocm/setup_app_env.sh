#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${QP_ROCM_VENV:-$HOME/.venvs/quant-platform-rocm}"

sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git wget

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Install the app without the generic ml extras. ROCm PyTorch wheels should be
# installed separately so pip does not replace them with a CPU-only wheel.
python -m pip install -e "${ROOT_DIR}[dev]"

cat <<EOF

Base app environment created at ${VENV_PATH}.

Next steps for Radeon WSL + ROCm:
1. Install the AMD WSL driver on Windows and the ROCm WSL stack inside Ubuntu.
2. Install AMD's ROCm PyTorch wheels inside ${VENV_PATH}.
3. Run scripts/wsl_rocm/verify_rocm.sh to confirm rocminfo and torch can see the GPU.
4. Start the backend with scripts/wsl_rocm/start_backend.sh.

This script intentionally does not install generic torch extras, because the
project's [ml] extras would overwrite ROCm-specific wheels with the wrong build.
EOF
