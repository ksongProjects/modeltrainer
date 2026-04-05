#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${QP_ROCM_VENV:-$HOME/.venvs/quant-platform-rocm}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "ROCm app environment not found at ${VENV_PATH}."
  echo "Run scripts/wsl_rocm/setup_app_env.sh first."
  exit 1
fi

source "${VENV_PATH}/bin/activate"

UBUNTU_VERSION="$(. /etc/os-release && echo "${VERSION_ID:-}")"
PYTHON_SHORT="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "${UBUNTU_VERSION}:${PYTHON_SHORT}" in
  "24.04:3.12")
    TORCH_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl"
    TORCHVISION_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
    TRITON_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl"
    TORCHAUDIO_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
    ;;
  "22.04:3.10")
    TORCH_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp310-cp310-linux_x86_64.whl"
    TORCHVISION_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp310-cp310-linux_x86_64.whl"
    TRITON_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp310-cp310-linux_x86_64.whl"
    TORCHAUDIO_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp310-cp310-linux_x86_64.whl"
    ;;
  *)
    echo "Unsupported Ubuntu/Python combination for the current ROCm wheel helper: ${UBUNTU_VERSION} / ${PYTHON_SHORT}"
    echo "Use Ubuntu 24.04 + Python 3.12 or Ubuntu 22.04 + Python 3.10."
    exit 1
    ;;
esac

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

python -m pip install --upgrade pip wheel
python -m pip install numpy==1.26.4

cd "${WORK_DIR}"
wget "${TORCH_URL}"
wget "${TORCHVISION_URL}"
wget "${TRITON_URL}"
wget "${TORCHAUDIO_URL}"

python -m pip uninstall -y torch torchvision triton torchaudio || true
python -m pip install ./*.whl

# AMD's WSL guide instructs removing the bundled HSA runtime so torch uses the
# WSL-compatible runtime provided by the ROCm stack.
TORCH_LIB_DIR="$(python - <<'PY'
import os
import site
import sys

candidates = [path for path in site.getsitepackages() if os.path.isdir(path)]
for root in candidates:
    lib_dir = os.path.join(root, "torch", "lib")
    if os.path.isdir(lib_dir):
        print(lib_dir)
        sys.exit(0)
sys.exit(1)
PY
)"
rm -f "${TORCH_LIB_DIR}"/libhsa-runtime64.so*

python - <<'PY'
import json
import torch

print(json.dumps({
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "hip_version": getattr(torch.version, "hip", None),
}, indent=2))
PY

cat <<EOF

ROCm PyTorch wheels installed into ${VENV_PATH}.

Next steps:
1. Run scripts/wsl_rocm/verify_rocm.sh.
2. Start the backend with scripts/wsl_rocm/start_backend.sh.
EOF
