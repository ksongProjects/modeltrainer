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

# LightGBM may use a prebuilt wheel, but these packages make source builds
# reliable on Ubuntu if pip needs to compile locally.
sudo apt update
sudo apt install -y build-essential cmake libomp-dev

python -m pip install --upgrade pip wheel setuptools
python -m pip install -e "${ROOT_DIR}[ml-runtime]"

python - <<'PY'
import importlib
import json

packages = {}
for name in ("torch", "lightgbm", "transformers"):
    try:
        mod = importlib.import_module(name)
        packages[name] = {
            "installed": True,
            "version": getattr(mod, "__version__", None),
        }
    except Exception as exc:
        packages[name] = {
            "installed": False,
            "error": str(exc),
        }

print(json.dumps(packages, indent=2))
PY

cat <<EOF

Optional ML packages installed into ${VENV_PATH}.

Installed safely:
- lightgbm
- transformers

This helper does not reinstall torch, so your ROCm PyTorch build stays intact.
If you plan to use models that require SentencePiece tokenizers, install it later with:
  python -m pip install sentencepiece
EOF
