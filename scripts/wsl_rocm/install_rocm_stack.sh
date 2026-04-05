#!/usr/bin/env bash
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.2}"
ROCDXG_REPO_URL="${ROCDXG_REPO_URL:-https://github.com/ROCm/librocdxg.git}"
ROCDXG_REF="${ROCDXG_REF:-}"
ROCDXG_SRC_DIR="${ROCDXG_SRC_DIR:-$HOME/.cache/librocdxg}"

UBUNTU_VERSION="$(. /etc/os-release && echo "${VERSION_ID:-}")"

case "${UBUNTU_VERSION}" in
  "24.04")
    AMDGPU_DEB_URL="https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${ROCM_VERSION}.70200-1_all.deb"
    ;;
  "22.04")
    AMDGPU_DEB_URL="https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/jammy/amdgpu-install_${ROCM_VERSION}.70200-1_all.deb"
    ;;
  *)
    echo "Unsupported Ubuntu version for the current WSL ROCm helper: ${UBUNTU_VERSION}"
    echo "Use Ubuntu 22.04 or 24.04 in WSL."
    exit 1
    ;;
esac

DEB_NAME="$(basename "${AMDGPU_DEB_URL}")"
WINDOWS_SDK_BASE="${WIN_SDK_BASE:-/mnt/c/Program Files (x86)/Windows Kits/10/Include}"

discover_windows_sdk() {
  if [[ -n "${WIN_SDK_PATH:-}" ]]; then
    printf '%s\n' "${WIN_SDK_PATH}"
    return 0
  fi

  if [[ ! -d "${WINDOWS_SDK_BASE}" ]]; then
    return 1
  fi

  find "${WINDOWS_SDK_BASE}" -mindepth 1 -maxdepth 1 -type d -name '10.*' | sort -V | tail -n 1
}

WIN_SDK_ROOT="$(discover_windows_sdk || true)"
WIN_SDK_SHARED_DIR="${WIN_SDK_ROOT:+${WIN_SDK_ROOT}/shared}"

if [[ -z "${WIN_SDK_ROOT}" || ! -d "${WIN_SDK_SHARED_DIR}" ]]; then
  echo "Windows SDK was not found from WSL."
  echo "Install the Windows 10/11 SDK on Windows, then rerun this script."
  echo "Expected a path like: /mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0"
  echo "You can also override detection with WIN_SDK_PATH='/mnt/c/.../Include/<version>'."
  exit 1
fi

sudo apt update
sudo apt install -y build-essential cmake git python3-setuptools python3-wheel wget
wget -O "${DEB_NAME}" "${AMDGPU_DEB_URL}"
sudo apt install -y "./${DEB_NAME}"
sudo apt update
sudo usermod -a -G render,video "${USER}" || true
sudo apt install -y rocm rocminfo

mkdir -p "$(dirname "${ROCDXG_SRC_DIR}")"
if [[ -d "${ROCDXG_SRC_DIR}/.git" ]]; then
  git -C "${ROCDXG_SRC_DIR}" fetch --all --tags
else
  git clone "${ROCDXG_REPO_URL}" "${ROCDXG_SRC_DIR}"
fi

if [[ -n "${ROCDXG_REF}" ]]; then
  git -C "${ROCDXG_SRC_DIR}" checkout "${ROCDXG_REF}"
fi

BUILD_DIR="${ROCDXG_SRC_DIR}/build"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cmake -S "${ROCDXG_SRC_DIR}" -B "${BUILD_DIR}" -DWIN_SDK="${WIN_SDK_SHARED_DIR}"
cmake --build "${BUILD_DIR}" --parallel
sudo cmake --install "${BUILD_DIR}"
sudo ldconfig

cat <<'EOF' > "${ROCDXG_SRC_DIR}/wsl-rocdxg.env"
export HSA_ENABLE_DXG_DETECTION=1
EOF

cat <<EOF

ROCm WSL stack install command completed using AMD's current ROCDXG path.

Next steps:
1. Open a new WSL shell so your updated group membership is applied.
2. Install ROCm PyTorch wheels with scripts/wsl_rocm/install_pytorch_rocm.sh.
3. Run scripts/wsl_rocm/verify_rocm.sh.

Installed components:
- ROCm userspace packages under /opt/rocm
- librocdxg built from ${ROCDXG_REPO_URL}
- DXG detection env helper at ${ROCDXG_SRC_DIR}/wsl-rocdxg.env

Windows SDK path used:
- ${WIN_SDK_ROOT}
EOF
