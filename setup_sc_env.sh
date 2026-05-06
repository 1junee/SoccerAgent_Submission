#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-socceragent-repro}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[INFO] Conda env ${ENV_NAME} already exists."
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e pipeline/toolbox/utils/GroundingDINO

echo "[DONE] Environment ${ENV_NAME} is ready."
