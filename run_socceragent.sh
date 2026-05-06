#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  source "${ROOT}/.env"
  set +a
fi

INPUT_FILE="${1:-challenge/test/test.json}"
OUTPUT_FILE="${2:-outputs/result.json}"
GPUS="${GPUS:-0,1,2,3}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo "[INFO] root=${ROOT}"
echo "[INFO] input=${INPUT_FILE}"
echo "[INFO] output=${OUTPUT_FILE}"
echo "[INFO] gpus=${GPUS}"
echo "[INFO] agent=${AGENT_MODEL_NAME:-}"
echo "[INFO] vision_backend=${VISION_BACKEND:-qwen}"
echo "[INFO] vlm=${VLM_MODEL_NAME:-}"
echo "[INFO] replay_backend=${REPLAY_GROUNDING_EMBED_BACKEND:-qwen}"
echo "[INFO] replay_embed=${QWEN3_VL_EMBED_MODEL:-}"

SOCCERAGENT_HOME="${ROOT}" \
CUDA_VISIBLE_DEVICES="${GPUS}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchvision}" \
python -m platform_full_version \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}"
