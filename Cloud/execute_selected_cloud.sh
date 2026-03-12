#!/usr/bin/env bash
set -euo pipefail

# Run cloud-side ensemble only on the selected samples
# routed from the Edge (os_indices/os_selected).
#
# Usage (from repo root or Cloud/):
#   bash Cloud/execute_selected_cloud.sh
#
# It assumes that:
#   - The Edge pipeline has already produced:
#       Edge/prediction_results/os_selected.txt
#   - You have a valid OS ensemble config and thresholds yaml.

# Cloud project root (directory containing this script)
CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Name of the conda environment to use on Cloud.
# Change this if your env name is different.
CLOUD_CONDA_ENV="ceco-lad-cloud"

CONFIG="model_config/bat_config/ensemble_test_os_config.yaml"
THRESHOLDS_YAML="model_config/threshold_config/ensemble_config_os.yaml"
SELECTED_DATA="selected_samples/os_selected.txt"
OUTPUT_PRED="selected_samples/cloud_selected_ensemble_preds.txt"

cd "${CLOUD_DIR}"

# If a conda env name is provided, try to activate it so that
# numpy and other Python deps are available in non-interactive SSH.
if [[ -n "${CLOUD_CONDA_ENV}" ]]; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    . "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    . "${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi
  if command -v conda >/dev/null 2>&1; then
    conda activate "${CLOUD_CONDA_ENV}"
  else
    echo "[execute_selected_cloud] WARNING: conda not found; running without env activation" >&2
  fi
fi

echo "[execute_selected_cloud] Using config: ${CONFIG}"
echo "[execute_selected_cloud] Using thresholds: ${THRESHOLDS_YAML}"
echo "[execute_selected_cloud] Selected data: ${SELECTED_DATA}"
echo "[execute_selected_cloud] Output predictions: ${OUTPUT_PRED}"

python predict_selected_subset.py \
  --config "${CONFIG}" \
  --thresholds_yaml "${THRESHOLDS_YAML}" \
  --selected_data "${SELECTED_DATA}" \
  --output_pred "${OUTPUT_PRED}"

echo "[execute_selected_cloud] Done. Saved to ${OUTPUT_PRED}"
