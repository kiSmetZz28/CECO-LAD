#!/usr/bin/env bash
# CECO-LAD unified runner — single entry point for all pipeline operations.
#
# Usage:
#   bash run.sh COMMAND [DATASET] [OPTIONS]
#
# Commands:
#   download [DATASET] [TYPE]    Download pre-trained checkpoints from Google Drive
#   train    [DATASET]           Train BAT ensemble  (bgl|hdfs|os, default: bgl)
#   eval     [DATASET] [VOTING]  Evaluate BAT ensemble (voting: majority|consensus|at-least-one|all)
#   convert  [DATASET]           Export BAT checkpoints to quantized Q-BAT .pte files
#   infer    [DATASET]           Run full inference pipeline: edge Q-BAT → routing → cloud BAT
#   test                         Run the test suite
#   help                         Show this message
#
# Examples:
#   bash run.sh download            # Download all checkpoints (skip training)
#   bash run.sh download bgl        # Download only BGL checkpoints
#   bash run.sh download bgl bat    # Download only BGL full-precision checkpoints
#   bash run.sh train bgl
#   bash run.sh eval os majority
#   bash run.sh infer bgl
#   bash run.sh test
set -euo pipefail

COMMAND="${1:-help}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

_usage() {
    cat <<'EOF'
Usage: bash run.sh COMMAND [DATASET] [OPTIONS]

Commands:
  download [DATASET] [TYPE]    Download pre-trained checkpoints from Google Drive
                                 DATASET : bgl | hdfs | os  (default: all)
                                 TYPE    : bat | qbat        (default: both)
  train    [DATASET]           Train BAT ensemble  (bgl | hdfs | os,  default: bgl)
  eval     [DATASET] [VOTING]  Evaluate BAT ensemble  (voting: majority | consensus | at-least-one | all)
  convert  [DATASET]           Export BAT .pth checkpoints to quantized Q-BAT .pte files
  infer    [DATASET]           Full inference pipeline  (edge Q-BAT + routing + cloud BAT)
  test                         Run the test suite

Examples:
  bash run.sh download            # Download all pre-trained checkpoints
  bash run.sh download bgl        # Download BGL checkpoints only
  bash run.sh download bgl bat    # Download BGL full-precision checkpoints only
  bash run.sh train bgl
  bash run.sh eval bgl all
  bash run.sh convert bgl
  bash run.sh infer bgl
  bash run.sh test
EOF
}

case "${COMMAND}" in
    download)
        DATASET="${2:-}"
        TYPE="${3:-}"
        echo "[run] Downloading pre-trained checkpoints — dataset: '${DATASET:-all}'  type: '${TYPE:-both}'"
        ARGS=()
        [[ -n "${DATASET}" ]] && ARGS+=(--dataset "${DATASET}")
        [[ -n "${TYPE}"    ]] && ARGS+=(--type    "${TYPE}")
        python tools/download_checkpoints.py "${ARGS[@]}"
        ;;

    train)
        DATASET="${2:-bgl}"
        echo "[run] Training BAT ensemble — dataset: ${DATASET}"
        python -m training_pipeline.train --config "configs/training/${DATASET}.yaml"
        ;;

    eval)
        DATASET="${2:-bgl}"
        VOTING="${3:-all}"
        echo "[run] Evaluating BAT ensemble — dataset: ${DATASET}  voting: ${VOTING}"
        python -m training_pipeline.evaluate \
            --config "configs/training/${DATASET}.yaml" \
            --voting "${VOTING}"
        ;;

    convert)
        DATASET="${2:-bgl}"
        echo "[run] Converting BAT checkpoints → Q-BAT .pte — dataset: ${DATASET}"
        python edge_pipeline/convert.py --config "configs/training/${DATASET}.yaml" --all
        ;;

    infer)
        DATASET="${2:-bgl}"
        echo "[run] Inference pipeline — dataset: ${DATASET}"
        python -m inference_pipeline.run --config "configs/inference/${DATASET}.yaml"
        ;;

    test)
        echo "[run] Running test suite"
        python -m pytest tests/ -v
        ;;

    help|--help|-h)
        _usage
        ;;

    *)
        echo "[run] Unknown command: '${COMMAND}'" >&2
        echo "" >&2
        _usage >&2
        exit 1
        ;;
esac
