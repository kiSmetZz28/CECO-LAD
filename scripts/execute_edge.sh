#!/usr/bin/env bash
set -euo pipefail

# Simple entrypoint to run the Edge pipeline without having to remember
# individual .sh scripts in Edge/edge_scripts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGE_DIR="${ROOT_DIR}/Edge"

usage() {
	cat <<EOF
Usage: bash scripts/execute_edge.sh [OPTIONS]

Options:
	-d, --dataset DATASET   Dataset: os | bgl | hdfs (default: os)
	-s, --stage STAGE       Stage to run (default: all)
													STAGE for os: scores | thresh | predict | routing | hybrid | all
													STAGE for bgl/hdfs: scores | all
	-h, --help              Show this help message

Examples:
	# Run full OpenStack (os) edge pipeline (ExecuTorch scores + thresholds + predict + routing + hybrid)
	bash scripts/execute_edge.sh -d os -s all

	# Only run ExecuTorch scoring for BGL (three models in parallel)
	bash scripts/execute_edge.sh -d bgl -s scores

	# Only run routing + hybrid evaluation for OpenStack (assumes scores, thresholds, predictions and cloud outputs exist)
	bash scripts/execute_edge.sh -d os -s routing
	bash scripts/execute_edge.sh -d os -s hybrid
EOF
}

DATASET="os"
STAGE="all"

while [[ $# -gt 0 ]]; do
	case "$1" in
		-d|--dataset)
			DATASET="$2"; shift 2 ;;
		-s|--stage)
			STAGE="$2"; shift 2 ;;
		-h|--help)
			usage; exit 0 ;;
		*)
			echo "[execute_edge] Unknown argument: $1" >&2
			usage
			exit 1 ;;
	esac
done

cd "${EDGE_DIR}"

run_os() {
	case "${STAGE}" in
		scores)
			echo "[execute_edge] Running OS ExecuTorch scoring (parallel_execute_os.sh)..."
			bash edge_scripts/parallel_execute_os.sh
			;;
		thresh)
			echo "[execute_edge] Computing OS thresholds (thresh.sh)..."
			bash edge_scripts/thresh.sh
			;;
		predict)
			echo "[execute_edge] Generating OS edge predictions (predict.sh)..."
			bash edge_scripts/predict.sh
			;;
		routing)
			echo "[execute_edge] Selecting OS routed samples (routing.sh)..."
			bash edge_scripts/routing.sh
			;;
		hybrid)
			echo "[execute_edge] Evaluating OS edge+cloud hybrid (hybrid.sh)..."
			bash edge_scripts/hybrid.sh
			;;
		all)
			echo "[execute_edge] Running full OS edge pipeline: scores -> thresh -> predict -> routing -> hybrid"
			bash edge_scripts/parallel_execute_os.sh
			bash edge_scripts/thresh.sh
			bash edge_scripts/predict.sh
			bash edge_scripts/routing.sh
			bash edge_scripts/hybrid.sh
			;;
		*)
			echo "[execute_edge] Unsupported stage for os: ${STAGE}" >&2
			usage
			exit 1 ;;
	esac
}

run_bgl() {
	case "${STAGE}" in
		scores|all)
			echo "[execute_edge] Running BGL ExecuTorch scoring (parallel_execute_bgl.sh)..."
			bash edge_scripts/parallel_execute_bgl.sh
			;;
		*)
			echo "[execute_edge] For bgl, only 'scores' (or 'all') is currently supported."
			usage
			exit 1 ;;
	esac
}

run_hdfs() {
	case "${STAGE}" in
		scores|all)
			echo "[execute_edge] Running HDFS ExecuTorch scoring (parallel_execute_hdfs.sh)..."
			bash edge_scripts/parallel_execute_hdfs.sh
			;;
		*)
			echo "[execute_edge] For hdfs, only 'scores' (or 'all') is currently supported."
			usage
			exit 1 ;;
	esac
}

case "${DATASET}" in
	os)
		run_os ;;
	bgl)
		run_bgl ;;
	hdfs)
		run_hdfs ;;
	*)
		echo "[execute_edge] Unknown dataset: ${DATASET} (expected: os | bgl | hdfs)" >&2
		usage
		exit 1 ;;
esac

