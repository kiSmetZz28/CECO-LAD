#!/usr/bin/env python3
"""Cloud inference phase — runs in the cloud conda env (e.g. 'hybrid').

Reads intermediate files saved by the edge phase and runs the BAT ensemble.
Called automatically by the dashboard when using split environments.

Usage (from project root):
    python dashboard/cloud_runner.py --config configs/inference/os.yaml
"""
import argparse
import logging
import os
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _point_adjust(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Fill entire GT anomaly segments once any window in the segment is detected."""
    gt   = gt.astype(int)
    pred = pred.astype(int).copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="CECO-LAD cloud inference phase.")
    parser.add_argument("--config", required=True, help="Inference YAML config path.")
    args = parser.parse_args()

    os.chdir(str(ROOT))

    from ceco_core.utils.config import load_config, setup_logging
    from ceco_core.utils.metrics import evaluate
    from inference_pipeline import cloud_expert

    setup_logging("cloud")
    cfg      = load_config(args.config)
    dataset  = cfg["dataset"]
    out_base = cfg.get("output_dir", f"outputs/{dataset.lower()}")

    def _load(fname: str) -> np.ndarray:
        path = os.path.join(out_base, fname)
        if not os.path.exists(path):
            logging.error(
                "Required file not found: %s\n"
                "Run the edge phase first (Phase 1/2).", path
            )
            sys.exit(1)
        return np.load(path)

    routed_windows  = _load("routed_windows.npy")
    ground_truth   = _load("ground_truth.npy")
    routed_indices = _load("routed_indices.npy")
    edge_preds_raw = _load("edge_preds_raw.npy")   # raw (not point-adjusted)

    cloud_cfg = cfg.get("cloud")
    if not cloud_cfg:
        logging.info("No 'cloud' section in config — nothing to do.")
        return

    if len(routed_windows) == 0:
        logging.info("No routed windows — skipping cloud inference.")
        return

    cloud_cfg.setdefault("dataset", dataset)
    cloud_cfg.setdefault("win_size", cfg["win_size"])
    cloud_cfg.setdefault("input_c", cfg["input_c"])

    logging.info("=== Cloud BAT Inference ===")
    cloud_preds = cloud_expert.run(routed_windows, cloud_cfg)
    np.save(os.path.join(out_base, "cloud_preds.npy"), cloud_preds)

    logging.info("=== Stage 4: Hybrid Evaluation ===")
    logging.info(
        "Cloud preds: %d anomalous / %d routed windows.",
        int(cloud_preds.sum()), len(cloud_preds),
    )

    # routed_indices are per-line; cloud_preds are per-window.
    # Re-derive the window index order (same sorted order as run.py) and map each
    # selected line to its window's cloud prediction, then update those lines only.
    win_size = cfg.get("win_size", 100)
    routed_window_indices = sorted(set(int(i) // win_size for i in routed_indices))
    window_pred_map = {w: cloud_preds[j] for j, w in enumerate(routed_window_indices)}

    hybrid_preds = edge_preds_raw.copy()
    line_idx = np.array(routed_indices, dtype=int)
    line_cloud_preds = np.array([window_pred_map[int(i) // win_size] for i in routed_indices], dtype=int)
    hybrid_preds[line_idx] = line_cloud_preds

    logging.info(
        "Hybrid raw preds: %d anomalous / %d total timesteps.",
        int(hybrid_preds.sum()), len(hybrid_preds),
    )
    hybrid_adj = _point_adjust(ground_truth, hybrid_preds)
    np.save(os.path.join(out_base, "hybrid_preds.npy"), hybrid_adj)
    evaluate(ground_truth, hybrid_adj, prefix="Hybrid")

    logging.info("=== Cloud inference complete. Outputs in '%s' ===", out_base)


if __name__ == "__main__":
    main()
