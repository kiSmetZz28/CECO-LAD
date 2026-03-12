import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def evaluate(gt: np.ndarray, pred: np.ndarray, prefix: str = "") -> None:
    preds = pred.astype(int).copy()
    gt = gt.astype(int)

    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and preds[i] == 1 and not anomaly_state:
            anomaly_state = True
            # back-fill within this anomaly segment
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                if preds[j] == 0:
                    preds[j] = 1
            # forward-fill within this anomaly segment
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                if preds[j] == 0:
                    preds[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            preds[i] = 1

    accuracy = accuracy_score(gt, preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        gt, preds, average="binary"
    )
    label = f"[{prefix}] " if prefix else ""
    print(
        f"{label}Accuracy : {accuracy:0.4f}, Precision : {precision:0.4f}, "
        f"Recall : {recall:0.4f}, F-score : {f_score:0.4f}"
    )


def compute_edge_ensemble(edge_pred_files, label_file):
    """Compute edge ensemble (majority vote over given prediction files) and evaluate.

    Returns
    -------
    gt : np.ndarray
        Ground-truth labels.
    """
    # load predictions
    preds_list = [np.loadtxt(p, dtype=int).reshape(-1) for p in edge_pred_files]
    gt = np.loadtxt(label_file, dtype=int).reshape(-1)

    # sanity check lengths
    lengths = [len(p) for p in preds_list]
    if not all(l == len(gt) for l in lengths):
        raise ValueError(
            f"Lengths mismatch: preds lengths {lengths}, labels length {len(gt)}"
        )

    # majority voting across all given prediction files
    stacked = np.vstack(preds_list)  # [n_models, n_samples]
    votes = stacked.sum(axis=0)
    edge_raw = (votes >= (stacked.shape[0] // 2 + 1)).astype(int)

    print("Edge ensemble pred shape:", edge_raw.shape)
    print("GT shape:", gt.shape)

    evaluate(gt, edge_raw, prefix="Edge")

    return edge_raw, gt


def compute_hybrid(edge_raw: np.ndarray, cloud_pred_file: str, indices_file: str, gt: np.ndarray):
    """Merge edge and cloud predictions using indices, then evaluate.

    Parameters
    ----------
    edge_raw : np.ndarray
        Raw edge ensemble predictions.
    cloud_pred_file : str
        Path to cloud predictions for selected indices.
    indices_file : str
        Path to indices file produced by mahalanobis_routing.py.
    gt : np.ndarray
        Ground-truth labels (full length).
    """
    cloud_pred = np.loadtxt(cloud_pred_file, dtype=int).reshape(-1)
    indices = np.loadtxt(indices_file, dtype=int).reshape(-1)

    n_cloud = cloud_pred.shape[0]
    n_idx = indices.shape[0]

    # Due to windowing on the cloud side, we may have fewer cloud
    # predictions than routed indices. In that case, only merge the
    # first `n_cloud` indices. If we somehow have more cloud preds
    # than indices, that's an error.
    if n_cloud > n_idx:
        raise ValueError(
            f"cloud_pred length {n_cloud} > indices length {n_idx}. "
            "Check selected data / windowing pipeline."
        )
    if n_cloud < n_idx:
        indices = indices[:n_cloud]

    if np.any(indices < 0) or np.any(indices >= edge_raw.shape[0]):
        raise ValueError("Some indices are out of range of edge prediction length.")

    hybrid_raw = edge_raw.copy()
    hybrid_raw[indices] = cloud_pred

    evaluate(gt, hybrid_raw, prefix="Hybrid")

    return hybrid_raw


def main():
    parser = argparse.ArgumentParser(
        description="Compute edge ensemble and hybrid (edge+cloud) evaluation results."
    )
    parser.add_argument(
        "--edge_preds",
        type=str,
        nargs="+",
        required=True,
        help="Paths to edge prediction files to ensemble (e.g., three Q-BAT models).",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Path to ground-truth label file.",
    )
    parser.add_argument(
        "--cloud_pred",
        type=str,
        default=None,
        help="Optional: path to cloud predictions for selected indices.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Optional: path to indices file (for hybrid).",
    )

    args = parser.parse_args()

    # Edge ensemble
    edge_raw, gt = compute_edge_ensemble(args.edge_preds, args.label)

    # Hybrid (if cloud + indices are provided)
    if args.cloud_pred is not None and args.indices is not None:
        compute_hybrid(edge_raw, args.cloud_pred, args.indices, gt)


if __name__ == "__main__":
    main()
