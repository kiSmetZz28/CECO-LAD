import argparse
import datetime
import os

import numpy as np
import yaml
from sklearn.mixture import GaussianMixture


def set_thresh_em(energy: np.ndarray,
                  n_components: int = 7,
                  covariance_type: str = "tied",
                  max_iter: int = 100,
                  init_params: str = "k-means++",
                  n_init: int = 10,
                  random_state: int = 42) -> np.ndarray:
    """Run EM-GMM on 1D energy values and return cluster labels.

    This mirrors the logic used in Cloud/solver_ensemble.py.
    """
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params=init_params,
        n_init=n_init,
        random_state=random_state,
    ).fit(energy)
    pred = gm.predict(energy)
    return pred


def get_anomaly_ratio(em_pred: np.ndarray):
    """Return sorted list of (label, percentage) pairs by descending percentage.

    The largest cluster is treated as normal in downstream thresholding,
    following Cloud/solver_ensemble.py.
    """
    unique, counts = np.unique(em_pred, return_counts=True)
    total = len(em_pred)

    label_percentages = {label: (count / total) * 100 for label, count in zip(unique, counts)}
    sorted_percentages = sorted(label_percentages.items(), key=lambda x: x[1], reverse=True)

    print(f"Label counts: {dict(zip(unique, counts))}")
    for label, percentage in sorted_percentages:
        print(f"Label {label}: {percentage:.6f}%")

    return sorted_percentages


def compute_threshold_from_energy(train_energy: np.ndarray,
                                  n_components: int = 7,
                                  covariance_type: str = "tied",
                                  max_iter: int = 100,
                                  init_params: str = "k-means++",
                                  n_init: int = 10) -> tuple[float, float, list[tuple[int, float]]]:
    """Compute anomaly threshold from training energy using EM-GMM.

    Parameters
    ----------
    train_energy : np.ndarray
        1D array of training energy values.

    Returns
    -------
    thresh : float
        Energy threshold (percentile) corresponding to the dominant cluster
        treated as normal.
    normal_ratio : float
        Percentage of samples in the dominant (normal) cluster.
    cluster_percentages : list[(label, percentage)]
        Cluster label percentages sorted by descending percentage.
    """
    train_energy = np.asarray(train_energy, dtype=float).reshape(-1)

    em_pred = set_thresh_em(
        train_energy.reshape(-1, 1),
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params=init_params,
        n_init=n_init,
    )

    cluster_percentages = get_anomaly_ratio(em_pred)
    normal_ratio = cluster_percentages[0][1]
    print(f"Normal data ratio: {normal_ratio:.6f}%")
    anomaly_ratio = 100.0 - normal_ratio
    print(f"Abnormal data ratio: {anomaly_ratio:.6f}%")

    # Use the normal-ratio percentile of train_energy as threshold,
    # consistent with Cloud/solver_ensemble.py.
    thresh = float(np.percentile(train_energy, normal_ratio))
    print(f"Threshold: {thresh:.10f}")

    return thresh, normal_ratio, cluster_percentages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute EM-GMM threshold from training energy.")
    parser.add_argument(
        "--train_energy",
        type=str,
        required=True,
        help="Path to training energy file (one value per line).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., bgl, hdfs, os) used as a key in the YAML file.",
    )
    parser.add_argument(
        "--energy_name",
        type=str,
        default=None,
        help=(
            "Optional logical name for this energy source; if omitted, "
            "the basename of --train_energy (without extension) is used as the key in YAML."
        ),
    )
    parser.add_argument(
        "--output_threshold",
        type=str,
        default=None,
        help="Optional path to save the computed threshold as a single-line text file.",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=7,
        help="Number of mixture components for GaussianMixture (default: 7).",
    )
    parser.add_argument(
        "--covariance_type",
        type=str,
        default="tied",
        choices=["full", "tied", "diag", "spherical"],
        help="Covariance type for GaussianMixture (default: tied).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of EM iterations (default: 100).",
    )
    parser.add_argument(
        "--init_params",
        type=str,
        default="k-means++",
        help="Initialization method for GaussianMixture (default: k-means++).",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=10,
        help="Number of initializations to perform (default: 10).",
    )
    parser.add_argument(
        "--yaml_output",
        type=str,
        default=None,
        help=(
            "Optional path to a YAML file where thresholds are stored. "
            "If not provided, a file named 'thresholds_<dataset>.yaml' in the current directory is used."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading training energy from {args.train_energy}")
    train_energy = np.loadtxt(args.train_energy, dtype=float).reshape(-1)

    thresh, normal_ratio, cluster_percentages = compute_threshold_from_energy(
        train_energy,
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        max_iter=args.max_iter,
        init_params=args.init_params,
        n_init=args.n_init,
    )

    print(f"Final threshold: {thresh:.10f}")

    # Optionally save raw threshold value
    if args.output_threshold is not None:
        np.savetxt(args.output_threshold, np.array([thresh], dtype=float), fmt="%.10f")
        print(f"Saved threshold value to {args.output_threshold}")

    # Save threshold together with the file name in a YAML file.
    dataset = args.dataset
    energy_name = args.energy_name
    if energy_name is None:
        energy_name = os.path.splitext(os.path.basename(args.train_energy))[0]

    if args.yaml_output is not None:
        yaml_path = args.yaml_output
    else:
        yaml_path = f"thresholds_{dataset}.yaml"

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            try:
                yaml_data = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                yaml_data = {}
    else:
        yaml_data = {}

    if not isinstance(yaml_data, dict):
        yaml_data = {}

    # Structure: {dataset: {energy_name: threshold, ...}, ...}
    dataset_block = yaml_data.get(dataset, {})
    if not isinstance(dataset_block, dict):
        dataset_block = {}

    dataset_block[energy_name] = float(thresh)
    yaml_data[dataset] = dataset_block

    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=True)
    print(
        f"Recorded threshold for dataset='{dataset}', energy='{energy_name}' "
        f"into YAML file: {yaml_path}"
    )

    return thresh


if __name__ == "__main__":
    main()
