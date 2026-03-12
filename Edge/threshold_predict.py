import argparse
import datetime
from typing import Optional

import numpy as np
import yaml


def load_threshold_from_yaml(
    yaml_path: str,
    dataset: str,
    energy_name: str,
) -> float:
    """Load a single threshold value from a YAML file.

    The YAML structure is expected to match what em_gmm_threshold.py writes:
        {dataset: {energy_name: threshold, ...}, ...}
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict) or dataset not in data:
        raise ValueError(f"Dataset '{dataset}' not found in YAML file: {yaml_path}")

    dataset_block = data.get(dataset, {})
    if not isinstance(dataset_block, dict):
        raise ValueError(f"Invalid structure under dataset '{dataset}' in YAML file: {yaml_path}")

    if energy_name not in dataset_block:
        raise ValueError(
            f"Energy/model name '{energy_name}' not found under dataset '{dataset}' in YAML file: {yaml_path}",
        )

    threshold = float(dataset_block[energy_name])
    print(
        f"Loaded threshold from YAML '{yaml_path}' for dataset='{dataset}', "
        f"energy='{energy_name}' -> threshold={threshold}",
    )
    return threshold


def compute_binary_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Compute 0/1 predictions from scores and threshold.

    This mirrors the core decision rule in Cloud/solver_ensemble.Solver.singlemodelpred:
        pred = (scores > thresh).astype(int)
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    preds = (scores > float(threshold)).astype(int)
    return preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 0/1 anomaly predictions from scores using a threshold "
            "(directly supplied or loaded from YAML)."
        ),
    )
    parser.add_argument(
        "--score_file",
        type=str,
        required=True,
        help="Path to score/energy file (one value per line).",
    )
    parser.add_argument(
        "--output_pred",
        type=str,
        required=True,
        help="Path to save 0/1 predictions as a text file.",
    )

    # Direct threshold option
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Direct numeric threshold. If omitted, YAML-based loading is used.",
    )

    # YAML-based threshold options (for thresholds written by em_gmm_threshold.py)
    parser.add_argument(
        "--thresholds_yaml",
        type=str,
        default=None,
        help=(
            "Optional path to YAML file containing thresholds written by em_gmm_threshold.py. "
            "Used when --threshold is not provided."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key under which thresholds are stored in the YAML file.",
    )
    parser.add_argument(
        "--energy_name",
        type=str,
        default=None,
        help="Energy/model name key for this score file within the YAML file.",
    )

    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help=(
            "Optional path to ground-truth label file (0/1 per line). "
            "If provided, anomaly segment smoothing identical to Solver.singlemodelpred "
            "will be applied using these labels before saving predictions."
        ),
    )

    return parser.parse_args()


def resolve_threshold(args: argparse.Namespace) -> float:
    """Resolve threshold either directly from CLI or from YAML settings."""
    if args.threshold is not None:
        return float(args.threshold)

    if args.thresholds_yaml is None or args.dataset is None or args.energy_name is None:
        raise ValueError(
            "When --threshold is not provided, you must supply "
            "--thresholds_yaml, --dataset, and --energy_name.",
        )

    return load_threshold_from_yaml(
        yaml_path=args.thresholds_yaml,
        dataset=args.dataset,
        energy_name=args.energy_name,
    )


def main() -> None:
    args = parse_args()
    print(f"Loading scores from {args.score_file}")
    scores = np.loadtxt(args.score_file, dtype=float).reshape(-1)

    threshold = resolve_threshold(args)
    print(f"Using threshold: {threshold}")

    preds = compute_binary_predictions(scores, threshold)

    np.savetxt(args.output_pred, preds.astype(int), fmt="%d")
    print(f"Saved binary predictions to {args.output_pred}")


if __name__ == "__main__":
    main()
