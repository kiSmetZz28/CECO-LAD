#!/usr/bin/env python3
"""Generate realistic demo output files for the CECO-LAD dashboard.

Run once from the project root:
    python dashboard/generate_demo.py

Creates .npy files under outputs/{bgl,hdfs,os}/ so all dashboard tabs
show populated results without needing to run the full inference pipeline.
"""
import os
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
rng  = np.random.default_rng(42)


def anomaly_segments(n: int, n_segments: int, seg_len_range=(5, 30)) -> np.ndarray:
    """Build a ground-truth array with contiguous anomaly segments."""
    gt = np.zeros(n, dtype=np.int32)
    for _ in range(n_segments):
        seg_len = rng.integers(*seg_len_range)
        start   = rng.integers(0, max(1, n - seg_len))
        gt[start:start + seg_len] = 1
    return gt


def noisy_preds(gt: np.ndarray, fp_rate=0.01, fn_rate=0.05) -> np.ndarray:
    """Simulate predictions with controlled false-positive/negative rates."""
    pred = gt.copy()
    # False negatives: miss some anomalies
    anom_idx  = np.where(gt == 1)[0]
    fn_mask   = rng.random(len(anom_idx)) < fn_rate
    pred[anom_idx[fn_mask]] = 0
    # False positives: flag some normal windows
    norm_idx  = np.where(gt == 0)[0]
    fp_mask   = rng.random(len(norm_idx)) < fp_rate
    pred[norm_idx[fp_mask]] = 1
    return pred.astype(np.int32)


def energy_matrix(gt: np.ndarray, n_models=3) -> np.ndarray:
    """Synthetic energy scores: higher near anomalies, with per-model noise."""
    n   = len(gt)
    mat = np.zeros((n, n_models), dtype=np.float32)
    base = gt.astype(np.float32) * 0.6 + 0.1   # anomaly ~0.7, normal ~0.1
    for m in range(n_models):
        noise   = rng.normal(0, 0.04, n).astype(np.float32)
        scale   = rng.uniform(0.85, 1.15)
        mat[:, m] = np.clip(base * scale + noise, 0.0, 1.0)
    return mat


def routing_indices(gt: np.ndarray, mat: np.ndarray,
                    thresholds: list, tolerance=0.1) -> np.ndarray:
    """Route windows whose energy most exceeds the per-model thresholds."""
    margin = np.mean(mat - np.array(thresholds, dtype=np.float32), axis=1)
    n_route = max(1, int(len(margin) * tolerance))
    return np.sort(np.argsort(margin)[-n_route:]).astype(np.int32)


def edge_thresholds(mat: np.ndarray, gt: np.ndarray) -> list:
    """Simple percentile threshold per model."""
    normal_energy = mat[gt == 0]
    return [float(np.percentile(normal_energy[:, m], 95)) for m in range(mat.shape[1])]


def save_dataset(ds: str, n: int, n_segments: int,
                 seg_range=(10, 60), fp=0.008, fn=0.04, tol=0.1):
    out_dir = ROOT / "outputs" / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    gt         = anomaly_segments(n, n_segments, seg_range)
    edge_pred  = noisy_preds(gt, fp_rate=fp, fn_rate=fn)
    mat        = energy_matrix(gt)
    threshs    = edge_thresholds(mat, gt)
    ri         = routing_indices(gt, mat, threshs, tolerance=tol)
    # Hybrid: cloud fixes most of the edge errors on routed windows
    hybrid     = edge_pred.copy()
    for idx in ri:
        hybrid[idx] = gt[idx]           # cloud gets it right on routed windows
    hybrid = hybrid.astype(np.int32)

    np.save(out_dir / "ground_truth.npy",   gt)
    np.save(out_dir / "edge_preds.npy",     edge_pred)
    np.save(out_dir / "hybrid_preds.npy",   hybrid)
    np.save(out_dir / "energy_matrix.npy",  mat)
    np.save(out_dir / "routed_indices.npy", ri)

    # Edge thresholds YAML
    models = [{"name": f"qbat_model_{i+1}", "threshold": t}
              for i, t in enumerate(threshs)]
    thr_path = ROOT / "outputs" / ds / "thresholds_edge.yaml"
    with open(thr_path, "w") as f:
        yaml.dump({"models": models}, f)

    # Quick stats
    tp = int(((gt == 1) & (hybrid == 1)).sum())
    fp_cnt = int(((gt == 0) & (hybrid == 1)).sum())
    fn_cnt = int(((gt == 1) & (hybrid == 0)).sum())
    prec = tp / (tp + fp_cnt) if (tp + fp_cnt) else 0
    rec  = tp / (tp + fn_cnt) if (tp + fn_cnt) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    anom_pct = gt.mean() * 100
    print(f"  [{ds:4s}] N={n:>6,}  anomaly={anom_pct:.1f}%  "
          f"routed={len(ri)} ({tol*100:.0f}%)  "
          f"hybrid F1={f1:.4f}")


def main():
    print("Generating demo output data…\n")
    #            ds      N        segs   seg_range      fp      fn    tol
    save_dataset("os",   1386,    12,    (5,  30),   0.010, 0.05, 0.10)
    save_dataset("bgl",  5000,    40,    (20, 120),  0.005, 0.03, 0.10)
    save_dataset("hdfs", 3000,    25,    (10, 60),   0.008, 0.04, 0.10)
    print("\nDone. Start the dashboard with:\n")
    print("    python dashboard/app.py\n")
    print("Then open  http://localhost:8765")


if __name__ == "__main__":
    main()
