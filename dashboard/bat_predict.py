#!/usr/bin/env python3
"""BAT ensemble prediction for one pre-scaled session window.

Called by dashboard/app.py via CLOUD_PYTHON (the hybrid conda env):

    python dashboard/bat_predict.py \\
        --input   /tmp/window_scaled.npy \\
        --config  configs/inference/os.yaml \\
        --max-models 9

Input:  .npy file containing a float32 array of shape [n_events, input_c]
        (the StandardScaler-normalised context window matrix produced by
        the dashboard's in-memory scaler in the ceco-lad env).

Output: single JSON object printed to stdout.
        Errors are also returned as {"error": "..."} on stdout so the
        caller can always parse the output with json.loads().
"""
import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-session BAT prediction.")
    parser.add_argument("--input",      required=True,
                        help="Path to .npy file: scaled [n_events, input_c] float32 array")
    parser.add_argument("--config",     required=True,
                        help="Inference YAML config path (relative to project root)")
    parser.add_argument("--max-models", type=int, default=9,
                        help="Maximum number of BAT checkpoints to load (for speed)")
    args = parser.parse_args()

    t0 = time.time()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    if not cfg_path.exists():
        _out({"error": f"Config not found: {args.config}"})
        return

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cloud_cfg = cfg.get("cloud")
    if not cloud_cfg:
        _out({"error": "No 'cloud' section in config"})
        return

    win_size = cfg.get("win_size", 100)
    input_c  = cfg.get("input_c", 10)
    output_c = cloud_cfg.get("output_c", input_c)

    # ── Load pre-scaled array, build window tensor ────────────────────────────
    if not os.path.exists(args.input):
        _out({"error": f"Input file not found: {args.input}"})
        return

    arr      = np.load(args.input).astype(np.float32)   # [n_events, input_c]
    n_events = len(arr)
    padded   = n_events < win_size

    if padded:
        pad = np.zeros((win_size - n_events, input_c), dtype=np.float32)
        arr = np.vstack([arr, pad])
    arr = arr[:win_size]                                 # [win_size, input_c]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(arr[np.newaxis]).float().to(device)  # [1, win_size, input_c]

    # ── Load thresholds ───────────────────────────────────────────────────────
    thresh_rel  = cloud_cfg.get("thresholds_yaml", "")
    thresh_path = ROOT / thresh_rel
    if not thresh_path.exists():
        _out({"error": f"Threshold file not found: {thresh_rel} — run 'eval' first"})
        return

    try:
        from inference_pipeline.cloud_expert import _load_thresholds
        thresholds_dict = _load_thresholds(str(thresh_path))
    except Exception as exc:
        _out({"error": f"Could not load thresholds: {exc}"})
        return

    # ── Load BAT models and score ─────────────────────────────────────────────
    from ceco_core.models.EMAT import EMAT
    from ceco_core.utils.energy import compute_energy_batch

    search_keys  = ["num_epochs", "k", "e_layer_num", "batch_size"]
    combinations = list(product(*[cloud_cfg[k] for k in search_keys]))
    model_dir    = ROOT / cloud_cfg.get("model_save_path", "")
    dataset_name = cloud_cfg.get("dataset", cfg.get("dataset", ""))

    model_results = []
    for combo in combinations:
        if len(model_results) >= args.max_models:
            break
        epochs, k_val, layers, bsz = combo
        mname = f"{dataset_name}_e{epochs}_k{k_val}_l{layers}_b{bsz}"
        ckpt  = model_dir / f"{mname}_checkpoint.pth"

        if not ckpt.exists() or mname not in thresholds_dict:
            continue

        thresh = thresholds_dict[mname]
        try:
            mdl = EMAT(win_size=win_size, enc_in=input_c, c_out=output_c, e_layers=layers)
            # strict=False: distances attribute presence varies across attn.py versions;
            # it is always re-computed correctly in __init__ so skipping is safe.
            mdl.load_state_dict(
                torch.load(str(ckpt), map_location=device, weights_only=True),
                strict=False,
            )
            mdl.to(device).eval()

            energy = compute_energy_batch(mdl, x, win_size)
            mean_e = float(energy.mean())
            model_results.append({
                "model":     mname,
                "energy":    round(mean_e, 6),
                "threshold": round(thresh, 6),
                "vote":      1 if mean_e > thresh else 0,
            })
        except Exception as exc:
            print(f"Skipping {mname}: {exc}", file=sys.stderr)

    if not model_results:
        _out({"error": "No BAT checkpoints found — download or train models first"})
        return

    n      = len(model_results)
    n_anom = sum(r["vote"] for r in model_results)
    final  = 1 if n_anom > n / 2 else 0

    _out({
        "prediction":      final,
        "label":           "ANOMALY" if final else "NORMAL",
        "n_models":        n,
        "n_anomaly_votes": n_anom,
        "n_normal_votes":  n - n_anom,
        "model_results":   model_results,
        "n_events":        n_events,
        "win_size":        win_size,
        "padded":          padded,
        "elapsed_s":       round(time.time() - t0, 2),
    })


def _out(obj: dict) -> None:
    """Print JSON to stdout and flush so the parent process reads it immediately."""
    print(json.dumps(obj), flush=True)


if __name__ == "__main__":
    main()
