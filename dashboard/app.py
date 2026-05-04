#!/usr/bin/env python3
"""CECO-LAD Web Dashboard — FastAPI backend."""
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
# Make dashboard/ importable so db.py / ingest.py can be imported directly
sys.path.insert(0, str(Path(__file__).parent))
# Make project root importable so ceco_core / inference_pipeline are accessible
sys.path.insert(0, str(ROOT))
import db as _db

app = FastAPI(title="CECO-LAD Dashboard")

# Set DEMO_MODE=1 to disable pipeline execution and live inference (for hosted demos).
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

# ── Per-dataset StandardScaler (matches loaders.py: Preprocessor.text → scaler) ─
_CTX_LEN = 10  # matches Preprocessor.sequence() context_length=10 (all preceding events)

# txt file root — same files the loaders use
_DATA_DIR = ROOT / "data"
_TXT_PATHS: dict[str, dict[str, list[Path]]] = {
    "bgl": {
        "train": [_DATA_DIR / "BGL" / "bgl_train.txt"],
        "test":  [_DATA_DIR / "BGL" / "bgl_test_normal.txt",
                  _DATA_DIR / "BGL" / "bgl_test_abnormal.txt"],
    },
    "hdfs": {
        "train": [_DATA_DIR / "HDFS" / "hdfs_train.txt"],
        "test":  [_DATA_DIR / "HDFS" / "hdfs_test_normal.txt",
                  _DATA_DIR / "HDFS" / "hdfs_test_abnormal.txt"],
    },
    "os": {
        "train": [_DATA_DIR / "OpenStack" / "train.txt"],
        "test":  [_DATA_DIR / "OpenStack" / "test_normal.txt",
                  _DATA_DIR / "OpenStack" / "test_abnormal.txt"],
    },
}

# scaler state per dataset
_scalers: dict[str, Optional[dict]] = {"bgl": None, "hdfs": None, "os": None}


def _read_txt_lines(paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for p in paths:
        if p.exists():
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        lines.append(s)
    return lines


def _sync_fit_scaler(dataset: str) -> None:
    """Fit StandardScaler for one dataset from its training txt file.
    Uses partial_fit to stay memory-efficient (BGL has ~8 M context rows)."""
    try:
        from sklearn.preprocessing import StandardScaler
        paths  = _TXT_PATHS[dataset]["train"]
        lines  = _read_txt_lines(paths)
        if not lines:
            return

        all_ids: set[int] = set()
        event_lists: list[list[int]] = []
        for line in lines:
            try:
                ids = list(map(int, line.split()))
                event_lists.append(ids)
                all_ids.update(ids)
            except ValueError:
                continue
        if not all_ids:
            return

        sorted_ids   = sorted(all_ids)
        event_map    = {v: i for i, v in enumerate(sorted_ids)}
        no_event_idx = len(sorted_ids)   # same as Preprocessor: NO_EVENT = last index

        BATCH = 50_000
        scaler = StandardScaler()
        batch: list[list[float]] = []

        for events in event_lists:
            mapped = [event_map.get(e, no_event_idx) for e in events]
            for i in range(len(mapped)):
                row = [float(mapped[i - back]) if (i - back) >= 0 else float(no_event_idx)
                       for back in range(_CTX_LEN, 0, -1)]
                batch.append(row)
                if len(batch) >= BATCH:
                    scaler.partial_fit(np.array(batch, dtype=np.float64))
                    batch = []
        if batch:
            scaler.partial_fit(np.array(batch, dtype=np.float64))

        _scalers[dataset] = {
            "ready":         True,
            "scaler":        scaler,
            "event_map":     event_map,
            "no_event_idx":  no_event_idx,
        }
        print(f"[scaler] {dataset}: fitted on {sum(len(e) for e in event_lists):,} events")
    except Exception as exc:
        print(f"[scaler] {dataset}: failed — {exc}")
        _scalers[dataset] = {"ready": False}


def _scale_content(dataset: str, content: str) -> Optional[list]:
    """Scale one session/window's event sequence → list of float rows."""
    st = _scalers.get(dataset)
    if not st or not st.get("ready") or not content.strip():
        return None
    try:
        event_map    = st["event_map"]
        no_event_idx = st["no_event_idx"]
        scaler       = st["scaler"]
        events_raw   = list(map(int, content.split()))
        mapped       = [event_map.get(e, no_event_idx) for e in events_raw]
        rows = []
        for i in range(len(mapped)):
            row = [float(mapped[i - back]) if (i - back) >= 0 else float(no_event_idx)
                   for back in range(_CTX_LEN, 0, -1)]
            rows.append(row)
        X_sc = scaler.transform(np.array(rows, dtype=np.float64))
        return [[round(float(v), 3) for v in r] for r in X_sc]
    except Exception:
        return None


# ── Process state ─────────────────────────────────────────────────────────────
_proc: Optional[asyncio.subprocess.Process] = None
_log_buf: list[str] = []          # buffered lines (survives page refresh)
_buf_lock = asyncio.Lock()
_last_run_start: Optional[float] = None   # epoch seconds when last run was launched
_last_run_ok: bool = True                 # True = last run completed successfully (or no run yet)


# ── Models ────────────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    command: str                   # train | eval | convert | infer | download
    dataset: str = "bgl"          # bgl | hdfs | os
    voting: str = "majority"
    routing_tolerance: float = 0.1
    routing_distance: str = "ma"


# ── Single-session prediction ─────────────────────────────────────────────────
# Edge: first 3 BAT combos (e3_k1_l3_b32/64/96) act as the Q-BAT proxy.
# Cloud: all 81 BAT combos — full ensemble, matches the batch pipeline.
_N_EDGE_MODELS  = 3
_N_CLOUD_MODELS = 999

# ── In-process BAT model cache ────────────────────────────────────────────────
# Models are loaded from disk once at startup and kept in RAM (~3.5 GB per dataset).
# Subsequent single-window predictions use the cache → ~2-3s instead of 78s.
_bat_cache: dict[str, dict] = {}        # dataset → {mname: {"model": EMAT, "threshold": float}}
_bat_cache_ready: dict[str, bool] = {}  # dataset → bool
_bat_cache_lock  = threading.Lock()


def _preload_bat_models(dataset: str) -> None:
    """Load all BAT models for one dataset into RAM. Called at startup in a bg thread."""
    cfg_path = ROOT / "configs" / "inference" / f"{dataset}.yaml"
    if not cfg_path.exists():
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cloud_cfg = cfg.get("cloud")
    if not cloud_cfg:
        return

    ckpt_dir     = ROOT / cloud_cfg["model_save_path"]
    thresh_path  = ROOT / cloud_cfg["thresholds_yaml"]
    win_size     = cfg.get("win_size", 100)
    input_c      = cfg.get("input_c", 10)
    dataset_name = cloud_cfg.get("dataset", cfg.get("dataset", ""))

    if not ckpt_dir.exists() or not thresh_path.exists():
        return

    try:
        from ceco_core.models.EMAT import EMAT
        from inference_pipeline.cloud_expert import _load_thresholds
        from itertools import product as iproduct
        import torch as _torch

        thresholds = _load_thresholds(str(thresh_path))
        try:
            _cuda = _torch.cuda.is_available()
        except Exception:
            _cuda = False
        device = _torch.device("cuda:0" if _cuda else "cpu")
        keys   = ["num_epochs", "k", "e_layer_num", "batch_size"]
        combos = list(iproduct(*[cloud_cfg[k] for k in keys]))

        _torch.set_num_threads(2)

        print(f"[cache:{dataset}] Loading {len(combos)} BAT models into RAM…", flush=True)
        loaded = 0
        ds_cache: dict = {}
        for ep, k, layers, bsz in combos:
            mname = f"{dataset_name}_e{ep}_k{k}_l{layers}_b{bsz}"
            ckpt  = ckpt_dir / f"{mname}_checkpoint.pth"
            if not ckpt.exists() or mname not in thresholds:
                continue
            try:
                mdl = EMAT(win_size=win_size, enc_in=input_c,
                           c_out=input_c, e_layers=layers)
                mdl.load_state_dict(
                    _torch.load(str(ckpt), map_location=device,
                                weights_only=True), strict=False)
                mdl.to(device).eval()
                ds_cache[mname] = {"model": mdl, "threshold": thresholds[mname]}
                loaded += 1
            except Exception as e:
                print(f"[cache:{dataset}] Skip {mname}: {e}", flush=True)

        with _bat_cache_lock:
            _bat_cache[dataset] = ds_cache
            _bat_cache_ready[dataset] = loaded > 0
        print(f"[cache:{dataset}] {loaded} BAT models ready in RAM.", flush=True)
    except Exception as exc:
        print(f"[cache:{dataset}] Model preload failed: {exc}", flush=True)


def _predict_from_cache(arr: "np.ndarray", win_size: int,
                        dataset: str, start: int, max_models: int) -> list:
    """Run inference using in-RAM models for the given dataset — no disk I/O."""
    from ceco_core.utils.energy import compute_energy_batch
    import torch as _torch
    import concurrent.futures

    try:
        _cuda = _torch.cuda.is_available()
    except Exception:
        _cuda = False
    device = _torch.device("cuda:0" if _cuda else "cpu")
    x = _torch.from_numpy(arr[:win_size] if len(arr) >= win_size
                          else np.vstack([arr,
                                np.zeros((win_size - len(arr), arr.shape[1]),
                                         dtype=np.float32)])).float().unsqueeze(0).to(device)

    with _bat_cache_lock:
        entries = list(_bat_cache.get(dataset, {}).items())[start : start + max_models]

    def _infer(item):
        mname, v = item
        try:
            energy = compute_energy_batch(v["model"], x, win_size)
            mean_e = float(energy.mean())
            return {"model": mname, "energy": round(mean_e, 6),
                    "threshold": round(v["threshold"], 6),
                    "vote": 1 if mean_e > v["threshold"] else 0}
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        return [r for r in ex.map(_infer, entries) if r is not None]


def _run_bat_subprocess(
    tmp_path: str,
    cfg_path: "Path",
    start: int,
    max_models: int,
    timeout: int = 300,
) -> dict:
    """Call bat_predict.py and return the parsed JSON result."""
    import json as _json
    import subprocess
    script = str(Path(__file__).parent / "bat_predict.py")
    proc = subprocess.run(
        [CLOUD_PYTHON, script,
         "--input",      tmp_path,
         "--config",     str(cfg_path),
         "--start",      str(start),
         "--max-models", str(max_models)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=timeout,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(msg[-1] if msg else "unknown error")
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("bat_predict.py produced no output")
    return _json.loads(stdout)


def _lookup_pipeline_result(dataset: str, session_idx: int) -> Optional[dict]:
    """Return single-session results from the last full pipeline run's .npy files.

    Returns None if pipeline outputs don't exist (fall back to fresh prediction).
    Uses point-adjusted predictions so results match the reported pipeline metrics.
    """
    out_base = ROOT / "outputs" / dataset.lower()
    needed   = ["edge_preds.npy", "hybrid_preds.npy"]
    if not all((out_base / f).exists() for f in needed):
        return None

    edge_adj = np.load(out_base / "edge_preds.npy")    # point-adjusted edge
    hybrid   = np.load(out_base / "hybrid_preds.npy")  # point-adjusted hybrid

    # Load actual routed event indices so routing display reflects reality.
    _ri_path = out_base / "routed_indices.npy"
    _routed_indices = np.load(_ri_path) if _ri_path.exists() else np.array([], dtype=np.int64)

    cfg_path = ROOT / "configs" / "inference" / f"{dataset}.yaml"
    win_size = 100
    if cfg_path.exists():
        with open(cfg_path) as _f:
            win_size = yaml.safe_load(_f).get("win_size", 100)

    # Both BGL and OS use proportional mapping from the index returned by
    # _find_window_for_raw to the correct position in the npy event arrays.
    # This ensures predictions match the pipeline exactly regardless of how
    # many events each session/line contributed.
    if dataset == "bgl":
        # BGL: session_idx = combined local test line index (raw log line).
        # Each line maps to one npy event proportionally.
        N_NORM_LINES  = _db.BGL_N_TEST_NORMAL_LINES    # 925712
        N_ANOM_LINES  = _db.BGL_N_TEST_ABNORMAL_LINES  # 348460
        N_NORM_EVENTS = _db.BGL_N_NORMAL_EVENTS         # 854957
        N_ANOM_EVENTS = _db.BGL_N_ABNORMAL_EVENTS       # 419143
        if session_idx < N_NORM_LINES:
            start = min(int(session_idx * N_NORM_EVENTS / N_NORM_LINES),
                        N_NORM_EVENTS - 1)
        else:
            k     = session_idx - N_NORM_LINES
            start = N_NORM_EVENTS + min(int(k * N_ANOM_EVENTS / N_ANOM_LINES),
                                        N_ANOM_EVENTS - 1)
        end = min(start + 1, len(hybrid))
    elif dataset == "os":
        # OS npy is per-line; session_idx is the direct npy position.
        start = min(session_idx, len(hybrid) - 1)
        end   = start + 1
    elif dataset == "hdfs":
        # HDFS: session_idx = local test line index (direct npy position).
        # Guard: if npy arrays are from an old short demo run, refuse to clamp
        # silently — return None so _fresh_predict is used instead.
        N_HDFS_TEST = _db.HDFS_N_TEST_NORMAL_LINES + _db.HDFS_N_TEST_ABNORMAL_LINES
        if len(hybrid) < N_HDFS_TEST // 2:   # clearly a stale short array
            return None
        start = min(session_idx, len(hybrid) - 1)
        end   = start + 1
    else:
        start = session_idx * win_size
        end   = min(start + win_size, len(hybrid))

    if start >= len(hybrid):
        return None

    n_lines     = end - start
    edge_pred   = 1 if edge_adj[start:end].sum() > 0 else 0
    hybrid_pred = 1 if hybrid[start:end].sum()   > 0 else 0

    # Check if the anchor event at `start` was routed to cloud.
    routed = bool(np.searchsorted(_routed_indices, start) < len(_routed_indices)
                  and _routed_indices[np.searchsorted(_routed_indices, start)] == start)

    def _load_thresholds_yaml(path):
        if not path.exists():
            return []
        with open(path) as _f:
            return yaml.safe_load(_f).get("models", [])

    # Edge: per-model point-adjusted votes from edge_preds_per_model.npy
    edge_model_results = []
    pm_path = out_base / "edge_preds_per_model.npy"
    if pm_path.exists():
        try:
            pm = np.load(pm_path)
            for col, t in enumerate(_load_thresholds_yaml(out_base / "thresholds_edge.yaml")):
                if col >= pm.shape[1]:
                    break
                vote = int(pm[start, col]) if start < len(pm) else edge_pred
                edge_model_results.append({
                    "model":     t["name"],
                    "energy":    None,
                    "threshold": round(float(t["threshold"]), 6),
                    "vote":      vote,
                })
        except Exception:
            pass

    n_edge      = len(edge_model_results) or 1
    n_anom_edge = sum(r["vote"] for r in edge_model_results) if edge_model_results else edge_pred
    edge_result = {
        "prediction": edge_pred, "label": "ANOMALY" if edge_pred else "NORMAL",
        "n_models": n_edge, "n_anomaly_votes": n_anom_edge,
        "n_normal_votes": n_edge - n_anom_edge, "model_results": edge_model_results,
    }

    # Cloud: per-model point-adjusted votes from cloud_preds_per_model.npy
    cloud_thresh_list   = _load_thresholds_yaml(out_base / "thresholds_cloud.yaml")
    cloud_model_results = []
    cp_path = out_base / "cloud_preds_per_model.npy"
    if cp_path.exists():
        try:
            cp = np.load(cp_path)
            for col, t in enumerate(cloud_thresh_list):
                if col >= cp.shape[1]:
                    break
                vote = int(cp[start, col]) if start < len(cp) else hybrid_pred
                cloud_model_results.append({
                    "model":     t["name"],
                    "energy":    None,
                    "threshold": round(float(t["threshold"]), 6),
                    "vote":      vote,
                })
        except Exception:
            pass
    if not cloud_model_results:
        cloud_model_results = [
            {"model": t["name"], "energy": None,
             "threshold": round(float(t["threshold"]), 6), "vote": hybrid_pred}
            for t in cloud_thresh_list
        ]
    n_cloud      = len(cloud_model_results) or 1
    n_anom_cloud = sum(r["vote"] for r in cloud_model_results) if cloud_model_results else hybrid_pred
    cloud_result = {
        "prediction": hybrid_pred, "label": "ANOMALY" if hybrid_pred else "NORMAL",
        "n_models": n_cloud, "n_anomaly_votes": n_anom_cloud,
        "n_normal_votes": n_cloud - n_anom_cloud, "model_results": cloud_model_results,
    }

    routing = {
        "routed":     routed,
        "avg_margin": 0.0,
        "reason":     "edge uncertain — routed to cloud" if routed
                      else "edge result confirmed by hybrid",
    }

    return {
        **cloud_result,
        "n_events":  n_lines,
        "elapsed_s": 0.0,
        "edge":      edge_result,
        "routing":   routing,
        "cloud":     cloud_result,
        "source":    "pipeline",
    }


def _fresh_predict(dataset: str, content: str) -> dict:
    """Score one session through the full three-stage pipeline.

    Uses in-RAM model cache when available (loaded at startup) so inference
    takes ~2-3s instead of 78s.  Falls back to subprocess on first call if
    the cache is not yet ready.
    """
    import time
    t0 = time.time()

    cfg_path = ROOT / "configs" / "inference" / f"{dataset}.yaml"
    if not cfg_path.exists():
        return {"error": f"No inference config for '{dataset}'"}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("cloud"):
        return {"error": f"Live prediction is not available for '{dataset}' — "
                          "no BAT cloud models configured. Switch to OpenStack."}

    win_size = cfg.get("win_size", 100)

    scaled = _scale_content(dataset, content)
    if not scaled:
        st = _scalers.get(dataset)
        if not st or not st.get("ready"):
            return {"error": "Scaler not ready — wait a moment after dashboard startup"}
        return {"error": "Session has no parseable events"}

    arr      = np.array(scaled, dtype=np.float32)
    n_events = len(arr)

    def _make_result(model_results):
        n      = len(model_results)
        n_anom = sum(r["vote"] for r in model_results)
        pred   = 1 if n_anom > n / 2 else 0
        padded = n_events < win_size
        return {
            "prediction":      pred,
            "label":           "ANOMALY" if pred else "NORMAL",
            "n_models":        n,
            "n_anomaly_votes": n_anom,
            "n_normal_votes":  n - n_anom,
            "model_results":   model_results,
            "n_events":        n_events,
            "win_size":        win_size,
            "padded":          padded,
        }

    try:
        if _bat_cache_ready.get(dataset, False):
            # ── Fast path: use in-RAM cache (no disk I/O) ────────────────────
            edge_results  = _predict_from_cache(arr, win_size, dataset,
                                                start=0, max_models=_N_EDGE_MODELS)
            cloud_results = _predict_from_cache(arr, win_size, dataset,
                                                start=0, max_models=_N_CLOUD_MODELS)
        else:
            # ── Slow path: subprocess (cache not ready yet) ───────────────────
            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                    tmp_path = f.name
                np.save(tmp_path, arr)
                edge_r  = _run_bat_subprocess(tmp_path, cfg_path,
                                              start=0, max_models=_N_EDGE_MODELS)
                cloud_r = _run_bat_subprocess(tmp_path, cfg_path,
                                              start=0, max_models=_N_CLOUD_MODELS)
                if "error" in edge_r:  return edge_r
                if "error" in cloud_r: return cloud_r
                edge_results  = edge_r.get("model_results", [])
                cloud_results = cloud_r.get("model_results", [])
            finally:
                if tmp_path:
                    try: os.unlink(tmp_path)
                    except OSError: pass

        edge  = _make_result(edge_results)
        cloud = _make_result(cloud_results)

        margins    = [abs(r["energy"] - r["threshold"]) / (r["threshold"] + 1e-9)
                      for r in edge_results]
        avg_margin = float(np.mean(margins)) if margins else 1.0
        routed     = avg_margin < 0.5 or (0 < edge["n_anomaly_votes"] < edge["n_models"])
        routing    = {
            "routed":     routed,
            "avg_margin": round(avg_margin, 4),
            "reason":     "uncertain (margin < 0.5 or models disagree)" if routed
                          else "confident — cloud confirms edge result",
        }

        return {**cloud,
                "elapsed_s": round(time.time() - t0, 2),
                "edge": edge, "routing": routing, "cloud": cloud}

    except subprocess.TimeoutExpired:
        return {"error": "Prediction timed out (>300 s)"}
    except Exception as exc:
        return {"error": f"Prediction error: {exc}"}


def _find_window_for_raw(dataset: str, split: str, line_number: int, block_id: str) -> Optional[int]:
    """Approximate: map a raw log line_number to the txt-file session index that contains it.

    The mapping is an estimate based on average raw-log lines per session.
    For HDFS the block_id gives an exact match.
    """
    import sqlite3
    db_path = str(Path(__file__).parent / "ceco_lad.db")

    if dataset == "os":
        if split == "train" or block_id == "train_normal":
            return None
        with sqlite3.connect(db_path, timeout=30) as c:
            row = c.execute(
                "SELECT MIN(line_number) FROM raw_logs WHERE dataset='os' AND block_id=?",
                (block_id,),
            ).fetchone()
        first = row[0] if (row and row[0] is not None) else 0
        local = max(0, line_number - first)
        if block_id == "test_normal":
            return min(int(local * _db.OS_N_NORMAL_EVENTS / _db.OS_N_TEST_NORMAL_LINES),
                       _db.OS_N_NORMAL_EVENTS - 1)
        else:  # test_abnormal
            k = min(int(local * _db.OS_N_ABNORMAL_EVENTS / _db.OS_N_TEST_ABNORMAL_LINES),
                    _db.OS_N_ABNORMAL_EVENTS - 1)
            return _db.OS_N_NORMAL_EVENTS + k

    elif dataset == "bgl":
        # Each raw BGL log line maps directly to one npy result — no session
        # grouping.  Return the combined local test index:
        #   test_normal  lines → 0 .. N_TEST_NORMAL_LINES-1
        #   test_abnormal lines → N_TEST_NORMAL_LINES .. N_TEST_NORMAL_LINES+N_TEST_ABNORMAL_LINES-1
        BGL_N_TRAIN_LINES = _db.BGL_N_TRAIN_LINES
        if block_id == "bgl_train" or split == "train":
            lines_per = _db.BGL_LINES_PER_SESSION.get("bgl_train", 41)
            return max(0, line_number // lines_per)
        local = max(0, line_number - BGL_N_TRAIN_LINES)
        return local  # combined test local line index

    elif dataset == "hdfs":
        # HDFS npy arrays are indexed by raw test line (same as BGL).
        # Return the test-local line index so _lookup_pipeline_result can index
        # directly into hybrid_preds.npy / routed_indices.npy.
        if split == "train":
            return None   # no pipeline results for train
        local = max(0, line_number - _db.HDFS_N_TRAIN_LINES)
        # The npy has one extra normal entry vs the DB (a blank line in
        # test_normal.log was skipped during ingest). Shift anomaly indices
        # by +1 so they align with the npy's anomaly region.
        if local >= _db.HDFS_N_TEST_NORMAL_LINES:
            local += 1
        return local

    return None


@app.get("/api/predict/from-raw")
async def predict_from_raw(
    dataset:     str = "os",
    split:       str = "test",
    line_number: int = 0,
    block_id:    str = "",
):
    """Find the session that contains a raw log line and predict its label.

    The session is looked up by mapping the raw log line_number to the
    corresponding entry in the source txt file, then scored with the
    cloud BAT ensemble exactly as predict_single does.
    """
    if line_number < 0:
        raise HTTPException(400, "line_number must be >= 0")

    window_idx = await asyncio.to_thread(
        _find_window_for_raw, dataset, split, line_number, block_id
    )
    if window_idx is None:
        return {"error": "Could not determine the session for this log line. "
                         "Check that the database is fully ingested."}

    paths     = _TXT_PATHS.get(dataset, {}).get(split, [])
    all_lines = await asyncio.to_thread(_read_txt_lines, paths)
    if not all_lines:
        return {"error": "No session data found. Check that the dataset is fully ingested."}

    # For BGL, HDFS, and OS the window_idx is a raw per-line npy index —
    # don't clamp against the txt-file session count.
    if dataset not in ("bgl", "hdfs", "os") and window_idx >= len(all_lines):
        window_idx = len(all_lines) - 1

    if split == "test":
        result = await asyncio.to_thread(_lookup_pipeline_result, dataset, window_idx)
    else:
        result = None

    if result is None:
        # For BGL: convert raw line index → session index for txt-file lookup.
        if dataset == "bgl" and split == "test":
            N_NORM = _db.BGL_N_TEST_NORMAL_LINES
            if window_idx < N_NORM:
                lp = _db.BGL_LINES_PER_SESSION.get("bgl_test_normal", 44)
                sess = min(window_idx // lp, _db.BGL_N_TEST_NORMAL_SESSIONS - 1)
            else:
                lp   = _db.BGL_LINES_PER_SESSION.get("bgl_test_abnormal", 27)
                k    = window_idx - N_NORM
                sess = _db.BGL_N_TEST_NORMAL_SESSIONS + min(k // lp, 13041)
            content = all_lines[sess] if sess < len(all_lines) else ""
        else:
            content = all_lines[window_idx]
        result  = await asyncio.to_thread(_fresh_predict, dataset, content)
        if "error" in result:
            return result
        result["source"] = "live"

    result["line_index"] = window_idx

    # Derive ground truth from pipeline npy gt array when available (HDFS, BGL),
    # or from block_id / windows table for other datasets.
    if dataset == "bgl":
        result["ground_truth"] = (
            1 if block_id == "bgl_test_abnormal" else
            0 if block_id in ("bgl_train", "bgl_test_normal") else None
        )
    elif dataset == "hdfs":
        # Ground truth derived directly from line_number: test_abnormal lines
        # start at HDFS_N_TRAIN_LINES + HDFS_N_TEST_NORMAL_LINES in the DB.
        # This is reliable regardless of npy file size.
        if split == "test" and line_number is not None:
            abnormal_start = _db.HDFS_N_TRAIN_LINES + _db.HDFS_N_TEST_NORMAL_LINES
            result["ground_truth"] = 1 if line_number >= abnormal_start else 0
        else:
            result["ground_truth"] = None
    elif dataset == "os":
        # OS npy position: normal region [0, OS_N_NORMAL_EVENTS), abnormal region after.
        if split == "test":
            result["ground_truth"] = 1 if window_idx >= _db.OS_N_NORMAL_EVENTS else 0
        else:
            result["ground_truth"] = None
    else:
        try:
            db_win = await _db.get_window_detail(dataset, split, window_idx)
            result["ground_truth"] = db_win.get("label") if db_win else None
        except Exception as exc:
            import logging as _log
            _log.warning("predict_from_raw: DB lookup failed %s/%s/%d: %s",
                         dataset, split, window_idx, exc)
            result["ground_truth"] = None

    return result


@app.get("/api/predict/single")
async def predict_single(
    dataset:      str = "os",
    split:        str = "test",
    window_index: int = 0,
):
    """Predict normal/anomaly for one session."""
    if window_index < 0:
        raise HTTPException(400, "window_index must be >= 0")

    paths     = _TXT_PATHS.get(dataset, {}).get(split, [])
    all_lines = await asyncio.to_thread(_read_txt_lines, paths)
    if window_index >= len(all_lines):
        raise HTTPException(404, f"Window index {window_index} out of range (total: {len(all_lines)})")

    content = all_lines[window_index]
    result  = await asyncio.to_thread(_fresh_predict, dataset, content)

    # _fresh_predict returns {"error": "..."} on failure — return as-is.
    if "error" in result:
        return result

    result["source"]       = "live"
    result["window_index"] = window_index

    # Ground truth from DB (label of this specific session; None if not ingested yet).
    try:
        db_win = await _db.get_window_detail(dataset, split, window_index)
        result["ground_truth"] = db_win.get("label") if db_win else None
    except Exception as exc:
        import logging as _log
        _log.warning("predict_single: DB lookup failed for %s/%s/%d: %s",
                     dataset, split, window_index, exc)
        result["ground_truth"] = None

    return result


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    await asyncio.to_thread(_db.init_db)
    # Always ingest if data is missing — ingest functions skip gracefully when
    # external log files are absent, and fall back to synthetic raw logs for OS.
    st = await _db.get_status()
    if not st.get("win_os_test") or not st.get("raw_os"):
        _trigger_ingest()
    # Re-ingest BGL windows from txt files if the old CSV-based import is present
    # (detects upgrade: old key is "bgl_windows"; new key is "bgl_windows_txt").
    import ingest as _ingest
    asyncio.create_task(asyncio.to_thread(_ingest.ingest_bgl_windows))
    # Fit StandardScaler for all datasets (needed for single-session prediction)
    for _ds in ("os", "hdfs", "bgl"):
        asyncio.create_task(asyncio.to_thread(_sync_fit_scaler, _ds))
    # Preload BAT models into RAM only when no separate cloud env is available
    # (i.e. HF Spaces / Docker where CLOUD_PYTHON == this interpreter).
    # Locally, the hybrid conda env subprocess is faster and uses the GPU.
    if CLOUD_PYTHON == sys.executable:
        for _ds in ("os", "bgl"):
            asyncio.create_task(asyncio.to_thread(_preload_bat_models, _ds))


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        (Path(__file__).parent / "index.html").read_text(),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/info")
async def info():
    """Return dashboard mode metadata (used by the frontend to detect demo mode)."""
    return {"demo_mode": DEMO_MODE}


# ── Process control ───────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    running = _proc is not None and _proc.returncode is None
    return {
        "running": running,
        "pid": _proc.pid if running else None,
        "returncode": _proc.returncode if _proc else None,
        "log_lines": len(_log_buf),
    }


@app.post("/api/run")
async def run(req: RunRequest):
    if DEMO_MODE and req.command in ("train", "eval", "convert", "download"):
        raise HTTPException(
            status_code=503,
            detail=f"'{req.command}' is disabled in demo mode.",
        )
    global _proc, _log_buf, _last_run_start, _last_run_ok
    if _proc is not None and _proc.returncode is None:
        raise HTTPException(status_code=409, detail="A process is already running")
    import time
    _last_run_start = time.time()
    _last_run_ok    = False

    async with _buf_lock:
        _log_buf.clear()

    cmd = _build_cmd(req)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "FORCE_COLOR": "0"}
    _proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(ROOT),
        env=env,
    )
    asyncio.create_task(_drain_proc())
    return {"pid": _proc.pid, "cmd": cmd}


@app.post("/api/stop")
async def stop():
    global _proc
    if _proc is None or _proc.returncode is not None:
        return {"stopped": False}
    _proc.terminate()
    try:
        await asyncio.wait_for(_proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        _proc.kill()
    return {"stopped": True}


async def _drain_proc():
    """Background task: read stdout into _log_buf."""
    global _proc, _last_run_ok
    while _proc and _proc.stdout and not _proc.stdout.at_eof():
        line = await _proc.stdout.readline()
        if line:
            async with _buf_lock:
                _log_buf.append(line.decode(errors="replace").rstrip())
    if _proc:
        await _proc.wait()
        if _proc.returncode == 0:
            _last_run_ok = True


# ── SSE log stream ────────────────────────────────────────────────────────────
@app.get("/api/stream")
async def stream():
    """Server-Sent Events: tail _log_buf in real-time."""
    async def gen():
        cursor = 0
        while True:
            async with _buf_lock:
                lines = _log_buf[cursor:]
                cursor += len(lines)
            for line in lines:
                yield f"data: {json.dumps(line)}\n\n"
            if _proc is None or _proc.returncode is not None:
                # Process finished — flush remaining then close
                async with _buf_lock:
                    tail = _log_buf[cursor:]
                for line in tail:
                    yield f"data: {json.dumps(line)}\n\n"
                yield "event: done\ndata: {}\n\n"
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Results ───────────────────────────────────────────────────────────────────
@app.get("/api/outputs/{dataset}")
async def outputs(dataset: str):
    out_dir = ROOT / "outputs" / dataset
    if not out_dir.exists():
        return {}
    if _results_are_stale(out_dir):
        return {}

    result: dict = {}

    # NumPy arrays — downsample long sequences for the browser
    npy_fields = [
        "ground_truth", "edge_preds", "hybrid_preds",
        "routed_indices", "energy_matrix", "cloud_preds",
    ]
    loaded: dict[str, np.ndarray] = {}
    for key in npy_fields:
        fpath = out_dir / f"{key}.npy"
        if fpath.exists():
            loaded[key] = np.load(str(fpath))

    MAX_PTS = 2000
    for key, arr in loaded.items():
        result[f"{key}_full_len"] = int(arr.shape[0])
        if arr.shape[0] > MAX_PTS:
            step = max(1, arr.shape[0] // MAX_PTS)
            result[key] = arr[::step].tolist()
            result[f"{key}_step"] = int(step)
        else:
            result[key] = arr.tolist()
            result[f"{key}_step"] = 1

    # YAML thresholds
    for fname in ("thresholds_cloud.yaml", "thresholds_edge.yaml"):
        fpath = out_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                key = fname.replace(".yaml", "").replace(".", "_")
                result[key] = yaml.safe_load(f)

    # Computed metrics
    gt = loaded.get("ground_truth")
    if gt is not None:
        stats: dict = {}
        for key, preds in [("edge", loaded.get("edge_preds")),
                            ("hybrid", loaded.get("hybrid_preds"))]:
            if preds is not None:
                stats.update(_metrics(gt, preds, key))
        result["stats"] = stats

    # Routing stats
    ri = loaded.get("routed_indices")
    gt = loaded.get("ground_truth")
    if ri is not None and gt is not None:
        result["routing_stats"] = {
            "total_windows": int(gt.shape[0]),
            "routed_windows": int(ri.shape[0]),
            "routing_pct": round(100.0 * ri.shape[0] / gt.shape[0], 2),
        }

    return result


def _metrics(gt: np.ndarray, pred: np.ndarray, prefix: str) -> dict:
    n = min(len(gt), len(pred))
    gt, pred = gt[:n].astype(int), pred[:n].astype(int)
    tp = int(((gt == 1) & (pred == 1)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    acc  = (tp + tn) / len(gt) if len(gt) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {
        f"{prefix}_accuracy":  round(acc,  4),
        f"{prefix}_precision": round(prec, 4),
        f"{prefix}_recall":    round(rec,  4),
        f"{prefix}_fscore":    round(f1,   4),
    }


# ── Config summary ───────────────────────────────────────────────────────────
@app.get("/api/config/{dataset}")
async def get_config_summary(dataset: str):
    """Return a human-readable summary of the inference config."""
    cfg_path = ROOT / "configs" / "inference" / f"{dataset}.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, "Config not found")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cloud = cfg.get("cloud") or {}

    # Count actual trained checkpoint files as the authoritative model count.
    # Fall back to the YAML sweep combination count if the directory is absent.
    n_cloud_ckpt = 0
    if cloud:
        ckpt_dir = cloud.get("model_save_path", "")
        if ckpt_dir:
            ckpt_path = ROOT / ckpt_dir
            if ckpt_path.exists():
                n_cloud_ckpt = len(list(ckpt_path.glob("*.pth")))
        if n_cloud_ckpt == 0:
            try:
                from itertools import product as iproduct
                keys = ["num_epochs", "k", "e_layer_num", "batch_size"]
                n_cloud_ckpt = len(list(iproduct(*[cloud[k] for k in keys])))
            except (KeyError, TypeError):
                n_cloud_ckpt = 0

    thresh_path = cfg.get("threshold_output", "")
    thresholds_ready = bool(thresh_path) and (ROOT / thresh_path).exists()

    return {
        "dataset":           cfg.get("dataset"),
        "win_size":          cfg.get("win_size"),
        "data_path":         cfg.get("data_path"),
        "output_dir":        cfg.get("output_dir"),
        "routing_tolerance": cfg.get("routing_tolerance", 0.1),
        "routing_distance":  cfg.get("routing_distance", "ma"),
        "edge_models":       [m["name"] for m in cfg.get("edge_models", [])],
        "has_cloud":         bool(cloud),
        "n_cloud_models":    n_cloud_ckpt,
        "cloud_voting":      cloud.get("voting", "majority") if cloud else None,
        "thresholds_ready":  thresholds_ready,
        "threshold_file":    thresh_path,
    }


# ── Logs ──────────────────────────────────────────────────────────────────────
@app.get("/api/logs")
async def list_logs():
    log_dir = ROOT / "logs"
    if not log_dir.exists():
        return {"logs": []}
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"logs": [{"name": p.name, "size": p.stat().st_size} for p in logs[:30]]}


@app.get("/api/logs/{filename}")
async def get_log(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    log_path = ROOT / "logs" / filename
    if not log_path.exists():
        raise HTTPException(404, "Not found")
    lines = log_path.read_text(errors="replace").splitlines()
    return {"lines": lines[-600:], "metrics": _parse_log_metrics(lines)}


def _parse_log_metrics(lines: list[str]) -> dict:
    metric_pat = re.compile(
        r"Accuracy:\s*([\d.]+)%?\s+Precision:\s*([\d.]+)%?\s+Recall:\s*([\d.]+)%?\s+F-score:\s*([\d.]+)%?"
    )
    voting_pat = re.compile(
        r"--\s*(majority|consensus|at.?least.?one)\s+voting\s+\(step\s+\d+/(\d+)\)"
    )
    steps: dict[str, list] = {"majority": [], "consensus": [], "at_least_one": []}
    current = None
    for line in lines:
        vm = voting_pat.search(line)
        if vm:
            name = vm.group(1).lower().replace(" ", "_").replace("-", "_")
            current = "at_least_one" if "least" in name else name
        mm = metric_pat.search(line)
        if mm and current and current in steps:
            steps[current].append({
                "acc":  float(mm.group(1)) / 100,
                "prec": float(mm.group(2)) / 100,
                "rec":  float(mm.group(3)) / 100,
                "f1":   float(mm.group(4)) / 100,
            })
    return steps


# ── Environment configuration ─────────────────────────────────────────────────
# Python interpreters for each pipeline phase.
# Override EDGE_PYTHON / CLOUD_PYTHON env vars to use a different interpreter
# (e.g. set both to sys.executable in Docker where conda envs don't exist).
_CONDA = Path.home() / "miniconda3" / "envs"
EDGE_PYTHON  = os.getenv("EDGE_PYTHON",  str(_CONDA / "ceco-lad" / "bin" / "python"))
CLOUD_PYTHON = os.getenv("CLOUD_PYTHON", str(_CONDA / "hybrid"   / "bin" / "python"))

# True when the full inference pipeline (run.py + ExecuTorch executor_runner) is present.
# Falls back to demo_runner.py when either component is missing.
_RUNNER_BIN = (
    ROOT / "inference_pipeline" / "executorch" / "cmake-out" / "executor_runner"
)
_FULL_PIPELINE_AVAILABLE = (
    (ROOT / "inference_pipeline" / "run.py").exists()
    and _RUNNER_BIN.exists()
)


# ── Config helpers ────────────────────────────────────────────────────────────
def _build_precomputed_infer_cmd(ds: str) -> list[str]:
    """Last-resort fallback: stream a brief status message then exit 0.
    Used when neither the full pipeline nor BAT checkpoints are available.
    The frontend then loads whatever is already in outputs/{ds}/."""
    script = "\n".join([
        f'echo "=== CECO-LAD  [{ds.upper()}] — no checkpoints available ==="',
        f'echo "Pre-computed results loaded from outputs/{ds}/. See the Results tab."',
    ])
    return ["bash", "-c", script]


def _build_container_infer_cmd(ds: str, tolerance: float, distance: str) -> list[str]:
    """Inference for container / HF Spaces using demo_runner.py.

    demo_runner.py runs the full four-stage pipeline entirely in Python using
    the BAT .pth checkpoints (no ExecuTorch required):
      Stage 1 — 3 fast BAT models as edge-proxy scan (parallel)
      Stage 2 — Mahalanobis routing
      Stage 3 — full BAT ensemble on routed windows (parallel)
      Stage 4 — point-adjusted hybrid metrics

    Falls back to the pre-computed status message when BAT checkpoints or a
    cloud section are absent for the requested dataset (e.g. BGL, HDFS).
    """
    # Need both a cloud section in the config and actual .pth checkpoints.
    if not _cfg_has_cloud(ds):
        return _build_precomputed_infer_cmd(ds)
    ckpt_dir = ROOT / "checkpoints" / "bat" / ds
    if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.pth")):
        return _build_precomputed_infer_cmd(ds)

    cfg_path   = _write_infer_cfg(ds, tolerance, distance, strip_cloud=False)
    runner     = str(Path(__file__).parent / "demo_runner.py")
    return [CLOUD_PYTHON, runner, "--config", cfg_path]


def _build_cmd(req: RunRequest) -> list[str]:
    ds = req.dataset
    if req.command == "infer" and not _FULL_PIPELINE_AVAILABLE:
        # Full inference pipeline not present (Docker / HF Spaces):
        # use demo_runner.py which executes BAT models for both edge and cloud.
        return _build_container_infer_cmd(ds, req.routing_tolerance, req.routing_distance)
    if req.command == "train":
        return [CLOUD_PYTHON, "-m", "training_pipeline.train",
                "--config", f"configs/training/{ds}.yaml"]
    if req.command == "eval":
        return [CLOUD_PYTHON, "-m", "training_pipeline.evaluate",
                "--config", f"configs/training/{ds}.yaml",
                "--voting", req.voting]
    if req.command == "convert":
        return [EDGE_PYTHON, "quantization/convert.py",
                "--config", f"configs/training/{ds}.yaml", "--all"]
    if req.command == "infer":
        return _build_infer_cmd(ds, req.routing_tolerance, req.routing_distance)
    if req.command == "download":
        return [CLOUD_PYTHON, "tools/download_checkpoints.py", "--dataset", ds]
    raise HTTPException(400, f"Unknown command: {req.command}")


def _build_infer_cmd(ds: str, tolerance: float, distance: str) -> list[str]:
    """Build a two-phase inference command (edge env → cloud env)."""
    # Edge-only config: cloud section stripped so run.py stops after routing
    edge_cfg  = _write_infer_cfg(ds, tolerance, distance, strip_cloud=True)
    # Full config: cloud section intact for cloud_runner.py
    cloud_cfg = _write_infer_cfg(ds, tolerance, distance, strip_cloud=False)

    has_cloud = _cfg_has_cloud(ds)

    if has_cloud and EDGE_PYTHON != CLOUD_PYTHON:
        script = (
            f'set -euo pipefail\n'
            f'echo "--- Phase 1/2: Edge inference  (env: ceco-lad) ---"\n'
            f'{EDGE_PYTHON} -m inference_pipeline.run --config {edge_cfg}\n'
            f'echo "--- Phase 2/2: Cloud inference (env: hybrid)   ---"\n'
            f'{CLOUD_PYTHON} dashboard/cloud_runner.py --config {cloud_cfg}\n'
        )
    else:
        # No cloud section, or same env — run everything in the edge env
        script = (
            f'set -euo pipefail\n'
            f'echo "--- Edge inference (env: ceco-lad) ---"\n'
            f'{EDGE_PYTHON} -m inference_pipeline.run --config {cloud_cfg}\n'
        )
    return ["bash", "-c", script]


def _write_infer_cfg(ds: str, tolerance: float, distance: str,
                     strip_cloud: bool) -> str:
    base = ROOT / "configs" / "inference" / f"{ds}.yaml"
    with open(base) as f:
        cfg = yaml.safe_load(f)
    cfg["routing_tolerance"] = tolerance
    cfg["routing_distance"]  = distance
    if strip_cloud:
        cfg.pop("cloud", None)
    suffix = "edge" if strip_cloud else "full"
    out = ROOT / "configs" / "inference" / f"{ds}_dashboard_{suffix}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return f"configs/inference/{ds}_dashboard_{suffix}.yaml"


def _cfg_has_cloud(ds: str) -> bool:
    base = ROOT / "configs" / "inference" / f"{ds}.yaml"
    with open(base) as f:
        return bool(yaml.safe_load(f).get("cloud"))


# ── Database ──────────────────────────────────────────────────────────────────
_ingest_running = False
_ingest_log: list[str] = []


def _trigger_ingest():
    """Start background ingestion if not already running."""
    global _ingest_running, _ingest_log
    if _ingest_running:
        return
    _ingest_log = []
    _ingest_running = True

    def _run():
        global _ingest_running
        try:
            import ingest as _ingest
            _ingest.run_full_ingest(lambda msg: _ingest_log.append(msg))
        finally:
            _ingest_running = False

    threading.Thread(target=_run, daemon=True).start()


@app.get("/api/db/status")
async def db_status():
    return {
        "importing": _ingest_running,
        "import_log": _ingest_log[-60:],
        **(await _db.get_status()),
    }


@app.get("/api/db/raw-logs")
async def api_raw_logs(
    dataset: str = "bgl",
    page: int = 0,
    per_page: int = 100,
    search: str = "",
    label: str = "",
):
    return await _db.query_raw_logs(dataset, page, per_page, search, label)


@app.get("/api/db/windows")
async def api_windows(
    dataset: str = "bgl",
    split: str = "test",
    page: int = 0,
    per_page: int = 50,
    label: str = "",
):
    return await _db.query_windows(dataset, split, page, per_page, label)


@app.get("/api/db/window/{dataset}/{split}/{window_index}")
async def api_window_detail(dataset: str, split: str, window_index: int):
    data = await _db.get_window_detail(dataset, split, window_index)
    if not data:
        raise HTTPException(404, "Window not found")
    return data


@app.get("/api/db/pipeline/{dataset}")
async def api_pipeline_stats(dataset: str):
    return await _db.get_pipeline_stats(dataset)


# ── Unified pipeline view (all datasets) ─────────────────────────────────────

@app.get("/api/pipeline/raw-logs")
async def api_pipeline_raw_logs(
    dataset: str = "os",
    split: str   = "train",
    page: int    = 0,
    per_page: int = 100,
    search: str  = "",
    label: str   = "",
):
    """Raw log lines for any dataset/split, with split-aware filtering."""
    return await _db.query_pipeline_raw(dataset, split, page, per_page, search, label)


@app.get("/api/pipeline/sessions")
async def api_pipeline_sessions(
    dataset:  str  = "os",
    split:    str  = "test",
    page:     int  = 0,
    per_page: int  = 20,
    scaled:   bool = True,
):
    """Paginated event sequences from dataset txt files, optionally scaled."""
    paths = _TXT_PATHS.get(dataset, {}).get(split, [])
    all_lines = await asyncio.to_thread(_read_txt_lines, paths)
    total      = len(all_lines)
    page_lines = all_lines[page * per_page: (page + 1) * per_page]

    st = _scalers.get(dataset)
    scaler_ready = bool(st and st.get("ready"))

    rows = []
    for line in page_lines:
        row: dict = {"n_events": len(line.split())}
        if scaled and scaler_ready:
            row["scaled_matrix"] = _scale_content(dataset, line)
        rows.append(row)

    return {"total": total, "page": page, "per_page": per_page,
            "rows": rows, "scaler_ready": scaler_ready}


# ── OpenStack pipeline view (legacy endpoints kept for backward compat) ────────

@app.get("/api/pipeline/os/sessions")
async def api_os_sessions(
    split: str = "test",
    page: int = 0,
    per_page: int = 20,
    label: str = "",
    scaled: bool = False,
):
    result = await _db.query_os_sessions(split, page, per_page, label)
    os_scaler = _scalers.get("os")
    result["scaler_ready"] = bool(os_scaler and os_scaler.get("ready"))
    if scaled and result["scaler_ready"]:
        for row in result["rows"]:
            row["scaled_matrix"] = _scale_content("os", row.get("content_preview") or "")
    return result


@app.get("/api/pipeline/os/session/{session_idx}")
async def api_os_session(session_idx: int, split: str = "test"):
    data = await _db.get_os_session(session_idx, split)
    if not data:
        raise HTTPException(404, "Session not found")
    return data


@app.get("/api/pipeline/os/raw")
async def api_os_raw(
    source: str = "test_normal",
    page: int = 0,
    per_page: int = 100,
    search: str = "",
):
    return await _db.query_os_raw(source, page, per_page, search)


@app.get("/api/pipeline/os/raw-split")
async def api_os_raw_split(
    split: str = "test",
    page: int = 0,
    per_page: int = 100,
    search: str = "",
):
    """Raw logs for a full split: 'train' or 'test' (test_normal + test_abnormal combined)."""
    return await _db.query_os_raw_split(split, page, per_page, search)


def _results_are_stale(out_dir: Path) -> bool:
    """Return True if results files predate the last run (run was stopped/failed)."""
    if _last_run_ok or _last_run_start is None:
        return False
    npy_files = list(out_dir.glob("*.npy"))
    if not npy_files:
        return False
    latest_mtime = max(f.stat().st_mtime for f in npy_files)
    return latest_mtime < _last_run_start


@app.get("/api/pipeline/os/results")
async def api_os_results():
    """Return OS inference results (predictions + routing) for the pipeline view."""
    out_dir = ROOT / "outputs" / "os"
    if not out_dir.exists():
        return {"available": False}
    if _results_are_stale(out_dir):
        return {"available": False, "reason": "stopped"}

    import numpy as np
    result: dict = {"available": True}
    for key in ("ground_truth", "edge_preds", "hybrid_preds", "routed_indices", "energy_matrix"):
        fp = out_dir / f"{key}.npy"
        if fp.exists():
            arr = np.load(str(fp))
            result[key] = arr.tolist()

    # Per-session summary for the pipeline table
    gt  = result.get("ground_truth", [])
    ep  = result.get("edge_preds",   [])
    hp  = result.get("hybrid_preds", [])
    ri  = set(result.get("routed_indices", []))

    result["session_summary"] = [
        {
            "idx":         i,
            "gt":          int(gt[i])  if i < len(gt) else None,
            "edge_pred":   int(ep[i])  if i < len(ep) else None,
            "hybrid_pred": int(hp[i])  if i < len(hp) else None,
            "routed":      i in ri,
        }
        for i in range(len(gt))
    ]
    return result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
