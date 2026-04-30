#!/usr/bin/env python3
"""CECO-LAD Web Dashboard — FastAPI backend."""
import asyncio
import json
import os
import re
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
# Number of BAT checkpoints to load for the interactive "Predict" button.
# Capped so a single prediction returns within ~2 seconds on CPU.
_SINGLE_PREDICT_MAX_MODELS = 9


def _fresh_predict(dataset: str, content: str) -> dict:
    """Score one event-ID sequence through the cloud BAT ensemble.

    Environment split — mirrors the full-pipeline two-phase approach:
      • Scaling runs here, in the ceco-lad env (scaler already fitted in memory).
      • Model loading + GPU inference runs in the hybrid env via bat_predict.py,
        called as a subprocess through CLOUD_PYTHON.

    Communication:
      • Input  → temp .npy file  (scaled [n_events, input_c] float32 array)
      • Output ← JSON on stdout  (prediction result dict)
    """
    import json as _json
    import subprocess
    import tempfile
    import time

    t0 = time.time()

    # ── Check config ─────────────────────────────────────────────────────────
    cfg_path = ROOT / "configs" / "inference" / f"{dataset}.yaml"
    if not cfg_path.exists():
        return {"error": f"No inference config for '{dataset}'"}

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("cloud"):
        return {"error": f"Live session prediction is not available for '{dataset}' — "
                          "no BAT cloud models are configured. Switch to OpenStack for a live demo."}

    win_size = cfg.get("win_size", 100)
    input_c  = cfg.get("input_c", 10)

    # ── Scale content in ceco-lad env ────────────────────────────────────────
    scaled = _scale_content(dataset, content)
    if not scaled:
        st = _scalers.get(dataset)
        if not st or not st.get("ready"):
            return {"error": "Scaler not ready yet — wait a moment after dashboard startup"}
        return {"error": "Session has no parseable events"}

    arr      = np.array(scaled, dtype=np.float32)   # [n_events, input_c]
    n_events = len(arr)

    # ── Call bat_predict.py in hybrid env ────────────────────────────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tmp_path = f.name
        np.save(tmp_path, arr)

        script = str(Path(__file__).parent / "bat_predict.py")
        proc   = subprocess.run(
            [
                CLOUD_PYTHON, script,
                "--input",      tmp_path,
                "--config",     str(cfg_path),
                "--max-models", str(_SINGLE_PREDICT_MAX_MODELS),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=120,
        )

        if proc.returncode != 0:
            lines = (proc.stderr or "").strip().splitlines()
            msg   = lines[-1] if lines else "unknown error"
            return {"error": f"BAT inference subprocess failed: {msg}"}

        stdout = proc.stdout.strip()
        if not stdout:
            return {"error": "BAT inference produced no output — check the hybrid env"}

        result = _json.loads(stdout)
        # Replace the sub-process elapsed time with wall-clock time including launch.
        if "elapsed_s" in result:
            result["elapsed_s"] = round(time.time() - t0, 2)
        # Attach n_events from this side (subprocess reports its own; may differ if padded).
        result.setdefault("n_events", n_events)
        return result

    except subprocess.TimeoutExpired:
        return {"error": "BAT inference timed out (>120 s) — try reducing max_models"}
    except _json.JSONDecodeError as exc:
        return {"error": f"Could not parse subprocess output: {exc}"}
    except Exception as exc:
        return {"error": f"Prediction error: {exc}"}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _find_window_for_raw(dataset: str, split: str, line_number: int, block_id: str) -> Optional[int]:
    """Approximate: map a raw log line_number to the txt-file session index that contains it.

    The mapping is an estimate based on average raw-log lines per session.
    For HDFS the block_id gives an exact match.
    """
    import sqlite3
    db_path = str(Path(__file__).parent / "ceco_lad.db")

    if dataset == "os":
        lines_per = {"train_normal": 136, "test_normal": 110,
                     "test_abnormal": 134}.get(block_id, 110)
        with sqlite3.connect(db_path, timeout=30) as c:
            row = c.execute(
                "SELECT MIN(line_number) FROM raw_logs WHERE dataset='os' AND block_id=?",
                (block_id,),
            ).fetchone()
        first = row[0] if (row and row[0] is not None) else 0
        local = max(0, line_number - first)
        idx   = local // lines_per
        if block_id == "test_abnormal":
            with sqlite3.connect(db_path, timeout=30) as c:
                n_normal = c.execute(
                    "SELECT COUNT(*) FROM windows "
                    "WHERE dataset='os' AND split='test' AND block_id='test_normal'"
                ).fetchone()[0]
            idx += n_normal
        return int(idx)

    elif dataset == "bgl":
        if split == "train":
            return max(0, line_number // 100)
        test_start = _db.BGL_N_TRAIN * 100
        return max(0, (line_number - test_start) // 100)

    elif dataset == "hdfs":
        if not block_id:
            return None
        with sqlite3.connect(db_path, timeout=30) as c:
            row = c.execute(
                "SELECT window_index FROM windows "
                "WHERE dataset='hdfs' AND split=? AND block_id=? LIMIT 1",
                (split, block_id),
            ).fetchone()
        return int(row[0]) if row else None

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
    if window_idx >= len(all_lines):
        return {"error": f"Mapped to session #{window_idx} but only {len(all_lines)} sessions exist. "
                         "This log line may be at the boundary of the dataset."}

    content = all_lines[window_idx]
    result  = await asyncio.to_thread(_fresh_predict, dataset, content)
    if "error" in result:
        return result

    result["source"]       = "live"
    result["window_index"] = window_idx

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
    # Fit StandardScaler for all datasets (needed for single-session prediction)
    for _ds in ("os", "hdfs", "bgl"):
        asyncio.create_task(asyncio.to_thread(_sync_fit_scaler, _ds))


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

# True when the full inference pipeline (edge_agent + run.py) is present on disk.
# In Docker / HF Spaces only cloud_expert.py is included, so this will be False
# and inference falls back to streaming pre-computed results.
_FULL_PIPELINE_AVAILABLE = (ROOT / "inference_pipeline" / "run.py").exists()


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
        return [EDGE_PYTHON, "edge_pipeline/convert.py",
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
):
    """Raw log lines for any dataset/split, with split-aware filtering."""
    return await _db.query_pipeline_raw(dataset, split, page, per_page, search)


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
