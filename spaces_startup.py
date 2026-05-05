#!/usr/bin/env python3
"""First-launch startup script for HF Spaces / cloud deployments.

Downloads large assets from a Hugging Face dataset repository (no rate limits,
no quota issues unlike Google Drive).  Subsequent starts skip downloads.

Assets repo: kiSmetZz/ceco-lad-assets  (dataset, public)
  executor_runner      — ExecuTorch C++ binary     (~43 MB)
  bat/os.zip           — BAT OS checkpoints        (~3.5 GB)
  bat/bgl.zip          — BAT BGL checkpoints       (~3.5 GB)
  bat/hdfs.zip         — BAT HDFS checkpoints      (~3.5 GB)
  os_logs.zip          — OpenStack raw logs        (~59 MB)
  bgl_logs.zip         — BGL processed data files  (~200 MB)
  bgl_raw.zip          — BGL raw log (BGL.log)     (~709 MB)
  hdfs_raw.zip         — HDFS raw log (HDFS.log)   (~1.5 GB)
  (HDFS processed data files are bundled in the repo)

Usage (set automatically via Dockerfile CMD):
    python spaces_startup.py
"""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent

# HF dataset repo that holds all large assets
HF_ASSETS_REPO = "kiSmetZz/ceco-lad-assets"

# ── BAT checkpoints ───────────────────────────────────────────────────────────
OS_CKPT_DIR   = ROOT / "checkpoints" / "bat" / "os"
BGL_CKPT_DIR  = ROOT / "checkpoints" / "bat" / "bgl"
HDFS_CKPT_DIR = ROOT / "checkpoints" / "bat" / "hdfs"
MIN_CKPTS     = 81

# ── executor_runner binary ────────────────────────────────────────────────────
RUNNER_PATH = ROOT / "inference_pipeline" / "executorch" / "cmake-out" / "executor_runner"

# ── OpenStack raw log files ───────────────────────────────────────────────────
OS_RAW_DIR   = ROOT / "data" / "OpenStack" / "raw"
OS_RAW_FILES = ["openstack_normal1.log", "openstack_normal2.log", "openstack_abnormal.log"]

# ── BGL data files ────────────────────────────────────────────────────────────
BGL_DATA_DIR   = ROOT / "data" / "BGL"
BGL_DATA_FILES = ["bgl_train.txt", "bgl_test_normal.txt", "bgl_test_abnormal.txt"]

# ── Raw structured log CSVs (for raw log display in the dashboard) ─────────────
LOG_DATA_ROOT    = Path.home() / "Desktop" / "Log Data"
BGL_LOG_DIR      = LOG_DATA_ROOT / "BGL"
BGL_SPLIT_DIR    = BGL_LOG_DIR / "split"
BGL_SPLIT_FILES  = ["bgl_train.log", "bgl_test_normal.log", "bgl_test_abnormal.log"]
HDFS_SPLIT_DIR   = LOG_DATA_ROOT          # train/test_normal/test_abnormal.log live here
HDFS_SPLIT_FILES = ["train.log", "test_normal.log", "test_abnormal.log"]


def _hf_download(filename: str, local_path: Path) -> bool:
    """Download one file from the HF assets repo to local_path."""
    try:
        from huggingface_hub import hf_hub_download
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = hf_hub_download(
            repo_id=HF_ASSETS_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_path.parent),
        )
        # hf_hub_download may put file in a subdirectory — move it if needed
        downloaded = Path(tmp)
        if downloaded != local_path:
            downloaded.rename(local_path)
        return local_path.exists()
    except Exception as exc:
        print(f"[startup] HF download error ({filename}): {exc}", flush=True)
        return False


# ── executor_runner ───────────────────────────────────────────────────────────

def _runner_present() -> bool:
    return RUNNER_PATH.is_file() and os.access(str(RUNNER_PATH), os.X_OK)


def _download_runner() -> bool:
    print("[startup] executor_runner not found. Downloading from HF (~43 MB)…", flush=True)
    if _hf_download("executor_runner", RUNNER_PATH):
        os.chmod(str(RUNNER_PATH), 0o755)
        print("[startup] executor_runner ready.", flush=True)
        return True
    print("[startup] executor_runner download failed — edge inference unavailable.", flush=True)
    return False


# ── OpenStack raw logs ────────────────────────────────────────────────────────

def _os_logs_present() -> bool:
    return OS_RAW_DIR.exists() and all(
        (OS_RAW_DIR / f).is_file() for f in OS_RAW_FILES
    )


def _download_os_logs() -> bool:
    print("[startup] OpenStack raw logs not found. Downloading from HF (~59 MB)…", flush=True)
    zip_path = ROOT / "data" / "OpenStack" / "os_logs.zip"
    if not _hf_download("os_logs.zip", zip_path):
        print("[startup] OpenStack log download failed — synthetic messages will be used.", flush=True)
        return False
    try:
        OS_RAW_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(OS_RAW_DIR))
        zip_path.unlink(missing_ok=True)
        if _os_logs_present():
            print("[startup] OpenStack raw logs ready.", flush=True)
            return True
        print("[startup] Extraction finished but files missing.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] OpenStack log extraction error: {exc}", flush=True)
        return False


# ── BGL split logs ────────────────────────────────────────────────────────────

def _bgl_split_present() -> bool:
    return BGL_SPLIT_DIR.exists() and all(
        (BGL_SPLIT_DIR / f).is_file() for f in BGL_SPLIT_FILES
    )


def _download_bgl_raw() -> bool:
    print("[startup] BGL split log files not found. Downloading from HF (~200–400 MB)…",
          flush=True)
    zip_path = BGL_LOG_DIR / "bgl_raw.zip"
    BGL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not _hf_download("bgl_raw.zip", zip_path):
        print("[startup] BGL raw log download failed — raw log panel will be empty.",
              flush=True)
        return False
    try:
        BGL_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(BGL_SPLIT_DIR))
        zip_path.unlink(missing_ok=True)
        if _bgl_split_present():
            print("[startup] BGL split log files ready.", flush=True)
            return True
        print("[startup] BGL extraction finished but split files missing.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] BGL raw log extraction error: {exc}", flush=True)
        return False


# ── HDFS split logs ───────────────────────────────────────────────────────────

def _hdfs_split_present() -> bool:
    return HDFS_SPLIT_DIR.exists() and all(
        (HDFS_SPLIT_DIR / f).is_file() for f in HDFS_SPLIT_FILES
    )


def _download_hdfs_split() -> bool:
    print("[startup] HDFS split log files not found. Downloading from HF (~1.6 GB)…",
          flush=True)
    zip_path = HDFS_SPLIT_DIR / "hdfs_split.zip"
    HDFS_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if not _hf_download("hdfs_split.zip", zip_path):
        print("[startup] HDFS split log download failed — raw log panel will be empty.",
              flush=True)
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(HDFS_SPLIT_DIR))
        zip_path.unlink(missing_ok=True)
        if _hdfs_split_present():
            print("[startup] HDFS split log files ready.", flush=True)
            return True
        print("[startup] HDFS extraction finished but split files missing.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] HDFS split log extraction error: {exc}", flush=True)
        return False


# ── BAT checkpoints ───────────────────────────────────────────────────────────

def _ckpts_present(ckpt_dir: Path) -> bool:
    return ckpt_dir.exists() and len(list(ckpt_dir.glob("*.pth"))) >= MIN_CKPTS


def _download_checkpoints(dataset: str, ckpt_dir: Path) -> bool:
    print(
        f"[startup] BAT {dataset.upper()} checkpoints not found. Downloading from HF (~3.5 GB) — "
        "this takes 5–15 min on first launch.",
        flush=True,
    )
    zip_path = ROOT / "checkpoints" / "bat" / f"{dataset}.zip"
    if not _hf_download(f"bat/{dataset}.zip", zip_path):
        print(f"[startup] {dataset.upper()} checkpoint download failed.", flush=True)
        return False
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(ckpt_dir))
        zip_path.unlink(missing_ok=True)
        n = len(list(ckpt_dir.glob("*.pth")))
        if n >= MIN_CKPTS:
            print(f"[startup] {n} BAT {dataset.upper()} checkpoints ready.", flush=True)
            return True
        print(f"[startup] Only {n}/{MIN_CKPTS} {dataset.upper()} checkpoints extracted.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] {dataset.upper()} checkpoint extraction error: {exc}", flush=True)
        return False


# ── BGL data files ────────────────────────────────────────────────────────────

def _bgl_data_present() -> bool:
    return BGL_DATA_DIR.exists() and all(
        (BGL_DATA_DIR / f).is_file() for f in BGL_DATA_FILES
    )


def _download_bgl_data() -> bool:
    print("[startup] BGL data files not found. Downloading from HF (~200 MB)…", flush=True)
    zip_path = BGL_DATA_DIR / "bgl_logs.zip"
    if not _hf_download("bgl_logs.zip", zip_path):
        print("[startup] BGL data download failed — BGL dataset unavailable.", flush=True)
        return False
    try:
        BGL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(BGL_DATA_DIR))
        zip_path.unlink(missing_ok=True)
        if _bgl_data_present():
            print("[startup] BGL data files ready.", flush=True)
            return True
        print("[startup] BGL extraction finished but files missing.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] BGL data extraction error: {exc}", flush=True)
        return False


# ── Pre-computed npy outputs ──────────────────────────────────────────────────

OUTPUT_NP_FILES = [
    "ground_truth.npy", "edge_preds.npy", "edge_preds_raw.npy",
    "hybrid_preds.npy", "cloud_preds.npy",
    "energy_matrix.npy", "routed_indices.npy",
    "edge_preds_per_model.npy", "cloud_preds_per_model.npy",
]


def _outputs_present(ds: str) -> bool:
    out_dir = ROOT / "outputs" / ds
    required = [
        "ground_truth.npy", "edge_preds.npy", "hybrid_preds.npy",
        "cloud_preds.npy", "routed_indices.npy",
        "edge_preds_per_model.npy", "cloud_preds_per_model.npy",
    ]
    return all((out_dir / f).exists() for f in required)


def _download_outputs(ds: str) -> bool:
    print(f"[startup] {ds.upper()} inference outputs not found. Downloading from HF…",
          flush=True)
    out_dir  = ROOT / "outputs" / ds
    zip_path = out_dir / f"{ds}_outputs.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _hf_download(f"outputs/{ds}.zip", zip_path):
        print(f"[startup] {ds.upper()} outputs download failed — prediction lookup unavailable.",
              flush=True)
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(out_dir))
        zip_path.unlink(missing_ok=True)
        if _outputs_present(ds):
            present = [f for f in OUTPUT_NP_FILES if (out_dir / f).exists()]
            print(f"[startup] {ds.upper()} outputs ready ({len(present)} npy files).", flush=True)
            return True
        print(f"[startup] {ds.upper()} outputs extraction incomplete.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] {ds.upper()} outputs extraction error: {exc}", flush=True)
        return False


# ── Startup ───────────────────────────────────────────────────────────────────

def _clear_precomputed_results() -> None:
    pass


def _launch_app() -> None:
    port = int(os.getenv("PORT", "7860"))
    print(f"[startup] Starting CECO-LAD dashboard on port {port} …", flush=True)
    os.environ["PORT"] = str(port)
    os.execv(sys.executable, [sys.executable, str(ROOT / "dashboard" / "app.py")])


if __name__ == "__main__":
    _clear_precomputed_results()

    for _ds in ("bgl", "hdfs", "os"):
        if _outputs_present(_ds):
            print(f"[startup] {_ds.upper()} inference outputs present — skipping download.",
                  flush=True)
        else:
            _download_outputs(_ds)

    if _os_logs_present():
        print("[startup] OpenStack raw logs present — skipping download.", flush=True)
    else:
        _download_os_logs()

    if _bgl_data_present():
        print("[startup] BGL data files present — skipping download.", flush=True)
    else:
        _download_bgl_data()

    if _bgl_split_present():
        print("[startup] BGL split log files present — skipping download.", flush=True)
    else:
        _download_bgl_raw()

    if _hdfs_split_present():
        print("[startup] HDFS split log files present — skipping download.", flush=True)
    else:
        _download_hdfs_split()

    if _runner_present():
        print("[startup] executor_runner present — skipping download.", flush=True)
    else:
        _download_runner()

    if _ckpts_present(OS_CKPT_DIR):
        print(f"[startup] {len(list(OS_CKPT_DIR.glob('*.pth')))} BAT OS checkpoints present — skipping download.", flush=True)
    else:
        _download_checkpoints("os", OS_CKPT_DIR)

    if _ckpts_present(BGL_CKPT_DIR):
        print(f"[startup] {len(list(BGL_CKPT_DIR.glob('*.pth')))} BAT BGL checkpoints present — skipping download.", flush=True)
    else:
        _download_checkpoints("bgl", BGL_CKPT_DIR)

    if _ckpts_present(HDFS_CKPT_DIR):
        print(f"[startup] {len(list(HDFS_CKPT_DIR.glob('*.pth')))} BAT HDFS checkpoints present — skipping download.", flush=True)
    else:
        _download_checkpoints("hdfs", HDFS_CKPT_DIR)

    _launch_app()
