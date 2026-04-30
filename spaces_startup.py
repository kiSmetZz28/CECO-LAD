#!/usr/bin/env python3
"""First-launch startup script for cloud deployments (HF Spaces / Railway / Fly.io).

On every cold start it checks whether the BAT model checkpoints are present.
If they are missing it downloads them from Google Drive (~3.5 GB, one-time).
Subsequent starts skip the download immediately.

Falls back gracefully when the download fails — the dashboard still works in
pre-computed-results mode (all four pipeline stages are shown; live inference
uses cached numpy outputs).

Usage (set automatically via Dockerfile CMD):
    python spaces_startup.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints" / "bat" / "os"
MIN_CKPTS = 81          # expected number of OS BAT checkpoints


def _ckpts_present() -> bool:
    if not CKPT_DIR.exists():
        return False
    return len(list(CKPT_DIR.glob("*.pth"))) >= MIN_CKPTS


def _download() -> bool:
    print("[startup] BAT checkpoints not found.", flush=True)
    print(
        "[startup] Downloading from Google Drive (~3.5 GB) — "
        "this takes 5-15 min on first launch; subsequent starts skip this step.",
        flush=True,
    )
    try:
        # tools/download_checkpoints.py handles gdown + extraction
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "download_checkpoints.py"),
                "--type", "bat",
                "--dataset", "os",
            ],
            cwd=str(ROOT),
        )
        if result.returncode == 0 and _ckpts_present():
            print(f"[startup] Downloaded {len(list(CKPT_DIR.glob('*.pth')))} checkpoints.", flush=True)
            return True
        print("[startup] Download finished but checkpoints missing — check quota/permissions.", flush=True)
        return False
    except Exception as exc:
        print(f"[startup] Download error: {exc}", flush=True)
        return False


def _launch_app() -> None:
    port = int(os.getenv("PORT", "7860"))
    print(f"[startup] Starting CECO-LAD dashboard on port {port} …", flush=True)
    # Replace this process with the app so signals are forwarded correctly
    os.environ["PORT"] = str(port)
    os.execv(sys.executable, [sys.executable, str(ROOT / "dashboard" / "app.py")])


if __name__ == "__main__":
    if _ckpts_present():
        print(
            f"[startup] {len(list(CKPT_DIR.glob('*.pth')))} BAT checkpoints present — skipping download.",
            flush=True,
        )
    else:
        ok = _download()
        if not ok:
            print(
                "[startup] Continuing without checkpoints — "
                "'Run Inference' will display pre-computed results.",
                flush=True,
            )

    _launch_app()
