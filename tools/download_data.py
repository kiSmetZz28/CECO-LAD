"""Download raw log split files needed by the CECO-LAD dashboard.

Downloads from the HF assets repo (kiSmetZz/ceco-lad-assets) to the local
log root directory.  By default logs are placed in ~/Desktop/Log Data, but
you can override this with the CECO_LOG_ROOT environment variable.

Files downloaded:
  bgl_raw.zip   → $CECO_LOG_ROOT/BGL/split/  (bgl_train.log, bgl_test_*.log)  ~700 MB
  hdfs_split.zip → $CECO_LOG_ROOT/           (train.log, test_normal.log, test_abnormal.log) ~1.6 GB

OpenStack raw logs are already bundled in data/OpenStack/raw/ (tracked in git).

Usage
-----
  python tools/download_data.py              # download all
  python tools/download_data.py --dataset bgl
  python tools/download_data.py --dataset hdfs
  python tools/download_data.py --list
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

HF_ASSETS_REPO = "kiSmetZz/ceco-lad-assets"

LOG_ROOT     = Path(os.environ.get("CECO_LOG_ROOT", Path.home() / "Desktop" / "Log Data"))
BGL_SPLIT_DIR  = LOG_ROOT / "BGL" / "split"
HDFS_SPLIT_DIR = LOG_ROOT

BGL_FILES  = ["bgl_train.log", "bgl_test_normal.log", "bgl_test_abnormal.log"]
HDFS_FILES = ["train.log", "test_normal.log", "test_abnormal.log"]


def _check_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'huggingface_hub' is not installed.\n"
            "Install it with:  pip install huggingface_hub\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def _hf_download(filename: str, local_path: Path) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = hf_hub_download(
            repo_id=HF_ASSETS_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_path.parent),
        )
        downloaded = Path(tmp)
        if downloaded != local_path:
            downloaded.rename(local_path)
        return local_path.exists()
    except Exception as exc:
        print(f"ERROR: download failed ({filename}): {exc}", file=sys.stderr)
        return False


def _extract_zip(zip_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(out_dir))
    zip_path.unlink(missing_ok=True)
    return sum(1 for f in out_dir.iterdir() if f.is_file())


def _bgl_present() -> bool:
    return BGL_SPLIT_DIR.exists() and all((BGL_SPLIT_DIR / f).is_file() for f in BGL_FILES)


def _hdfs_present() -> bool:
    return HDFS_SPLIT_DIR.exists() and all((HDFS_SPLIT_DIR / f).is_file() for f in HDFS_FILES)


def download_bgl() -> bool:
    if _bgl_present():
        print(f"BGL split logs already present in {BGL_SPLIT_DIR} — skipping.")
        return True
    print(f"Downloading BGL split logs (~700 MB) → {BGL_SPLIT_DIR} …")
    zip_path = LOG_ROOT / "BGL" / "bgl_raw.zip"
    if not _hf_download("bgl_raw.zip", zip_path):
        return False
    _extract_zip(zip_path, BGL_SPLIT_DIR)
    if _bgl_present():
        print(f"BGL split logs ready ({len(BGL_FILES)} files).")
        return True
    print("WARNING: extraction finished but some BGL files are missing.", file=sys.stderr)
    return False


def download_hdfs() -> bool:
    if _hdfs_present():
        print(f"HDFS split logs already present in {HDFS_SPLIT_DIR} — skipping.")
        return True
    print(f"Downloading HDFS split logs (~1.6 GB) → {HDFS_SPLIT_DIR} …")
    zip_path = HDFS_SPLIT_DIR / "hdfs_split.zip"
    HDFS_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if not _hf_download("hdfs_split.zip", zip_path):
        return False
    _extract_zip(zip_path, HDFS_SPLIT_DIR)
    if _hdfs_present():
        print(f"HDFS split logs ready ({len(HDFS_FILES)} files).")
        return True
    print("WARNING: extraction finished but some HDFS files are missing.", file=sys.stderr)
    return False


def list_status() -> None:
    print(f"Log root: {LOG_ROOT}\n")
    print("BGL split logs:")
    for f in BGL_FILES:
        p = BGL_SPLIT_DIR / f
        status = f"{p.stat().st_size // 1024 // 1024} MB" if p.exists() else "MISSING"
        print(f"  {f}: {status}")
    print("\nHDFS split logs:")
    for f in HDFS_FILES:
        p = HDFS_SPLIT_DIR / f
        status = f"{p.stat().st_size // 1024 // 1024} MB" if p.exists() else "MISSING"
        print(f"  {f}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw log split files for the CECO-LAD dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", choices=["bgl", "hdfs"], default=None,
        help="Download only this dataset. Omit to download both.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print download status and exit.",
    )
    args = parser.parse_args()

    if args.list:
        list_status()
        return

    _check_huggingface_hub()

    datasets = [args.dataset] if args.dataset else ["bgl", "hdfs"]
    ok = True
    if "bgl" in datasets:
        ok &= download_bgl()
    if "hdfs" in datasets:
        ok &= download_hdfs()

    if ok:
        print(f"\nAll requested logs are ready under {LOG_ROOT}")
        print("You can now run the dashboard — raw log messages will be accessible.")
    else:
        print("\nSome downloads failed. Check the errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
