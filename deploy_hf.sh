#!/usr/bin/env bash
# Deploy CECO-LAD to Hugging Face Spaces using the Python upload API.
# No git-lfs setup needed — huggingface_hub handles large files automatically.
#
# Usage:  bash deploy_hf.sh <hf-username> [space-name]
# Example: bash deploy_hf.sh kiSmetZz
set -euo pipefail

HF_USER="${1:-}"
SPACE_NAME="${2:-ceco-lad}"

if [[ -z "$HF_USER" ]]; then
    echo "Usage: bash deploy_hf.sh <hf-username> [space-name]"
    exit 1
fi

echo ""
echo "Deploying CECO-LAD → HF Space: ${HF_USER}/${SPACE_NAME}"
echo "Public URL after build:  https://${HF_USER}-${SPACE_NAME}.hf.space"
echo ""

pip install -q "huggingface_hub>=0.20"

python3 - "$HF_USER" "$SPACE_NAME" <<'PYEOF'
import sys, os
from pathlib import Path
from huggingface_hub import HfApi, login

hf_user   = sys.argv[1]
space_name = sys.argv[2]
repo_id   = f"{hf_user}/{space_name}"
root      = Path(__file__).parent if "__file__" in dir() else Path(".")
# When called via bash heredoc __file__ isn't set; use cwd instead
root = Path(os.getcwd())

print("Step 1/3  Log in to Hugging Face")
print("  (get your token at https://huggingface.co/settings/tokens — needs Write access)")
login()

api = HfApi()

print(f"\nStep 2/3  Creating Space '{repo_id}' (Docker, public)…")
api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
)
print("  Space ready.")

# Files/dirs to exclude from upload
IGNORE = {
    ".git", "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache",
    "dashboard/ceco_lad.db",          # local DB — too large, rebuilt on startup
    "checkpoints/bat",                 # 3.5 GB — downloaded at runtime by spaces_startup.py
    "inference_pipeline/executorch/third-party",  # 3 GB C++ source
    "inference_pipeline/executorch/cmake-out/CMakeFiles",
    "inference_pipeline/executorch/.git",
    "logs",
    "pictures",
    ".vscode", ".idea",
    "environment",
}

def should_ignore(path: Path) -> bool:
    parts = path.parts
    for ign in IGNORE:
        if ign.startswith("*"):
            if path.name.endswith(ign[1:]):
                return True
        else:
            ign_parts = Path(ign).parts
            for i in range(len(parts) - len(ign_parts) + 1):
                if parts[i:i+len(ign_parts)] == ign_parts:
                    return True
    return False

# Collect files to upload
files = []
for f in root.rglob("*"):
    if f.is_file() and not should_ignore(f.relative_to(root)):
        files.append(f)

total_mb = sum(f.stat().st_size for f in files) / 1e6
print(f"\nStep 3/3  Uploading {len(files)} files ({total_mb:.0f} MB) to HF…")
print("  Large files (checkpoints, binaries, numpy arrays) are uploaded automatically.")
print("  This may take 10–30 minutes depending on your connection speed.")
print()

# Upload in one call — HfApi handles chunking and retries
api.upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=str(root),
    ignore_patterns=[
        ".git/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "dashboard/ceco_lad.db",
        "checkpoints/bat/**",
        "inference_pipeline/executorch/third-party/**",
        "inference_pipeline/executorch/cmake-out/CMakeFiles/**",
        "inference_pipeline/executorch/.git/**",
        "logs/**",
        "pictures/**",
        ".vscode/**",
        ".idea/**",
        "environment/**",
    ],
    run_as_future=False,
)

print()
print("=" * 60)
print("  Upload complete!")
print()
print(f"  HF is building the Docker image now (~10 min).")
print(f"  Space:      https://huggingface.co/spaces/{repo_id}")
print(f"  Public URL: https://{hf_user}-{space_name}.hf.space")
print()
print("  On first visitor:")
print("    BAT checkpoints download from Google Drive (~3.5 GB, ~10 min)")
print("    Subsequent visits are instant.")
print("=" * 60)
PYEOF
