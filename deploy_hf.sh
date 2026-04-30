#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_hf.sh  —  Deploy CECO-LAD to Hugging Face Spaces (one command)
#
# Usage:
#   bash deploy_hf.sh <hf-username> [space-name]
#
# Examples:
#   bash deploy_hf.sh QinxuanShi
#   bash deploy_hf.sh QinxuanShi ceco-lad-demo
#
# What it does:
#   1. Logs you in to Hugging Face (prompts for token once)
#   2. Creates the Space if it doesn't exist
#   3. Pushes all project files to the Space
#
# After the push, HF builds the Docker image (~10 min first time).
# Your public URL will be:  https://<username>-<space-name>.hf.space
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HF_USER="${1:-}"
SPACE_NAME="${2:-ceco-lad}"

if [[ -z "$HF_USER" ]]; then
    echo "Usage: bash deploy_hf.sh <hf-username> [space-name]"
    echo "  Get your username at: https://huggingface.co/settings/profile"
    exit 1
fi

SPACE_ID="${HF_USER}/${SPACE_NAME}"
SPACE_URL="https://huggingface.co/spaces/${SPACE_ID}"
SPACE_GIT="https://huggingface.co/spaces/${SPACE_ID}.git"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  CECO-LAD  →  Hugging Face Spaces"
echo "  Space : ${SPACE_ID}"
echo "  URL   : ${SPACE_URL}"
echo "═══════════════════════════════════════════════════"
echo ""

# ── 1. Log in ─────────────────────────────────────────────────────────────────
echo "Step 1/4  Logging in to Hugging Face…"
echo "  (get your token at https://huggingface.co/settings/tokens)"
huggingface-cli login

# ── 2. Create the Space ───────────────────────────────────────────────────────
echo ""
echo "Step 2/4  Creating Space '${SPACE_ID}' (Docker, public)…"
python3 - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(
        repo_id="${SPACE_ID}",
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    print("  Space ready.")
except Exception as e:
    print(f"  Note: {e}")
PYEOF

# ── 3. Set up git remote and LFS ──────────────────────────────────────────────
echo ""
echo "Step 3/4  Configuring git…"

# Make sure git-lfs is installed (needed for *.npy files > 5 MB)
if ! git lfs version &>/dev/null 2>&1; then
    echo "  Installing git-lfs…"
    sudo apt-get install -y git-lfs 2>/dev/null || brew install git-lfs 2>/dev/null || {
        echo "  ERROR: git-lfs not found. Install it: https://git-lfs.com"
        exit 1
    }
fi
git lfs install --local

# Track large numpy / yaml result files with LFS
git lfs track "outputs/**/*.npy"   2>/dev/null || true
git lfs track "outputs/**/*.yaml"  2>/dev/null || true
git add .gitattributes 2>/dev/null || true

# Add or update the HF remote
if git remote get-url hf &>/dev/null 2>&1; then
    git remote set-url hf "${SPACE_GIT}"
else
    git remote add hf "${SPACE_GIT}"
fi

# ── 4. Commit and push ────────────────────────────────────────────────────────
echo ""
echo "Step 4/4  Pushing to Hugging Face Spaces…"
echo "  (first push may take a few minutes — outputs/*.npy files are uploaded via LFS)"

git add .
git diff --cached --quiet || git commit -m "deploy: CECO-LAD dashboard"

git push hf main --force

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Deployment complete!"
echo ""
echo "  Your Space is building now (~10 min first time)."
echo "  Public URL:  ${SPACE_URL}"
echo ""
echo "  On first visitor:"
echo "    • BAT checkpoints download from Google Drive (~3.5 GB, ~10 min)"
echo "    • Subsequent visits are instant"
echo "═══════════════════════════════════════════════════"
