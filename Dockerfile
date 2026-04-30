FROM python:3.10-slim

WORKDIR /app

# CPU-only PyTorch (saves ~1.3 GB vs the default CUDA build)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Dashboard + inference dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.100" \
    "uvicorn[standard]>=0.20" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "scikit-learn>=1.3" \
    "pyyaml>=6.0" \
    "pandas>=1.5" \
    "tqdm>=4.0" \
    "gdown>=4.6"

# ── Project source ────────────────────────────────────────────────────────────
COPY ceco_core/                         /app/ceco_core/
COPY inference_pipeline/__init__.py     /app/inference_pipeline/__init__.py
COPY inference_pipeline/cloud_expert.py /app/inference_pipeline/cloud_expert.py
COPY inference_pipeline/routing.py      /app/inference_pipeline/routing.py
COPY dashboard/app.py                   /app/dashboard/app.py
COPY dashboard/db.py                    /app/dashboard/db.py
COPY dashboard/ingest.py                /app/dashboard/ingest.py
COPY dashboard/index.html               /app/dashboard/index.html
COPY dashboard/bat_predict.py           /app/dashboard/bat_predict.py
COPY dashboard/cloud_runner.py          /app/dashboard/cloud_runner.py
COPY dashboard/demo_runner.py           /app/dashboard/demo_runner.py

# ── Static data (pre-computed results + log sequences + configs) ──────────────
# Pre-computed inference results — shown immediately before any inference runs
COPY outputs/   /app/outputs/
# Preprocessed log sequences — pipeline view + StandardScaler fitting
COPY data/      /app/data/
# YAML inference configs (training and inference)
COPY configs/   /app/configs/

# ── Download helper + startup script ─────────────────────────────────────────
# spaces_startup.py downloads BAT checkpoints from Google Drive on first launch
# (~3.5 GB, one-time; subsequent starts skip the download automatically).
COPY tools/download_checkpoints.py /app/tools/download_checkpoints.py
COPY spaces_startup.py             /app/spaces_startup.py

# ── Environment ───────────────────────────────────────────────────────────────
# Use the container's single Python for both edge and cloud inference phases
ENV EDGE_PYTHON=/usr/local/bin/python
ENV CLOUD_PYTHON=/usr/local/bin/python

# Demo mode: disable train / eval / convert; inference and predict stay enabled
ENV DEMO_MODE=1

# HF Spaces and most cloud platforms expect port 7860
ENV PORT=7860

EXPOSE 7860

# spaces_startup.py checks for checkpoints, downloads if missing, then
# launches dashboard/app.py via os.execv (same PID, correct signal handling)
CMD ["python", "spaces_startup.py"]
