# CECO-LAD

**Cloud-Edge Collaborative Log Anomaly Detection**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![HF Demo](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-yellow)](https://kismetzz-ceco-lad.hf.space/)

**[🤗 Try the live demo on Hugging Face Spaces](https://kismetzz-ceco-lad.hf.space/)**

---

## How It Works

CECO-LAD detects anomalies in system logs using a hybrid Cloud-Edge collaboration pipeline:

- **Q-BAT edge models** run locally on resource-constrained edge devices.
- Only the **most uncertain lines** are forwarded to the **BAT cloud ensemble** for more accurate reevaluation.

<p align="center">
  <img src="pictures/framework.png" width="700">
</p>

### Models

| Model     | Where      | What                                                                              |
| --------- | ---------- | --------------------------------------------------------------------------------- |
| **BAT**   | Cloud      | 81 EM-AT models with varied hyperparameters and bootstrap-resampled training data |
| **Q-BAT** | Edge (CPU) | BAT checkpoints quantized to A8W4 format via ExecuTorch                           |
| **EM-AT** | Cloud      | EM-GMM Enhanced Anomaly Transformer — base learner                                |

### Framework Overview

System logs are first generated from diverse servers and applications and collected by distributed log collection servers. A log processing pipeline then parses raw log messages into structured formats, partitions them into sequences, and converts them into feature matrices as input for the anomaly detector.

For deployment in heterogeneous cloud-edge environments, CECO-LAD adopts a collaborative inference strategy: the BAT model is hosted on the cloud server, while the lightweight Q-BAT model runs on resource-constrained edge devices. A Mahalanobis distance-based routing policy enables collaborative anomaly analysis, forwarding hard cases from the edge to the cloud for more accurate prediction. Finally, the Green-LADE method is integrated to assess computational resource efficiency, quantifying the trade-off between resource consumption and detection capability across cloud and edge deployments.

---

## Repository Structure

```
CECO-LAD/
├── start.py                   # One-command local setup + dashboard launch
├── run.py                     # CLI: train / eval / infer / convert / download
│
├── ceco_core/                 # Shared model library (EM-AT, energy scoring, voting)
├── training_pipeline/         # BAT ensemble training and threshold evaluation
├── edge_pipeline/             # Q-BAT quantization (BAT → ExecuTorch .pte)
├── inference_pipeline/        # Full edge → routing → cloud pipeline
│   └── executorch/            # Pre-built ExecuTorch 0.5.0 runtime (downloaded by start.py)
├── dashboard/                 # FastAPI + single-page web dashboard
│
├── configs/                   # Training and inference YAML configs per dataset
├── data/                      # Pre-processed dataset files (included in repo)
├── checkpoints/               # Model weights — downloaded by start.py
├── outputs/                   # Inference results — generated at runtime
│
└── tools/
    ├── download_checkpoints.py  # Downloads BAT / Q-BAT checkpoints
    └── download_data.py         # Downloads ExecuTorch + raw logs
```

---

## Quick Start — Dashboard

Try the **[live demo on Hugging Face Spaces](https://kismetzz-ceco-lad.hf.space/)** — no setup needed.

For local use:

```bash
# 1. Clone
git clone https://github.com/kiSmetZz28/CECO-LAD.git
cd CECO-LAD

# 2. Set up the environment
conda create -yn ceco-lad python=3.10.0
conda activate ceco-lad
pip install torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r environment/cloud/requirements.txt
pip install -e .
pip install gdown huggingface_hub

# 3. Download assets and launch
python start.py
```

Open **http://localhost:8765**. A built-in **? Help** button guides you through all features.

> `start.py` automatically downloads all required assets (~4 GB) on first run. On first launch the **Database** indicator shows **Loading** while log data is imported (~1–2 min).

---

## Full Setup

All commands run from the project root. Two Conda environments are needed:

| Environment | Purpose                                                      |
| ----------- | ------------------------------------------------------------ |
| `ceco-lad`  | Dashboard, evaluation, edge inference (CPU)                  |
| `hybrid`    | Training (GPU recommended) and cloud BAT inference           |

### Step 1 — Set up environments

```bash
# ceco-lad (CPU — for dashboard and edge inference)
conda create -yn ceco-lad python=3.10.0 && conda activate ceco-lad
pip install torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r environment/cloud/requirements.txt && pip install -e . && pip install gdown huggingface_hub

# hybrid (GPU — for training and cloud inference)
conda create -yn hybrid python=3.10.0 && conda activate hybrid
pip install -r environment/cloud/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

> ExecuTorch **0.5.0** ([docs](https://docs.pytorch.org/executorch/0.5/)) is downloaded and its Python bindings installed automatically by `start.py` — no manual compilation needed. Linux/macOS only; Windows users: use WSL2.

### Step 2 — Download pre-trained models

```bash
# Download everything (ExecuTorch + Q-BAT + raw logs + BAT checkpoints for all datasets)
python start.py --setup-only

# Download BAT checkpoints for a specific dataset only
python tools/download_checkpoints.py --dataset bgl    # BGL only
python tools/download_checkpoints.py --dataset hdfs   # HDFS only
python tools/download_checkpoints.py --dataset os     # OpenStack only

# Download Q-BAT edge models only (~220 MB total, no BAT needed for edge-only inference)
python tools/download_checkpoints.py --type qbat
```

### Step 3 — Run inference

Pre-computed thresholds for all three datasets are included in the repository. Runs the full pipeline: edge scan → routing → cloud re-check → final prediction (~5–15 min per dataset on CPU).

```bash
conda activate ceco-lad
python run.py infer os
python run.py infer bgl
python run.py infer hdfs
```

**Run edge only** (no cloud re-check):
```bash
conda activate ceco-lad
python -m inference_pipeline.run --config configs/inference/os.yaml --edge-only
```

**Run cloud re-check only** (requires edge outputs to already exist):
```bash
conda activate hybrid
python dashboard/cloud_runner.py --config configs/inference/os.yaml
```

---

## Advanced Options

### Train from scratch

```bash
conda activate hybrid   # GPU recommended — 2–8 hours per dataset
python run.py train os && python run.py train bgl && python run.py train hdfs
```

Runs a hyperparameter sweep — 81 models per dataset. Then regenerate thresholds:

```bash
conda activate ceco-lad
python run.py eval os && python run.py eval bgl && python run.py eval hdfs
```

### Convert to edge models

```bash
conda activate ceco-lad
python run.py convert os && python run.py convert bgl && python run.py convert hdfs
```

Applies A8W4 quantization and exports `.pte` files to `checkpoints/qbat/{dataset}/`. Skip if you downloaded Q-BAT checkpoints in Step 2.

---

---

## Results

### OpenStack

<p align="center">
  <img src="pictures/openstack_results.png" width="700">
</p>

### HDFS

<p align="center">
  <img src="pictures/hdfs_results.png" width="700">
</p>

### BGL

<p align="center">
  <img src="pictures/bgl_results.png" width="700">
</p>
