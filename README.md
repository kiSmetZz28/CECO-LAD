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

| Model     | Where | What                                                                                                        |
| --------- | ----- | ----------------------------------------------------------------------------------------------------------- |
| **BAT**   | Cloud | Bagging-style ensemble of 81 EM-AT models with varied hyperparameters and bootstrap-resampled training data |
| **Q-BAT** | Edge  | Bagging-style ensemble of 3 quantized EM-AT base models                                                     |

### Framework Overview

System logs are first generated from diverse servers and applications and collected by distributed log collection servers. A log processing pipeline then parses raw log messages into structured formats, partitions them into sequences, and converts them into feature matrices as input for the anomaly detector.

For deployment in heterogeneous cloud-edge environments, CECO-LAD adopts a collaborative inference strategy: the BAT model is hosted on the cloud server, while the lightweight Q-BAT model runs on resource-constrained edge devices. A Mahalanobis distance-based routing policy enables collaborative anomaly analysis, forwarding hard cases from the edge to the cloud for more accurate prediction. Finally, the Green-LADE method is integrated to assess computational resource efficiency, quantifying the trade-off between resource consumption and detection capability across cloud and edge deployments.

---

## Repository Structure

Two entry points cover the typical local workflow: `start.py` for one-command setup + dashboard, and `run.py` to run individual pipeline stages. Everything else falls into one of three groups — **shared library**, **pipeline stages**, or **supporting assets**.

```
CECO-LAD/
│
│ ── Entry points (run from project root) ─────────────────────────────────
├── start.py                       # Download missing assets + launch dashboard
├── run.py                         # CLI: train | eval | convert | infer | download
│
│ ── Shared library ───────────────────────────────────────────────────────
├── ceco_core/                     # Imported by every pipeline below
│   ├── models/                    #   EM-AT architecture (attention, embedding)
│   ├── data/                      #   Dataset loaders + log preprocessor
│   └── utils/                     #   Energy scoring, voting, config I/O, metrics
│
│ ── Pipeline stages ──────────────────────────────────────────────────────
├── training_pipeline/             # 1. Train BAT ensemble (81 EM-AT models)
│   ├── train.py                   #    Hyperparameter sweep
│   ├── solver.py                  #    Anomaly-Transformer minimax training loop
│   └── evaluate.py                #    Per-model F1 + EM-GMM thresholds
│
├── quantization/                  # 2. Convert BAT → Q-BAT
│   └── convert.py                 #    A8W4 quantize, export to ExecuTorch .pte
│
├── inference_pipeline/            # 3. Edge → routing → cloud
│   ├── edge_agent.py              #    Q-BAT inference via ExecuTorch
│   ├── routing.py                 #    Mahalanobis routing (uncertain → cloud)
│   ├── cloud_expert.py            #    Full BAT ensemble re-prediction
│   ├── run.py                     #    Orchestrator (spawns cloud env as subprocess)
│   └── executorch/                #    Pre-built ExecuTorch 0.5.0 runtime (downloaded)
│
├── dashboard/                     # 4. FastAPI backend + single-page web UI
│   ├── app.py                     #    REST API, SSE log stream, in-RAM BAT cache
│   ├── index.html                 #    Vanilla JS + Chart.js front-end
│   ├── db.py + ingest.py          #    SQLite ingestion / queries for raw logs
│   ├── cloud_runner.py            #    Cloud-env subprocess (called from edge env)
│   ├── demo_runner.py             #    All-Python pipeline used inside the container
│   └── bat_predict.py             #    Single-session BAT scorer
│
│ ── Configuration & data ─────────────────────────────────────────────────
├── configs/
│   ├── training/{bgl,hdfs,os}.yaml
│   └── inference/{bgl,hdfs,os}.yaml
│
├── data/                          # Pre-processed event sequences (in repo)
├── checkpoints/                   # bat/*.pth + qbat/*.pte (downloaded by start.py)
├── outputs/                       # Predictions, energy matrices, thresholds (runtime)
├── logs/                          # Training & inference log files (runtime)
├── tests/                         # Pytest unit tests
│
├── environment/                   # Python dependency lists
│   ├── cloud/requirements.txt     #   Training, eval, cloud inference, dashboard
│   └── edge/requirements.txt      #   Extra packages for ExecuTorch edge inference
│
│ ── Tooling ──────────────────────────────────────────────────────────────
└── tools/
    ├── download_checkpoints.py    # Fetch BAT / Q-BAT checkpoints
    ├── download_data.py           # Fetch ExecuTorch runtime + raw logs
    └── deploy/                    # Maintainer-only — Hugging Face Space deployment
        ├── deploy_hf.py           #   Push repo to a HF Space
        ├── upload_assets.py       #   Upload large assets to HF dataset repo
        └── spaces_startup.py      #   Container entrypoint (run by Dockerfile CMD)
```

> **Local users only need:** `start.py`, `run.py`, and the dashboard.
> `tools/deploy/` and the top-level `Dockerfile` are for publishing the live demo to Hugging Face Spaces — they don't affect local runs.

---

## Full Setup

All commands run from the project root. CECO-LAD uses **two Conda environments**, one for each inference tier:

| Environment | Tier      | Stack                        | What runs in it                                                                                                                                     |
| ----------- | --------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ceco-lad`  | **Edge**  | PyTorch 2.4 + ExecuTorch 0.5 | Dashboard, Q-BAT edge inference (`.pte` via ExecuTorch), pipeline orchestration. CPU is sufficient.                                                 |
| `hybrid`    | **Cloud** | PyTorch 2.4 + CUDA 12.4      | BAT ensemble training (81 models) and cloud re-check inference. GPU strongly recommended; the inference pipeline launches this env as a subprocess. |

**Why two environments?** Edge and cloud have different runtime needs. Edge uses ExecuTorch (compact, CPU-only, runs `.pte` quantized models), while cloud uses full-precision PyTorch with CUDA. Splitting them keeps each install minimal and avoids version conflicts between ExecuTorch and CUDA PyTorch.

### Step 1 — Set up environments

#### Edge environment (`ceco-lad`) — required for the dashboard

```bash
conda create -yn ceco-lad python=3.10.0
conda activate ceco-lad
pip install -r environment/cloud/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

> The CUDA build of PyTorch is also used here — it works on CPU-only machines (no GPU acceleration, but the install succeeds and the dashboard runs fine).

ExecuTorch **0.5.0** ([docs](https://docs.pytorch.org/executorch/0.5/)) is downloaded and its Python bindings installed automatically by `start.py` — no manual compilation needed. **Linux/macOS only**; Windows users: use WSL2.

#### Cloud environment (`hybrid`) — required for training and full inference

```bash
conda create -yn hybrid python=3.10.0
conda activate hybrid
pip install -r environment/cloud/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

> The two environments install the same requirements file with the same index URL. They are kept separate so each can evolve independently (e.g. you can add GPU-only debug tools to `hybrid` without polluting `ceco-lad`).

**Verify both environments exist:**

```bash
conda env list   # should list both 'ceco-lad' and 'hybrid'
```

### Step 2 — Download assets and launch the dashboard

`start.py` downloads ExecuTorch, Q-BAT, raw logs, and BAT checkpoints (skipping any that already exist), then launches the dashboard at **http://localhost:8765**:

```bash
conda activate ceco-lad
python start.py                # download everything + launch dashboard
python start.py --setup-only   # download only, do not launch
python start.py --no-bat       # skip BAT checkpoints (~3.5 GB × dataset)
python start.py --status       # show what is present / missing, then exit
```

A built-in **? Help** button in the dashboard guides you through all features. On first launch the **Database** indicator shows **Loading** while log data is imported.

To download checkpoints for a specific dataset without launching the dashboard:

```bash
python tools/download_checkpoints.py --dataset bgl    # BGL only
python tools/download_checkpoints.py --dataset hdfs   # HDFS only
python tools/download_checkpoints.py --dataset os     # OpenStack only
python tools/download_checkpoints.py --type qbat      # Q-BAT edge models only (~220 MB)
```

### Step 3 — Run inference from the CLI

Pre-computed thresholds for all three datasets are included in the repository. Runs the full pipeline: edge scan → routing → cloud re-check → final prediction.

```bash
conda activate ceco-lad
python run.py infer os
python run.py infer bgl
python run.py infer hdfs
```

---

## Advanced Options

### Run edge or cloud only

```bash
# Edge inference only (no cloud re-check)
conda activate ceco-lad
python -m inference_pipeline.run --config configs/inference/os.yaml --edge-only

# Cloud re-check only (requires edge outputs to already exist)
conda activate hybrid
python dashboard/cloud_runner.py --config configs/inference/os.yaml
```

### Train from scratch

```bash
conda activate hybrid
python run.py train os
python run.py train bgl
python run.py train hdfs
```

Runs a hyperparameter sweep — 81 models per dataset. Then regenerate thresholds:

```bash
conda activate ceco-lad
python run.py eval os
python run.py eval bgl
python run.py eval hdfs
```

### Convert to edge models

```bash
conda activate ceco-lad
python run.py convert os
python run.py convert bgl
python run.py convert hdfs
```

Applies A8W4 quantization and exports `.pte` files to `checkpoints/qbat/{dataset}/`. Skip if you downloaded Q-BAT checkpoints in Step 2.

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
