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

System logs are first generated from diverse servers and applications and collected by distributed log collection servers. A log processing pipeline then parses raw log messages into structured formats, partitions them into sequences, and converts them into feature matrices as input for the anomaly detector. For deployment in heterogeneous cloud-edge environments, CECO-LAD adopts a collaborative inference strategy: the BAT model is hosted on the cloud server, while the lightweight Q-BAT model runs on resource-constrained edge devices. A Mahalanobis distance-based routing policy enables collaborative anomaly analysis, forwarding hard cases from the edge to the cloud for more accurate prediction. Finally, the Green-LADE method is integrated to assess computational resource efficiency, quantifying the trade-off between resource consumption and detection capability across cloud and edge deployments.

### Paper ↔ Code Mapping

| Paper component                                            | Code location                                                                                                  |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Log preprocessing (raw logs → feature matrix)              | [ceco_core/data/](ceco_core/data/)                                                                             |
| Enhanced Anomaly Transformer (EM-AT) base learner          | [ceco_core/models/](ceco_core/models/)                                                                         |
| BAT — cloud bagging ensemble (training)                    | [training_pipeline/](training_pipeline/)                                                                       |
| Q-BAT — edge quantized ensemble (A8W4 + ExecuTorch export) | [quantization/qbat_export.py](quantization/qbat_export.py)                                                     |
| Q-BAT edge inference                                       | [inference_pipeline/qbat_edge.py](inference_pipeline/qbat_edge.py)                                             |
| Mahalanobis distance-based routing policy                  | [inference_pipeline/routing.py](inference_pipeline/routing.py)                                                 |
| BAT cloud re-prediction on routed samples                  | [inference_pipeline/bat_cloud.py](inference_pipeline/bat_cloud.py)                                             |
| Cloud–edge collaborative orchestration                     | [inference_pipeline/run.py](inference_pipeline/run.py), [dashboard/cloud_runner.py](dashboard/cloud_runner.py) |

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
│   └── qbat_export.py             #    A8W4 quantize, export to ExecuTorch .pte
│
├── inference_pipeline/            # 3. Edge → routing → cloud
│   ├── qbat_edge.py               #    Q-BAT inference via ExecuTorch
│   ├── routing.py                 #    Mahalanobis routing (uncertain → cloud)
│   ├── bat_cloud.py               #    Full BAT ensemble re-prediction
│   ├── run.py                     #    Orchestrator (spawns cloud env as subprocess)
│   └── executorch/                #    Pre-built ExecuTorch runtime
│
├── dashboard/                     # 4. Front-end UI for visualization
│
│ ── Configuration & data ─────────────────────────────────────────────────
├── configs/
│   ├── training/{bgl,hdfs,os}.yaml
│   └── inference/{bgl,hdfs,os}.yaml
│
├── data/                          # Pre-processed event sequences
├── outputs/                       # Thresholds, predictions
├── checkpoints/                   # bat/*.pth + qbat/*.pte — downloaded by start.py
├── logs/                          # Training & inference logs
│
├── environment/                   # Python dependency lists
│   ├── cloud/requirements.txt     #   Training, eval, cloud inference, dashboard
│   └── edge/requirements.txt      #   ExecuTorch edge inference
│
│ ── Tooling ──────────────────────────────────────────────────────────────
└── tools/
    ├── download_checkpoints.py    # Fetch BAT / Q-BAT checkpoints
    ├── download_data.py           # Fetch ExecuTorch runtime + raw logs
    └── deploy/                    # Maintainer-only — Hugging Face Space deployment
```

---

## Full Setup

All commands run from the project root. CECO-LAD uses **two Conda environments**, one for each inference tier:

| Environment      | Tier      | Stack                              | What runs in it                                                                                                                                     |
| ---------------- | --------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ceco-lad-edge`  | **Edge**  | PyTorch 2.6 (CPU) + ExecuTorch 0.5 | Dashboard, Q-BAT edge inference (`.pte` via ExecuTorch), pipeline orchestration. CPU is sufficient.                                                 |
| `ceco-lad-cloud` | **Cloud** | PyTorch 2.4 + CUDA 12.4            | BAT ensemble training (81 models) and cloud re-check inference. GPU strongly recommended; the inference pipeline launches this env as a subprocess. |

**Why two environments?** Edge and cloud have different runtime needs. Edge uses ExecuTorch (compact, CPU-only, runs `.pte` quantized models), while cloud uses full-precision PyTorch with CUDA. Splitting them keeps each install minimal and avoids version conflicts between ExecuTorch and CUDA PyTorch.

### Step 1 — Set up environments

#### Edge environment (`ceco-lad-edge`)

```bash
conda create -yn ceco-lad-edge python=3.10.0
conda activate ceco-lad-edge
pip install -r environment/edge/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

ExecuTorch **0.5.0** ([docs](https://docs.pytorch.org/executorch/0.5/)) and its bundled `torchao` build are downloaded and installed automatically by `start.py` — no manual compilation needed (they carry PEP 440 local version labels and are not on PyPI, hence commented out in `requirements.txt`). **Linux/macOS only**; Windows users: use WSL2.

#### Cloud environment (`ceco-lad-cloud`)

```bash
conda create -yn ceco-lad-cloud python=3.10.0
conda activate ceco-lad-cloud
pip install -r environment/cloud/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

> Edge and cloud install **different** requirements files and PyTorch builds (CPU vs CUDA), so they must be separate envs. The inference pipeline also relies on this split: edge orchestrates and spawns cloud inference as a subprocess pointed at the cloud env's interpreter.

**Verify both environments exist:**

```bash
conda env list   # should list both 'ceco-lad-edge' and 'ceco-lad-cloud'
```

### Step 2 — Download assets and launch the dashboard

`start.py` downloads ExecuTorch, Q-BAT, raw logs, and BAT checkpoints (skipping any that already exist), then launches the dashboard at **http://localhost:8765**:

```bash
conda activate ceco-lad-edge
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
conda activate ceco-lad-edge
python run.py infer os
python run.py infer bgl
python run.py infer hdfs
```

---

## Advanced Options

### Run edge or cloud only

```bash
# Edge inference only (no cloud re-check)
conda activate ceco-lad-edge
python -m inference_pipeline.run --config configs/inference/os.yaml --edge-only

# Cloud re-check only (requires edge outputs to already exist)
conda activate ceco-lad-cloud
python dashboard/cloud_runner.py --config configs/inference/os.yaml
```

### Train from scratch

```bash
conda activate ceco-lad-cloud
python run.py train os
python run.py train bgl
python run.py train hdfs
```

Runs a hyperparameter sweep — 81 models per dataset. Then regenerate thresholds:

```bash
conda activate ceco-lad-edge
python run.py eval os
python run.py eval bgl
python run.py eval hdfs
```

### Convert to edge models

```bash
conda activate ceco-lad-edge
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
