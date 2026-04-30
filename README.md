---
title: CECO-LAD
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Cloud-Edge Collaborative Log Anomaly Detection demo
---

# CECO-LAD

**Cloud-Edge Collaborative Log Anomaly Detection**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What This Project Does

CECO-LAD automatically finds problems in system logs using a two-tier AI approach:

- All **Q-BAT edge models** run locally on every session in parallel — no GPU or internet needed.
- Only the **uncertain cases** are forwarded to the **BAT cloud ensemble**, where all checkpoints run simultaneously for a second opinion.
- The final prediction combines both, delivering **cloud-level accuracy at near-edge speed**.

<p align="center">
  <img src="pictures/framework.png" width="700">
</p>

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start — Dashboard](#quick-start--dashboard)
3. [Full Setup](#full-setup)
   - [Step 1 — Set up the environments](#step-1--set-up-the-environments)
   - [Step 2 — Download pre-trained models](#step-2--download-pre-trained-models)
   - [Step 3 — Generate detection thresholds](#step-3--generate-detection-thresholds)
   - [Step 4 — Run inference](#step-4--run-inference)
   - [Step 5 — (Optional) Train from scratch](#step-5--optional-train-from-scratch)
   - [Step 6 — (Optional) Convert to edge models](#step-6--optional-convert-to-edge-models)
   - [Step 7 — Run tests](#step-7--run-tests)
4. [Using the Dashboard](#using-the-dashboard)
5. [Repository Structure](#repository-structure)
6. [How It Works](#how-it-works)
7. [Results](#results)
8. [Troubleshooting](#troubleshooting)
9. [Citation](#citation)

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 30 GB free |
| GPU | Not required | NVIDIA GPU with CUDA 12.4 (speeds up training ~10×) |

> **A GPU is only needed for training (Step 5).** All other steps — evaluation, inference, and the dashboard — run on CPU.

### Operating System

| Task | Linux | macOS | Windows |
|------|-------|-------|---------|
| Training, evaluation, dashboard | ✅ | ✅ | ✅ |
| Edge inference with ExecuTorch | ✅ | ✅ | ⚠️ Use WSL2 |

### Prerequisites: Conda and Git

<details>
<summary><b>Install Conda (click to expand)</b></summary>

1. Go to https://docs.conda.io/en/latest/miniconda.html
2. Download and run the installer for your OS
3. Restart your terminal
4. Verify: `conda --version`

</details>

<details>
<summary><b>Install Git (click to expand)</b></summary>

- **Linux:** `sudo apt install git`
- **macOS:** `xcode-select --install`
- **Windows:** https://git-scm.com/download/win

Verify: `git --version`

</details>

---

## Quick Start — Dashboard

The dashboard lets you explore the data pipeline, view pre-computed results, and run new inference — all from your browser. The processed dataset files are already included in this repository, so no extra data download is required.

```bash
# 1. Clone the repository
git clone <repo-url> CECO-LAD
cd CECO-LAD

# 2. Create the Python environment
conda create -yn ceco-lad python=3.10.0
conda activate ceco-lad
pip install torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r environment/cloud/requirements.txt
pip install -e .

# 3. Start the dashboard
python dashboard/app.py
```

Open **http://localhost:8765** in your browser.

The dashboard automatically imports log data into a local database on first launch. The **Database** indicator in the sidebar shows the progress — once it says **Ready**, all data is fully browsable in the **Pipeline** tab.

Pre-computed inference results for OpenStack are also included, so the **Results** and **Analysis** tabs are populated immediately without running inference.

> **To run new inference from the dashboard**, complete the Full Setup below first (models must be downloaded or trained, and ExecuTorch must be installed in the `ceco-lad` environment).

---

## Full Setup

All commands are run from the **project root** (`CECO-LAD/`).

The project uses two Conda environments:

| Environment | Used for |
|-------------|----------|
| `ceco-lad` | Training, evaluation, dashboard, edge inference (requires ExecuTorch) |
| `hybrid` | Cloud GPU inference (optional — only needed if running cloud re-check) |

> These names match the defaults in `dashboard/app.py`. If you use different names, update `EDGE_PYTHON` and `CLOUD_PYTHON` at the top of `dashboard/app.py`.

---

### Step 1 — Set up the environments

#### Main environment (`ceco-lad`)

```bash
conda create -yn ceco-lad python=3.10.0
conda activate ceco-lad

# GPU training (recommended if you have an NVIDIA GPU):
pip install -r environment/cloud/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

# CPU only (sufficient for evaluation, inference, and the dashboard):
pip install torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r environment/cloud/requirements.txt

pip install -e .
```

**Verify:**
```bash
python -c "import torch; print('PyTorch', torch.__version__, '· CUDA:', torch.cuda.is_available())"
```

#### Adding ExecuTorch for edge inference

ExecuTorch is required to run Q-BAT models locally. Install it into the `ceco-lad` environment.

> **Linux and macOS only.** Windows users: use WSL2.

```bash
# Clone ExecuTorch into the project (one-time setup)
cd inference_pipeline
git clone --branch release/0.5 https://github.com/pytorch/executorch.git
cd executorch
git submodule sync && git submodule update --init

# Install ExecuTorch into ceco-lad (takes 5–15 minutes)
conda activate ceco-lad
python install_requirements.py

# Install remaining edge dependencies
cd ../..
pip install -r environment/edge/requirements.txt
```

**Verify:**
```bash
python -c "from executorch.runtime import Runtime; print('ExecuTorch OK')"
```

#### Cloud GPU environment (`hybrid`, optional)

Only needed if running the cloud re-check phase of inference on a GPU machine.

```bash
conda create -yn hybrid python=3.10.0
conda activate hybrid
pip install -r environment/cloud/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

---

### Step 2 — Download pre-trained models

The processed dataset files (`data/`) are already in this repository. You only need to download the model checkpoints.

```bash
conda activate ceco-lad

# Download all datasets at once
python run.py download

# Or download one dataset at a time
python run.py download os
python run.py download bgl
python run.py download hdfs
```

This saves:
- `.pth` files (full-precision BAT models) → `checkpoints/bat/{dataset}/`
- `.pte` files (quantized Q-BAT models) → `checkpoints/qbat/{dataset}/`

---

### Step 3 — Generate detection thresholds

This step evaluates the BAT ensemble on the training set to compute per-model anomaly thresholds. **This must be run before inference** — the threshold files are required by Step 4.

```bash
conda activate ceco-lad

python run.py eval os
python run.py eval bgl
python run.py eval hdfs
```

**What it does:**
1. Loads all downloaded BAT checkpoints for the dataset
2. Runs each model on the test set to compute anomaly energy scores
3. Fits a Gaussian mixture model (GMM) on the training energy distribution
4. Sets the threshold at the boundary of the normal cluster
5. Writes `outputs/{dataset}/thresholds_cloud.yaml`

**Output:** Prints accuracy, precision, recall, and F-score for three voting strategies — `majority`, `at-least-one`, and `consensus`.

---

### Step 4 — Run inference

Runs the full collaborative pipeline from edge scan to final prediction.

```bash
conda activate ceco-lad

python run.py infer os
python run.py infer bgl
python run.py infer hdfs
```

**Prerequisite:** `outputs/{dataset}/thresholds_cloud.yaml` must exist (from Step 3).

**The four stages:**

| Stage | What happens |
|-------|-------------|
| **Edge scan** | All Q-BAT models score every test session **simultaneously** (one thread per model) |
| **Routing** | The most uncertain sessions (10% by default) are identified via Mahalanobis distance and forwarded to the cloud |
| **Cloud re-check** | All BAT checkpoints re-score the routed sessions **simultaneously** (thread pool, separate CUDA stream per worker) |
| **Merge** | Edge and cloud predictions are combined; final metrics are printed |

**Output files** written to `outputs/{dataset}/`:

| File | Contents |
|------|----------|
| `ground_truth.npy` | True labels — `[N]` binary array |
| `edge_preds.npy` | Edge AI predictions — `[N]` binary array |
| `hybrid_preds.npy` | Final merged predictions — `[N]` binary array |
| `energy_matrix.npy` | Per-model anomaly energy scores — `[N, 3]` |
| `routed_indices.npy` | Indices of sessions forwarded to the cloud |
| `thresholds_edge.yaml` | Per-model edge detection thresholds |
| `thresholds_cloud.yaml` | Per-model cloud detection thresholds |

---

### Step 5 — (Optional) Train from scratch

Trains a full ensemble of BAT models using a hyperparameter sweep. Requires a GPU and takes 2–8 hours per dataset.

```bash
conda activate ceco-lad   # or hybrid for GPU

python run.py train os
python run.py train bgl
python run.py train hdfs
```

**What it does:** For each combination of `(num_epochs, k, e_layer_num, batch_size)` defined in `configs/training/{dataset}.yaml`, it bootstraps the training data and trains one EM-AT model. Each run uses a deterministic seed derived from the hyperparameter tuple, making results fully reproducible.

Checkpoints are saved to `checkpoints/bat/{dataset}/` with filenames like:
```
Openstack_e10_k5_l3_b64_checkpoint.pth
```

After training, run Step 3 to compute thresholds from the new checkpoints.

---

### Step 6 — (Optional) Convert to edge models

Quantizes full-precision BAT checkpoints into lightweight Q-BAT models for edge deployment.

```bash
conda activate ceco-lad   # must have ExecuTorch installed

python run.py convert os
python run.py convert bgl
python run.py convert hdfs
```

**What it does:** Applies A8W4 quantization (8-bit activations, 4-bit weights) to selected BAT checkpoints and exports them as ExecuTorch `.pte` files. These files run on CPU without a GPU.

Output: `checkpoints/qbat/{dataset}/`

> Skip this step if you downloaded pre-trained Q-BAT checkpoints in Step 2.

---

### Step 7 — Run tests

```bash
conda activate ceco-lad

python run.py test
# or:
python -m pytest tests/ -v
```

25 unit tests covering anomaly energy scoring, ensemble voting, GMM thresholding, and binary prediction.

---

## Using the Dashboard

The dashboard is a browser-based interface showing the complete data pipeline from raw log files to anomaly predictions.

### Start

```bash
conda activate ceco-lad
python dashboard/app.py
```

Open **http://localhost:8765**.

### Overview

**Pipeline banner** — a strip at the top showing all 7 stages: Raw Logs → Parse → Sessions → Edge AI → Routing → Cloud AI → Result. Each stage lights up green as data becomes available. Click any stage to jump to the relevant tab.

**Sidebar** — three controls:
- **Choose Dataset** — select OpenStack, BGL, or HDFS
- **Detection Settings** — adjust cloud routing rate (default 10%) and uncertainty metric
- **Run Analysis** — start / stop inference

**Four tabs:**

| Tab | What it shows |
|-----|---------------|
| **Pipeline** | Side-by-side view of (1) raw log lines, (2) processed event sequences, and (3) inference predictions. Click any session row to see its source logs and full prediction breakdown. |
| **Results** | Accuracy, precision, recall, and F-score cards. Includes the anomaly timeline — a pixel strip showing every session coloured by its prediction. |
| **Analysis** | Per-model anomaly energy curves (used to detect anomalies), and a routing pie chart showing what fraction of sessions went to the cloud. |
| **System** | Per-model detection thresholds (calibrated during evaluation) and run log files. |

### Database loading

On first launch, the dashboard automatically imports the raw log files and processed sessions into a local SQLite database in the background. The **Database** status indicator in the sidebar tracks progress:

- **Importing…** — data is loading (takes 1–5 minutes depending on dataset size)
- **Ready** — all data is available for browsing

The processed session files from `data/OpenStack/`, `data/BGL/`, and `data/HDFS/` load quickly. The raw log files from `~/Desktop/Log Data/` take longer and are optional — if they are not present, the left column of the Pipeline tab will be empty, but sessions and results will still work.

### Running inference from the dashboard

1. Make sure Steps 1–3 are complete (environment with ExecuTorch, downloaded checkpoints, thresholds generated)
2. Select a dataset in the sidebar
3. Optionally adjust the cloud check rate
4. Click **Start Analysis**

The console shows live output. The pipeline banner stages highlight as each phase runs. When finished, the Results and Analysis tabs update automatically.

---

## Repository Structure

```
CECO-LAD/
│
├── run.py                          # Main entry point: train / eval / convert / infer / test
├── run.sh                          # Bash shortcut (Linux/macOS)
│
├── ceco_core/                      # Shared library used by all pipelines
│   ├── models/
│   │   ├── EMAT.py                 # EM-AT: Enhanced Anomaly Transformer (base model)
│   │   ├── attn.py                 # Anomaly attention with Gaussian prior
│   │   └── embed.py                # Positional and token embeddings
│   ├── data/
│   │   ├── preprocessor.py         # Raw log lines → event ID sequences
│   │   └── loaders.py              # Sliding-window DataLoaders for all three datasets
│   └── utils/
│       ├── energy.py               # Attention-weighted anomaly energy scoring
│       ├── voting.py               # Ensemble voting (majority / at-least-one / consensus)
│       ├── metrics.py              # Point-adjusted evaluation (standard protocol)
│       ├── config.py               # YAML config loading and logging setup
│       └── random_state.py         # Deterministic seed from hyperparameter tuple
│
├── training_pipeline/              # BAT training and evaluation  [ceco-lad env]
│   ├── train.py                    # Hyperparameter sweep: trains the full ensemble
│   ├── evaluate.py                 # Loads all checkpoints, evaluates, writes thresholds
│   └── solver.py                   # Per-model train / predict / GMM threshold logic
│
├── edge_pipeline/                  # Q-BAT quantization  [ceco-lad env + ExecuTorch]
│   └── convert.py                  # Converts BAT .pth → quantized ExecuTorch .pte
│
├── inference_pipeline/             # Collaborative edge + cloud inference  [ceco-lad env]
│   ├── run.py                      # Orchestrates all four inference stages
│   ├── edge_agent.py               # Runs Q-BAT models, computes GMM thresholds
│   ├── cloud_expert.py             # Runs BAT ensemble on routed sessions
│   └── routing.py                  # Mahalanobis distance routing
│
├── dashboard/                      # Web dashboard  [ceco-lad env]
│   ├── app.py                      # FastAPI backend — API endpoints + process control
│   ├── index.html                  # Frontend — single-page app
│   ├── db.py                       # SQLite database layer
│   └── ingest.py                   # Background data import from log files
│
├── configs/
│   ├── training/
│   │   ├── bgl.yaml                # Hyperparameter sweep config for BGL
│   │   ├── hdfs.yaml               # Hyperparameter sweep config for HDFS
│   │   └── os.yaml                 # Hyperparameter sweep config for OpenStack
│   └── inference/
│       ├── bgl.yaml                # Inference config: Q-BAT + routing settings
│       ├── hdfs.yaml
│       └── os.yaml
│
├── data/                           # Pre-processed dataset files (included in repo)
│   ├── BGL/                        # bgl_train.txt, bgl_test_normal.txt, bgl_test_abnormal.txt
│   ├── HDFS/                       # hdfs_train.txt, hdfs_test_normal.txt, hdfs_test_abnormal.txt
│   └── OpenStack/                  # train.txt, test_normal.txt, test_abnormal.txt
│
├── checkpoints/                    # Model weights — downloaded or trained by you
│   ├── bat/                        # Full-precision BAT checkpoints (.pth)
│   └── qbat/                       # Quantized Q-BAT checkpoints (.pte)
│
├── outputs/                        # Inference results — generated at runtime
│   ├── bgl/
│   ├── hdfs/
│   └── os/                         # Pre-computed results for OpenStack are included
│
└── environment/
    ├── cloud/requirements.txt      # Dependencies for training and evaluation
    └── edge/requirements.txt       # Extra dependencies for ExecuTorch inference
```

---

## How It Works

### The three model types

| Model | Where it runs | What it is |
|-------|--------------|------------|
| **EM-AT** | Cloud | Enhanced Anomaly Transformer — the base learner, trained with a minimax energy loss |
| **BAT** | Cloud | Bagging Anomaly Transformer — an ensemble of EM-AT models with varied hyperparameters and bootstrap-resampled training data |
| **Q-BAT** | Edge (CPU) | Selected BAT checkpoints quantized to A8W4 format (8-bit activations, 4-bit weights) via ExecuTorch |

### Training

The training sweep explores combinations of `(num_epochs, k, e_layer_num, batch_size)` as defined in `configs/training/{dataset}.yaml`. For each combination:

1. A deterministic seed is derived from the hyperparameter tuple
2. A bootstrap resample of the training data is created using that seed
3. One EM-AT model is trained on the resample
4. The checkpoint is saved to `checkpoints/bat/{dataset}/`

After training, `run.py eval` loads all checkpoints, computes per-model energy distributions on the training set, fits a 7-component GMM to each, and writes the decision thresholds to `outputs/{dataset}/thresholds_cloud.yaml`.

### Inference

```
Stage 1 — Edge scan  [parallel]
  Load all Q-BAT models (.pte files) via ExecuTorch
  Run all models simultaneously — one thread per model, GIL released during C inference
  Each model computes anomaly energy for every test session
  Compare energy to per-model GMM threshold → binary prediction per model
  Majority vote across Q-BAT models → edge prediction array

Stage 2 — Routing
  Stack per-model energy scores into a multi-dimensional feature vector per session
  Estimate the covariance matrix from training energy scores
  Compute Mahalanobis distance: how far each session is from the normal cluster
  Route the most uncertain sessions (top 10% by distance) to the cloud

Stage 3 — Cloud re-check  [parallel]
  Run all BAT checkpoints on the routed sessions simultaneously
  Thread pool (up to 4 workers by default), each worker uses its own CUDA stream
  GPU kernels from different models can overlap on the same device
  Majority vote across all BAT models → cloud prediction for each routed session

Stage 4 — Merge and evaluate
  Replace edge predictions at routed positions with cloud predictions
  Compute: accuracy, precision, recall, F-score
  (Point-adjustment is applied: if any session in a ground-truth anomaly
   segment is detected, the whole segment is credited — standard protocol)
```

### Hyperparameter sweep structure

Configured in `configs/training/{dataset}.yaml`:

```yaml
num_epochs:  [3, 6, 10]    # number of training epochs
k:           [3, 4, 5]     # attention head parameter
e_layer_num: [3, 6, 8]     # number of encoder layers
batch_size:  [32, 64, 96]  # training batch size
```

Each combination produces one checkpoint named:
`{DATASET}_e{epochs}_k{k}_l{layers}_b{batch}_checkpoint.pth`

The inference config (`configs/inference/{dataset}.yaml`) can use any subset of these checkpoints. OpenStack inference uses 3 cloud models by default; BGL and HDFS use the full ensemble.

---

## Results

<p align="center">
  <img src="pictures/openstack_results.png" width="700">
</p>

<p align="center">
  <img src="pictures/hdfs_results.png" width="700">
</p>

<p align="center">
  <img src="pictures/bgl_results.png" width="700">
</p>

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'gdown'`**
```bash
pip install "gdown>=6.0"
```

**`ModuleNotFoundError: No module named 'executorch'`**
ExecuTorch was not installed into the active environment. With `ceco-lad` active, run:
```bash
cd inference_pipeline/executorch
python install_requirements.py
```

**`FileNotFoundError: configs/training/hdfs.yaml`**
All commands must be run from the project root, not a subdirectory:
```bash
cd /path/to/CECO-LAD
python run.py train hdfs
```

**`FileNotFoundError: outputs/os/thresholds_cloud.yaml`**
Run Step 3 (evaluate) before Step 4 (infer):
```bash
python run.py eval os
```

**`CUDA out of memory` during training**
Reduce the batch sizes in `configs/training/{dataset}.yaml`:
```yaml
batch_size: [32]
```

**Dashboard shows "Database: Empty" after a long wait**
Check that the processed data files exist in `data/OpenStack/`, `data/BGL/`, and `data/HDFS/`. If the raw log files under `~/Desktop/Log Data/` are missing, only the processed sessions will be unavailable — the rest of the dashboard still works.

**Dashboard Pipeline tab — raw log column shows no results**
The raw log files are imported in the background and may take a few minutes. Watch the **Database** status indicator in the sidebar — it changes from **Importing…** to **Ready** when the import finishes.

**Dashboard `Start Analysis` button has no effect**
The dashboard calls inference using the `ceco-lad` and `hybrid` Conda environments (configured in `dashboard/app.py`). Verify that:
1. Both environments exist: `conda env list`
2. The threshold file exists: `outputs/{dataset}/thresholds_cloud.yaml`
3. The Q-BAT checkpoints exist: `checkpoints/qbat/{dataset}/`

---

## Citation

If you use CECO-LAD in your research, please cite:

```bibtex
@article{cecolad2025,
  title   = {CECO-LAD: Cloud-Edge Collaborative Log Anomaly Detection},
  author  = {},
  journal = {},
  year    = {2025}
}
```
