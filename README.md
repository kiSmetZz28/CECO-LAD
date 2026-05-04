# CECO-LAD

**Cloud-Edge Collaborative Log Anomaly Detection**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What This Project Does

CECO-LAD automatically detects anomalies in system logs using a two-tier AI approach:

- **Q-BAT edge models** run locally on every log line in parallel — no GPU or internet needed.
- Only the **most uncertain lines** are forwarded to the **BAT cloud ensemble**, where 81 checkpoints re-score them.
- The final prediction combines both tiers, delivering **cloud-level accuracy at near-edge speed**.

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

> **A GPU is only needed for training (Step 5).** Evaluation, inference, and the dashboard all run on CPU.

> **Disk note:** Downloading BAT checkpoints requires ~3.5 GB per dataset (up to ~10.5 GB for all three). Q-BAT edge checkpoints are ~220 MB total.

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

The dashboard lets you explore the data pipeline, view pre-computed results, and run new inference — all from your browser. The processed dataset files are already included in this repository.

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/CECO-LAD.git CECO-LAD
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

**What you'll see on first launch:**

- The **Database** indicator in the sidebar will show **Loading** while log data is imported into a local database. Once it says **Ready**, all data is browsable in the **Pipeline** tab.
- Pre-computed inference results for OpenStack are included, so the **Results** and **Analysis** tabs are populated immediately.

> **To run new inference from the dashboard**, complete the Full Setup below first (models must be downloaded or trained, and ExecuTorch must be installed in the `ceco-lad` environment).

---

## Full Setup

All commands are run from the **project root** (`CECO-LAD/`).

### Two environments

The project uses two Conda environments, one for each inference tier:

| Environment | Purpose |
|-------------|---------|
| `ceco-lad` | Training, evaluation, dashboard, and edge inference (requires ExecuTorch) |
| `hybrid` | Cloud BAT inference — runs 81 checkpoints for the re-check phase |

> These names match the defaults in `dashboard/app.py`. If you use different names, update `EDGE_PYTHON` and `CLOUD_PYTHON` at the top of that file.

### Step dependencies at a glance

```
Step 1 (environments)
  └─▶ Step 2 (download models)
        └─▶ Step 3 (generate thresholds)
              └─▶ Step 4 (run inference)

Optional paths:
  Step 5 (train from scratch) ─▶ Step 3 ─▶ Step 4
  Step 6 (convert to edge)    ─▶ Step 4
```

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
# Clone ExecuTorch into the project (one-time setup, ~5 minutes to clone)
cd inference_pipeline
git clone --branch release/0.5 https://github.com/pytorch/executorch.git
cd executorch
git submodule sync && git submodule update --init

# Install ExecuTorch into ceco-lad (takes 5–15 minutes to build)
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

#### Cloud environment (`hybrid`)

Required for the cloud re-check phase of inference.

```bash
conda create -yn hybrid python=3.10.0
conda activate hybrid
pip install -r environment/cloud/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

**Verify:**
```bash
conda env list   # should show both 'ceco-lad' and 'hybrid'
```

---

### Step 2 — Download pre-trained models

> **Requires:** Step 1 complete.
> **Disk space:** ~3.5 GB per dataset for BAT checkpoints; ~220 MB total for Q-BAT edge models.

```bash
conda activate ceco-lad

# Download checkpoints for all three datasets at once
python run.py download

# Or download one dataset at a time
python run.py download os
python run.py download bgl
python run.py download hdfs

# Download only the quantized edge models (smaller, ~220 MB total)
python run.py download os qbat
```

This saves:
- `.pth` files (full-precision BAT cloud models) → `checkpoints/bat/{dataset}/`
- `.pte` files (quantized Q-BAT edge models) → `checkpoints/qbat/{dataset}/`

**Verify:**
```bash
ls checkpoints/bat/os/   # should list 81 .pth files
ls checkpoints/qbat/os/  # should list 3 .pte files
```

---

### Step 3 — Generate detection thresholds

> **Requires:** Step 2 complete (BAT checkpoints must exist).
> **Must be run before inference.**

This step evaluates the BAT ensemble on the test set to compute per-model anomaly thresholds.

```bash
conda activate ceco-lad

python run.py eval os
python run.py eval bgl
python run.py eval hdfs
```

**What it does:**
1. Loads all 81 downloaded BAT checkpoints for the dataset
2. Runs each model on the test set to compute anomaly energy scores
3. Fits a 7-component Gaussian Mixture Model (GMM) on the energy distribution
4. Sets the detection threshold at the boundary of the normal cluster
5. Writes `outputs/{dataset}/thresholds_cloud.yaml`

**Verify:**
```bash
cat outputs/os/thresholds_cloud.yaml   # should show per-model threshold values
```

---

### Step 4 — Run inference

> **Requires:** Steps 1–3 complete (`thresholds_cloud.yaml` must exist and ExecuTorch must be installed).
> **Expected runtime:** ~5–15 minutes per dataset on CPU.

Runs the full collaborative pipeline: edge scan → routing → cloud re-check → final prediction.

```bash
conda activate ceco-lad

python run.py infer os
python run.py infer bgl
python run.py infer hdfs
```

**The four stages:**

| Stage | What happens |
|-------|-------------|
| **Edge scan** | All Q-BAT models score every test log line simultaneously (one thread per model) |
| **Routing** | Per-line energy scores select the most uncertain 10% of lines via Mahalanobis distance; their feature vectors are saved for cloud processing |
| **Cloud re-check** | Routed lines are reshaped into windows and scored by all 81 BAT checkpoints simultaneously |
| **Merge** | Cloud predictions replace edge predictions at routed positions; final point-adjusted metrics are printed |

**Running only the cloud phase** (if you already have edge outputs):

```bash
conda activate hybrid
python dashboard/cloud_runner.py --config configs/inference/os.yaml
```

**Output files** written to `outputs/{dataset}/`:

| File | Contents |
|------|----------|
| `ground_truth.npy` | True labels — `[N]` binary array |
| `edge_preds.npy` | Point-adjusted edge predictions — `[N]` binary array |
| `edge_preds_raw.npy` | Raw (non-adjusted) edge predictions — `[N]` binary array |
| `hybrid_preds.npy` | Point-adjusted final predictions — `[N]` binary array |
| `energy_matrix.npy` | Per-model energy scores — `[N, n_edge_models]` |
| `routed_indices.npy` | Line indices forwarded to cloud |
| `routed_lines.npy` | Feature vectors of routed lines — `[N_routed, input_c]` |
| `cloud_preds.npy` | Cloud predictions for routed lines |
| `thresholds_edge.yaml` | Per-model edge detection thresholds |
| `thresholds_cloud.yaml` | Per-model cloud detection thresholds |

**Verify:**
```bash
ls outputs/os/   # should include hybrid_preds.npy and edge_preds.npy
```

---

### Step 5 — (Optional) Train from scratch

> **Requires:** Step 1 complete.
> **Time:** 2–8 hours per dataset. **GPU strongly recommended.**

Trains a full ensemble of 81 BAT models using a hyperparameter sweep.

```bash
conda activate ceco-lad   # or hybrid for GPU training

python run.py train os
python run.py train bgl
python run.py train hdfs
```

Each combination of `(num_epochs, k, e_layer_num, batch_size)` from `configs/training/{dataset}.yaml` produces one checkpoint trained on a deterministic bootstrap resample of the training data. Checkpoints are saved to `checkpoints/bat/{dataset}/`.

After training, run **Step 3** to compute thresholds from the new checkpoints.

---

### Step 6 — (Optional) Convert to edge models

> **Requires:** Step 1 complete with ExecuTorch installed, and BAT checkpoints available (Step 2 or Step 5).

Quantizes full-precision BAT checkpoints into lightweight Q-BAT models for edge deployment.

```bash
conda activate ceco-lad   # must have ExecuTorch installed

python run.py convert os
python run.py convert bgl
python run.py convert hdfs
```

Applies A8W4 quantization (8-bit activations, 4-bit weights) and exports `.pte` files to `checkpoints/qbat/{dataset}/`.

> Skip this step if you downloaded pre-trained Q-BAT checkpoints in Step 2.

---

### Step 7 — Run tests

```bash
conda activate ceco-lad

python run.py test
# or equivalently:
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

> **Note:** On first launch, the dashboard preloads all BAT models into RAM (up to ~3.5 GB per dataset). The **Database** indicator will show **Loading** while log data is imported. Allow 1–2 minutes before all tabs are fully populated.

### Overview

A **? Help** button in the top-right corner opens a guided introduction to all dashboard features.

**Pipeline banner** — a strip at the top showing all stages: Raw Logs → Edge AI → Routing → Cloud AI → Result. Each stage lights up as data becomes available.

**Sidebar** — three controls:
- **Choose Dataset** — select OpenStack, BGL, or HDFS
- **Detection Settings** — adjust cloud routing rate (default 10%) and uncertainty metric
- **Run Analysis** — start / stop inference

**Four tabs:**

| Tab | What it shows |
|-----|---------------|
| **Pipeline** | Side-by-side view of raw log lines and processed event sequences. Filter logs by **All / Normal / Anomaly**. Click any log row to run single-log prediction. |
| **Results** | Accuracy, precision, recall, and F-score cards with an anomaly timeline strip. |
| **Analysis** | Per-model anomaly energy curves and cloud routing breakdown. |
| **System** | Per-model detection thresholds and run log files. |

### Single-log prediction

In the **Pipeline** tab, click any raw log row to instantly run the full Edge AI → Routing → Cloud AI pipeline on that single log line. The result panel shows:

- **Edge AI** prediction (Q-BAT models)
- **Routing** decision (was the line uncertain enough to send to cloud?)
- **Cloud AI** prediction (81 BAT models)
- **Final verdict** with ground-truth comparison (when available)

When a full pipeline run exists for the dataset, predictions are looked up from saved point-adjusted results so they stay consistent with the reported metrics.

### Running inference from the dashboard

1. Complete Steps 1–3 (environment with ExecuTorch, downloaded checkpoints, thresholds generated)
2. Select a dataset in the sidebar
3. Optionally adjust the cloud check rate
4. Click **Start Analysis**

The console streams live output. The pipeline banner stages highlight as each phase completes.

---

## Repository Structure

```
CECO-LAD/
│
├── run.py                          # Main entry point: train / eval / convert / download / infer / test
├── run.sh                          # Bash shortcut (Linux/macOS)
│
├── ceco_core/                      # Shared library used by all pipelines
│   ├── models/
│   │   ├── EMAT.py                 # Enhanced Anomaly Transformer (base model)
│   │   ├── attn.py                 # Anomaly attention with Gaussian prior
│   │   └── embed.py                # Positional and token embeddings
│   ├── data/
│   │   ├── preprocessor.py         # Raw log lines → event ID sequences
│   │   └── loaders.py              # Sliding-window DataLoaders for all three datasets
│   └── utils/
│       ├── energy.py               # Attention-weighted anomaly energy scoring [B, win_size]
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
├── inference_pipeline/             # Collaborative edge + cloud inference
│   ├── run.py                      # Orchestrates all four stages  [ceco-lad env]
│   ├── edge_agent.py               # Runs Q-BAT models, saves per-line energy scores
│   ├── cloud_expert.py             # Runs BAT ensemble on routed lines  [hybrid env]
│   └── routing.py                  # Mahalanobis distance routing (line-level)
│
├── dashboard/                      # Web dashboard  [ceco-lad env]
│   ├── app.py                      # FastAPI backend — API endpoints + inference control
│   ├── index.html                  # Frontend — single-page app
│   ├── db.py                       # SQLite database layer
│   ├── ingest.py                   # Background data import from log files
│   ├── cloud_runner.py             # Cloud inference phase — runs in hybrid env
│   ├── bat_predict.py              # Single-window BAT prediction subprocess
│   └── demo_runner.py              # Container demo pipeline (HF Spaces, BAT-only)
│
├── configs/
│   ├── training/
│   │   ├── bgl.yaml                # Hyperparameter sweep config for BGL
│   │   ├── hdfs.yaml               # Hyperparameter sweep config for HDFS
│   │   └── os.yaml                 # Hyperparameter sweep config for OpenStack
│   └── inference/
│       ├── bgl.yaml                # Inference config: edge + routing + cloud settings
│       ├── hdfs.yaml
│       └── os.yaml
│
├── data/                           # Pre-processed dataset files (included in repo)
│   ├── BGL/                        # bgl_train.txt, bgl_test_normal.txt, bgl_test_abnormal.txt
│   ├── HDFS/                       # hdfs_train.txt, hdfs_test_normal.txt, hdfs_test_abnormal.txt
│   └── OpenStack/                  # train.txt, test_normal.txt, test_abnormal.txt
│
├── checkpoints/                    # Model weights — downloaded or trained by you
│   ├── bat/                        # Full-precision BAT checkpoints (.pth), ~3.5 GB per dataset
│   └── qbat/                       # Quantized Q-BAT checkpoints (.pte), ~220 MB total
│
├── outputs/                        # Inference results — generated at runtime
│   ├── bgl/
│   ├── hdfs/
│   └── os/                         # Pre-computed results for OpenStack are included
│
├── spaces_startup.py               # HF Spaces container startup (downloads assets)
├── deploy_hf.py                    # Deploys dashboard to Hugging Face Spaces
├── upload_assets.py                # Uploads checkpoints and logs to HF Hub
│
└── environment/
    ├── cloud/requirements.txt      # Dependencies for training, evaluation, and cloud inference
    └── edge/requirements.txt       # Extra dependencies for ExecuTorch edge inference
```

---

## How It Works

### The three model types

| Model | Where it runs | What it is |
|-------|--------------|------------|
| **EM-AT** | Cloud | Enhanced Anomaly Transformer — the base learner, trained with a minimax energy loss |
| **BAT** | Cloud | Bagging Anomaly Transformer — 81 EM-AT models with varied hyperparameters and bootstrap-resampled training data |
| **Q-BAT** | Edge (CPU) | Selected BAT checkpoints quantized to A8W4 format (8-bit activations, 4-bit weights) via ExecuTorch |

### Training

The training sweep explores all combinations of `(num_epochs, k, e_layer_num, batch_size)` defined in `configs/training/{dataset}.yaml`. For each combination:

1. A deterministic seed is derived from the hyperparameter tuple
2. A bootstrap resample of the training data is created using that seed
3. One EM-AT model is trained on the resample
4. The checkpoint is saved to `checkpoints/bat/{dataset}/`

After training, `run.py eval` loads all checkpoints, computes per-model energy distributions on the test set, fits a 7-component GMM, and writes decision thresholds to `outputs/{dataset}/thresholds_cloud.yaml`.

### Hyperparameter sweep structure

Each dataset has its own sweep config. Example for BGL (`configs/training/bgl.yaml`):

```yaml
num_epochs:  [3, 6, 10]    # number of training epochs
k:           [3, 4, 5]     # attention head parameter
e_layer_num: [3, 6, 8]     # number of encoder layers
batch_size:  [32, 64, 96]  # training batch size
```

3 × 3 × 3 × 3 = **81 checkpoints** per dataset, named:
`{DATASET}_e{epochs}_k{k}_l{layers}_b{batch}_checkpoint.pth`

> Note: `k` values differ between datasets (e.g., OpenStack uses `[1, 3, 5]`). Check the relevant config file for each dataset.

### Inference

```
Stage 1 — Edge scan  [ceco-lad env, parallel]
  Load all Q-BAT models (.pte files) via ExecuTorch
  Run all models simultaneously — one thread per model, GIL released during C inference
  Each model scores every test log line → per-line energy scores [N_lines, n_edge_models]
  Compare energy to per-model threshold → binary prediction per line per model
  Majority vote across Q-BAT models → edge prediction array [N_lines]

Stage 2 — Routing  [ceco-lad env]
  Stack per-model energy scores: [N_lines, n_edge_models] feature matrix
  Estimate covariance from training energy scores (Mahalanobis)
  Compute distance from the normal cluster for each line
  Route the most uncertain lines (top 10% by distance) to the cloud
  Save routed line feature vectors as routed_lines.npy [N_routed, input_c]

Stage 3 — Cloud re-check  [hybrid env, parallel]
  Reshape routed lines [N_routed, input_c] → windows [N_routed//win_size, win_size, input_c]
  Run all 81 BAT checkpoints on these windows simultaneously (thread pool)
  Each model outputs per-line energy scores → threshold → binary per line
  Majority vote across all BAT models → one cloud prediction per routed line

Stage 4 — Merge and evaluate  [hybrid env]
  Replace edge predictions at routed line positions with cloud predictions
  Apply point-adjustment: if any line in a ground-truth anomaly segment is
  detected, the whole segment is credited (standard evaluation protocol)
  Print accuracy, precision, recall, F-score for edge and hybrid results
```

### Routing granularity

Routing operates at the **line level** (individual log events), not at the window level. The edge agent produces one energy score per log line. The router selects the 10% of lines with the highest Mahalanobis distance from the normal cluster and sends their feature vectors to the cloud. The cloud reshapes groups of lines into windows for EMAT processing and returns one prediction per line.

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

**`ModuleNotFoundError: No module named 'ceco_core'`**

Run all commands from the project root and make sure the package is installed:
```bash
cd /path/to/CECO-LAD
pip install -e .
```

**`ModuleNotFoundError: No module named 'executorch'`**

ExecuTorch was not installed into the active environment. With `ceco-lad` active:
```bash
cd inference_pipeline/executorch
python install_requirements.py
```

**`FileNotFoundError: outputs/os/thresholds_cloud.yaml`**

Run Step 3 (evaluate) before Step 4 (infer):
```bash
python run.py eval os
```

**`FileNotFoundError: outputs/os/routed_lines.npy`**

The edge phase has not been run yet. Run the full pipeline first:
```bash
python run.py infer os
```
Or run only the cloud phase after the edge phase completes:
```bash
conda activate hybrid
python dashboard/cloud_runner.py --config configs/inference/os.yaml
```

**`FileNotFoundError: checkpoints/bat/os/...`**

Checkpoints have not been downloaded. Run Step 2:
```bash
python run.py download os
```

**`CUDA out of memory` during training**

Reduce batch sizes in `configs/training/{dataset}.yaml`:
```yaml
batch_size: [32]
```

**Dashboard shows "Database: Empty" after a long wait**

Check that the processed data files exist:
```bash
ls data/OpenStack/ data/BGL/ data/HDFS/
```

**Dashboard `Start Analysis` button has no effect**

Verify all prerequisites:
1. Both environments exist: `conda env list`
2. Threshold file exists: `ls outputs/{dataset}/thresholds_cloud.yaml`
3. Q-BAT checkpoints exist: `ls checkpoints/qbat/{dataset}/`
4. ExecuTorch is installed: `python -c "from executorch.runtime import Runtime; print('OK')"`

**`[cache] Skip ... Torch not compiled with CUDA enabled`**

This warning appears when the `ceco-lad` environment uses a CPU-only PyTorch build. The in-RAM model cache will not load, but single-log predictions automatically fall back to the `hybrid` environment subprocess. This does not affect full pipeline inference results.

**Dashboard is slow on first single-log prediction**

On first use, the dashboard loads all 81 BAT models into RAM (~3.5 GB per dataset). Subsequent predictions are fast (2–3 seconds). The first prediction may take up to 90 seconds while models load.

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
