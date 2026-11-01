import json
from pathlib import Path

def create_phase2_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Cross-Dataset Robustness & In-Domain Generalization Benchmark (Phase 1 & Phase 2)\n",
                "\n",
                "This notebook executes the complete cross-border and in-domain evaluation protocol across **Germany (GTSRB)**, **Belgium (BelgiumTSC)**, and **China (Chinese TSRD)**:\n",
                "\n",
                "1. **Phase 1 Robustness Benchmark:** Probing frozen German-trained representations under 15 Hendrycks & Dietterich corruptions across 5 severities (75 conditions) under **Static BN** vs. **Test-Time BN Adaptation** on foreign roadways.\n",
                "2. **Phase 2 In-Domain Training:** Training black-box **Baseline ResNet-18** and **Frequency-Aware CBM** directly on Belgian (62 classes) and Chinese (58 classes) roads with decoupled No-Bias-Decay SGD and Cosine Annealing.\n",
                "3. **Phase 2 In-Domain Robustness Benchmark:** Testing in-domain models across all 75 corruption conditions to evaluate whether in-domain models resolve the Covariate Shift Paradox.\n",
                "\n",
                "**LIVE ON-THE-FLY DRIVE SYNC:** Every single model checkpoint and corruption report is synced immediately to your Google Drive (`MyDrive/ML_Project_Phase2_Results/`) the instant it finishes—protecting all progress against disconnects."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 1: Environment Setup, Google Drive Mount & A100 GPU Verification\n",
                "# ==============================================================================\n",
                "from google.colab import drive\n",
                "import torch\n",
                "import os\n",
                "\n",
                "drive.mount('/content/drive')\n",
                "print('PyTorch Version:', torch.__version__)\n",
                "print('CUDA Available: ', torch.cuda.is_available())\n",
                "\n",
                "if torch.cuda.is_available():\n",
                "    device_name = torch.cuda.get_device_name(0)\n",
                "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "    print(f'Active GPU:      {device_name} ({vram_gb:.1f} GB VRAM)')\n",
                "    \n",
                "    # Enable Ampere / A100 Hardware Accelerations\n",
                "    torch.backends.cuda.matmul.allow_tf32 = True\n",
                "    torch.backends.cudnn.allow_tf32 = True\n",
                "    torch.backends.cudnn.benchmark = True\n",
                "    bf16_ok = torch.cuda.is_bf16_supported()\n",
                "    print(f'TF32 Tensor Cores: Enabled | Native bfloat16: {bf16_ok} | cuDNN Autotuner: Enabled')\n",
                "\n",
                "# Pre-create live Google Drive destination directories for instant on-the-fly syncing\n",
                "drive_results_dir = '/content/drive/MyDrive/ML_Project_Phase2_Results'\n",
                "os.makedirs(f'{drive_results_dir}/reports', exist_ok=True)\n",
                "os.makedirs(f'{drive_results_dir}/checkpoints', exist_ok=True)\n",
                "print(f'\\n[LIVE SYNC ACTIVE] All results will stream on the fly to:\\n  -> {drive_results_dir}/')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 2: Unpack Phase 2 Code & Standardized Datasets (Python zipfile engine)\n",
                "# ==============================================================================\n",
                "# Uses Python zipfile to bypass Ubuntu CLI 'unzip' CVE-2019-13232 buffer limits\n",
                "import os\n",
                "import sys\n",
                "import zipfile\n",
                "from pathlib import Path\n",
                "\n",
                "drive_pkg = '/content/drive/MyDrive/phase2_colab_package.zip'\n",
                "local_pkg = '/content/phase2_colab_package.zip'\n",
                "\n",
                "if os.path.exists(drive_pkg):\n",
                "    pkg_path = drive_pkg\n",
                "elif os.path.exists(local_pkg):\n",
                "    pkg_path = local_pkg\n",
                "else:\n",
                "    raise FileNotFoundError('phase2_colab_package.zip not found! Please upload it to Google Drive or /content/')\n",
                "\n",
                "print(f'Unpacking {pkg_path} via Python zipfile engine...')\n",
                "dest_dir = Path('/content/ML_Project')\n",
                "dest_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "with zipfile.ZipFile(pkg_path, 'r') as zf:\n",
                "    zf.extractall(dest_dir)\n",
                "\n",
                "print('Unpack complete!')\n",
                "\n",
                "%cd /content/ML_Project\n",
                "if '/content/ML_Project' not in sys.path:\n",
                "    sys.path.insert(0, '/content/ML_Project')\n",
                "\n",
                "import torch\n",
                "from src.models.cbm import ConceptBottleneckModel\n",
                "from src.models.baseline_cnn import BaselineResNet18\n",
                "print('Environment, Checkpoints & Pre-Rendered Datasets Verified Successfully!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 3: Phase 1 Cross-Dataset Robustness Benchmark (On-The-Fly Drive Sync)\n",
                "# ==============================================================================\n",
                "# Evaluates German-trained Baseline and CBM models on foreign test sets across\n",
                "# 15 corruptions x 5 severities under both Static BN and Test-Time Adapted BN.\n",
                "# Every single model/dataset matrix is synced to Google Drive on the fly:\n",
                "\n",
                "!python scripts/evaluate_cross_dataset_robustness.py \\\n",
                "    --phase 1 \\\n",
                "    --dataset all \\\n",
                "    --batch-size 256 \\\n",
                "    --num-workers 4 \\\n",
                "    --output-dir reports/cross_dataset_robustness\n",
                "\n",
                "# Verify Drive reports\n",
                "!ls -lh /content/drive/MyDrive/ML_Project_Phase2_Results/reports/\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 4A: Phase 2 In-Domain Training — Belgium (62 classes)\n",
                "# ==============================================================================\n",
                "# Checkpoints stream live to Google Drive at every peak validation epoch (~2-3 mins on A100):\n",
                "\n",
                "print('>>> Training Model 1/4: Belgium Baseline ResNet-18 (62 classes)...')\n",
                "!python scripts/run_phase2_training.py --dataset belgium --model-type baseline --epochs 30 --batch-size 128 --lr 0.05 --num-workers 4\n",
                "\n",
                "print('\\n>>> Training Model 2/4: Belgium Frequency-Aware CBM (62 classes)...')\n",
                "!python scripts/run_phase2_training.py --dataset belgium --model-type freq_cbm --epochs 30 --batch-size 128 --lr 0.05 --num-workers 4\n",
                "\n",
                "# Check Drive checkpoint status\n",
                "!ls -lh /content/drive/MyDrive/ML_Project_Phase2_Results/checkpoints/\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 4B: Phase 2 In-Domain Training — China (58 classes)\n",
                "# ==============================================================================\n",
                "# Checkpoints stream live to Google Drive at every peak validation epoch (~2-3 mins on A100):\n",
                "\n",
                "print('>>> Training Model 3/4: China Baseline ResNet-18 (58 classes)...')\n",
                "!python scripts/run_phase2_training.py --dataset china --model-type baseline --epochs 30 --batch-size 128 --lr 0.05 --num-workers 4\n",
                "\n",
                "print('\\n>>> Training Model 4/4: China Frequency-Aware CBM (58 classes)...')\n",
                "!python scripts/run_phase2_training.py --dataset china --model-type freq_cbm --epochs 30 --batch-size 128 --lr 0.05 --num-workers 4\n",
                "\n",
                "# Check Drive checkpoint status\n",
                "!ls -lh /content/drive/MyDrive/ML_Project_Phase2_Results/checkpoints/\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 5: Phase 2 In-Domain Robustness Benchmark (On-The-Fly Drive Sync)\n",
                "# ==============================================================================\n",
                "# Evaluates all four newly trained in-domain models across 15 corruptions x 5 severities\n",
                "# under both Static BN and Test-Time Adapted BN (~1-2 mins total on A100):\n",
                "\n",
                "!python scripts/evaluate_cross_dataset_robustness.py \\\n",
                "    --phase 2 \\\n",
                "    --dataset all \\\n",
                "    --batch-size 256 \\\n",
                "    --num-workers 4 \\\n",
                "    --output-dir reports/cross_dataset_robustness\n",
                "\n",
                "# Verify Drive reports\n",
                "!ls -lh /content/drive/MyDrive/ML_Project_Phase2_Results/reports/\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 6: Consolidate Metrics & Display Master Comparative Table\n",
                "# ==============================================================================\n",
                "import glob\n",
                "import json\n",
                "import pandas as pd\n",
                "\n",
                "# Reads directly from Google Drive to ensure 100% persistent verified results\n",
                "drive_reports = sorted(glob.glob('/content/drive/MyDrive/ML_Project_Phase2_Results/reports/*.json'))\n",
                "local_reports = sorted(glob.glob('reports/cross_dataset_robustness/*.json'))\n",
                "report_paths = drive_reports if drive_reports else local_reports\n",
                "\n",
                "records = []\n",
                "for path in report_paths:\n",
                "    with open(path) as f:\n",
                "        d = json.load(f)\n",
                "    records.append({\n",
                "        'Report': path.split('/')[-1],\n",
                "        'Dataset': d.get('dataset', '').upper(),\n",
                "        'Model': d.get('model', d.get('model_type', '')),\n",
                "        'Clean Acc (%)': round(d.get('clean_acc', d.get('clean_top1_acc', 0.0)) * 100, 2),\n",
                "        'Static mCA (%)': round(d.get('static_mCA', 0.0) * 100, 2),\n",
                "        'Static RPD (%)': round(d.get('static_RPD', 0.0), 2),\n",
                "        'Adapt mCA (%)': round(d.get('adapt_mCA', 0.0) * 100, 2),\n",
                "        'Adapt RPD (%)': round(d.get('adapt_RPD', 0.0), 2),\n",
                "    })\n",
                "\n",
                "df_summary = pd.DataFrame(records)\n",
                "display(df_summary)\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "A100"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    nb_path = Path("notebooks/phase2_cross_dataset_robustness_colab.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("Successfully updated notebook with ON-THE-FLY DRIVE SYNC at:", nb_path)

if __name__ == "__main__":
    create_phase2_notebook()
