"""
Unified Cross-Dataset Robustness Benchmark Suite (Phase 1 and Phase 2)
======================================================================
Evaluates:
- Phase 1: German-trained models (Baseline & CBM) on Belgium & China corruptions
           (15 corruptions x 5 severities = 75 conditions under Static vs Adapted BN)
- Phase 2: In-domain trained models (Baseline & Freq-CBM) on Belgium & China corruptions
           (75 conditions under Static vs Adapted BN)

Optimized for High-Throughput Modern GPUs (NVIDIA A100 / V100 / T4):
- Automatic TF32 matrix multiplication acceleration (Ampere / A100)
- cuDNN benchmark autotuning
- Automatic bfloat16 / float16 mixed precision
- Multi-worker threaded data loading with persistent workers and prefetch
- Parallelized linear probe solver (n_jobs=-1)
- Live On-The-Fly Google Drive synchronization after every model/dataset run
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import CORRUPTIONS_REGISTRY
from src.models.baseline_cnn import BaselineResNet18
from src.models.cbm import ConceptBottleneckModel
from src.models.concept_encoder import (
    BORDER_VOCAB,
    COLOUR_VOCAB,
    SHAPE_VOCAB,
    SYMCAT_VOCAB,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def configure_gpu_performance() -> torch.dtype:
    """Configures TF32 and cuDNN autotuning for maximum A100/Ampere throughput."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float16


def sync_reports_to_drive(file_paths: List[Path], custom_drive_dir: Optional[str] = None) -> None:
    """Copies generated JSON/CSV report files to Google Drive immediately on the fly."""
    target_dir = None
    if custom_drive_dir:
        target_dir = Path(custom_drive_dir)
    else:
        colab_default = Path("/content/drive/MyDrive/ML_Project_Phase2_Results/reports")
        if colab_default.parent.exists():
            target_dir = colab_default

    if target_dir is not None:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            for p in file_paths:
                if p.exists():
                    dst = target_dir / p.name
                    shutil.copy2(p, dst)
                    print(f"  [DRIVE SYNC] Synced {p.name} -> {dst}")
        except Exception as e:
            print(f"  [DRIVE SYNC WARNING] Could not sync to drive: {e}")


def set_bn_mode(model: nn.Module, mode: str) -> None:
    """Configures normalization statistics regime."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if mode == "adapt":
                m.train()
                m.track_running_stats = True
                m.momentum = None  # Cumulative mini-batch statistics
            elif mode == "static":
                m.eval()
                m.track_running_stats = False


class CorruptedTSRDataset(Dataset):
    """Loads standardized 64x64 images and applies on-the-fly Hendrycks corruptions."""

    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        corruption_fn: Optional[Callable[[Image.Image, int], Image.Image]] = None,
        severity: int = 1,
    ):
        self.data_root = Path(data_root).resolve()
        self.df = pd.read_csv(manifest_path)
        self.corruption_fn = corruption_fn
        self.severity = severity
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        self.shapes = [SHAPE_VOCAB.get(s, 0) for s in self.df["shape"]]
        self.colours = [COLOUR_VOCAB.get(c, 0) for c in self.df["colour_profile"]]
        self.borders = [BORDER_VOCAB.get(b, 0) for b in self.df["border_type"]]
        self.symbols = [SYMCAT_VOCAB.get(s, 0) for s in self.df["symbol_category"]]
        self.class_ids = self.df["class_id"].values.astype(np.int64)
        self.paths = self.df["canonical_image_path"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        full_path = self.data_root / self.paths[idx]
        with Image.open(full_path) as img:
            img = img.convert("RGB")

            if self.corruption_fn is not None:
                img = self.corruption_fn(img, self.severity)
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))

            tensor = TF.to_tensor(img)
            tensor = self.normalize(tensor)

        return {
            "image": tensor,
            "label": torch.tensor(self.class_ids[idx], dtype=torch.long),
            "shape": torch.tensor(self.shapes[idx], dtype=torch.long),
            "colour": torch.tensor(self.colours[idx], dtype=torch.long),
            "border": torch.tensor(self.borders[idx], dtype=torch.long),
            "symbol": torch.tensor(self.symbols[idx], dtype=torch.long),
        }


# ==============================================================================
# Feature Extraction Routine
# ==============================================================================

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_cbm: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, np.ndarray]:
    features_list = []
    labels_list = []
    s_preds, c_preds, b_preds, y_preds = [], [], [], []
    s_tgts, c_tgts, b_tgts, y_tgts = [], [], [], []

    use_amp = (device.type == "cuda")

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            if is_cbm:
                outputs = model(images)
                feat = outputs["features"]
                s_preds.extend(torch.argmax(outputs["logits_shape"], dim=1).cpu().numpy())
                c_preds.extend(torch.argmax(outputs["logits_colour"], dim=1).cpu().numpy())
                b_preds.extend(torch.argmax(outputs["logits_border"], dim=1).cpu().numpy())
                y_preds.extend(torch.argmax(outputs["logits_symcat"], dim=1).cpu().numpy())

                s_tgts.extend(batch["shape"].numpy())
                c_tgts.extend(batch["colour"].numpy())
                b_tgts.extend(batch["border"].numpy())
                y_tgts.extend(batch["symbol"].numpy())
            else:
                feat = model.extract_features(images)

        features_list.append(feat.float().cpu().numpy())
        labels_list.append(batch["label"].numpy())

    out = {
        "features": np.vstack(features_list),
        "labels": np.concatenate(labels_list),
    }
    if is_cbm:
        out.update({
            "s_preds": np.array(s_preds), "s_tgts": np.array(s_tgts),
            "c_preds": np.array(c_preds), "c_tgts": np.array(c_tgts),
            "b_preds": np.array(b_preds), "b_tgts": np.array(b_tgts),
            "y_preds": np.array(y_preds), "y_tgts": np.array(y_tgts),
        })
    return out


# ==============================================================================
# Phase 1: Probed Cross-Dataset Robustness Evaluation
# ==============================================================================

def run_phase1_robustness(
    dataset_key: str,
    model_name: str,
    device: torch.device,
    reports_dir: Path,
    batch_size: int = 128,
    num_workers: int = 4,
    amp_dtype: torch.dtype = torch.float16,
    drive_dir: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n>>> Running Phase 1 Robustness: German {model_name} on {dataset_key.upper()}...")
    is_cbm = ("cbm" in model_name.lower())
    data_root = PROJECT_ROOT / "data" / ("data_v2" if dataset_key == "belgium" else "data_v3")
    train_manifest = data_root / "metadata" / "train.csv"
    test_manifest = data_root / "metadata" / "test.csv"

    # Load German Pretrained Model
    if is_cbm:
        ckpt_path = PROJECT_ROOT / "reports" / "cbm_resnet18" / "best_cbm_resnet18.pt"
        model = ConceptBottleneckModel(num_classes=43, pretrained=False, bottleneck_mode="residual").to(device)
    else:
        ckpt_path = PROJECT_ROOT / "reports" / "baseline_resnet18" / "best_baseline_resnet18.pt"
        model = BaselineResNet18(num_classes=43, pretrained=False).to(device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"German checkpoint missing: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 1. Fit Linear Probe on Clean Target Training Features
    clean_train_ds = CorruptedTSRDataset(data_root, train_manifest, corruption_fn=None)
    clean_train_loader = DataLoader(
        clean_train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available(),
    )
    set_bn_mode(model, "static")
    train_feat = extract_features(model, clean_train_loader, device, is_cbm=is_cbm, amp_dtype=amp_dtype)

    # Parallel L-BFGS linear classifier
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42, n_jobs=-1)
    clf.fit(train_feat["features"], train_feat["labels"])

    # 2. Evaluate Clean Test
    clean_test_ds = CorruptedTSRDataset(data_root, test_manifest, corruption_fn=None)
    clean_test_loader = DataLoader(
        clean_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available(),
    )
    clean_test_feat = extract_features(model, clean_test_loader, device, is_cbm=is_cbm, amp_dtype=amp_dtype)
    clean_acc = float(accuracy_score(clean_test_feat["labels"], clf.predict(clean_test_feat["features"])))

    # 3. Sweep All 75 Corruptions under Static BN and Adapted BN
    results: Dict[str, Any] = {
        "dataset": dataset_key,
        "model": model_name,
        "clean_acc": clean_acc,
        "static_matrix": {},
        "adapt_matrix": {},
    }

    for bn_mode in ["static", "adapt"]:
        matrix = {c: [] for c in CORRUPTIONS_REGISTRY.keys()}
        print(f"  Evaluating {bn_mode.upper()} BN across 15 corruptions x 5 severities...")

        for c_name, c_fn in CORRUPTIONS_REGISTRY.items():
            for sev in range(1, 6):
                eval_model = copy.deepcopy(model)
                set_bn_mode(eval_model, bn_mode)

                corrupt_ds = CorruptedTSRDataset(data_root, test_manifest, corruption_fn=c_fn, severity=sev)
                corrupt_loader = DataLoader(
                    corrupt_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers if sys.platform != "win32" else 0,
                    pin_memory=torch.cuda.is_available(),
                )

                feat_out = extract_features(eval_model, corrupt_loader, device, is_cbm=is_cbm, amp_dtype=amp_dtype)
                y_pred = clf.predict(feat_out["features"])
                acc = float(accuracy_score(feat_out["labels"], y_pred))
                matrix[c_name].append(acc)

        results[f"{bn_mode}_matrix"] = matrix
        all_vals = [v for vals in matrix.values() for v in vals]
        mca = float(np.mean(all_vals))
        rpd = float((clean_acc - mca) / clean_acc * 100.0) if clean_acc > 0 else 0.0
        results[f"{bn_mode}_mCA"] = mca
        results[f"{bn_mode}_RPD"] = rpd
        print(f"  [{bn_mode.upper()}] Clean: {clean_acc*100:.2f}% | mCA: {mca*100:.2f}% | RPD: {rpd:.2f}%")

    out_json = reports_dir / f"phase1_robustness_{dataset_key}_{model_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    f_static = reports_dir / f"phase1_matrix_{dataset_key}_{model_name}_static.csv"
    df_static = pd.DataFrame(results["static_matrix"], index=[f"sev_{s}" for s in range(1, 6)]).T
    df_static.to_csv(f_static)

    f_adapt = reports_dir / f"phase1_matrix_{dataset_key}_{model_name}_adapt.csv"
    df_adapt = pd.DataFrame(results["adapt_matrix"], index=[f"sev_{s}" for s in range(1, 6)]).T
    df_adapt.to_csv(f_adapt)

    # Live sync to Google Drive on the fly
    sync_reports_to_drive([out_json, f_static, f_adapt], custom_drive_dir=drive_dir)

    return results


# ==============================================================================
# Phase 2: In-Domain Robustness Evaluation
# ==============================================================================

def run_phase2_robustness(
    dataset_key: str,
    model_type: str,
    ckpt_path: Path,
    device: torch.device,
    reports_dir: Path,
    batch_size: int = 128,
    num_workers: int = 4,
    amp_dtype: torch.dtype = torch.float16,
    drive_dir: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n>>> Running Phase 2 In-Domain Robustness: {model_type.upper()} on {dataset_key.upper()}...")
    is_cbm = (model_type in ["freq_cbm", "cbm"])
    data_root = PROJECT_ROOT / "data" / ("data_v2" if dataset_key == "belgium" else "data_v3")
    test_manifest = data_root / "metadata" / "test.csv"
    num_classes = 62 if dataset_key == "belgium" else 58

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase 2 Checkpoint missing: {ckpt_path}")

    if is_cbm:
        model = ConceptBottleneckModel(num_classes=num_classes, pretrained=False, bottleneck_mode="residual").to(device)
    else:
        model = BaselineResNet18(num_classes=num_classes, pretrained=False).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 1. Evaluate Clean In-Domain Test
    clean_ds = CorruptedTSRDataset(data_root, test_manifest, corruption_fn=None)
    clean_loader = DataLoader(
        clean_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    set_bn_mode(model, "static")
    model.eval()
    clean_preds, clean_targets = [], []
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for batch in clean_loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                if is_cbm:
                    out = model(images)
                    preds = torch.argmax(out["logits"], dim=1).cpu().numpy()
                else:
                    preds = torch.argmax(model(images), dim=1).cpu().numpy()
            clean_preds.extend(preds)
            clean_targets.extend(batch["label"].numpy())

    clean_acc = float(accuracy_score(clean_targets, clean_preds))
    clean_f1 = float(f1_score(clean_targets, clean_preds, average="macro", zero_division=0))

    results: Dict[str, Any] = {
        "dataset": dataset_key,
        "model_type": model_type,
        "num_classes": num_classes,
        "clean_top1_acc": clean_acc,
        "clean_macro_f1": clean_f1,
        "static_matrix": {},
        "adapt_matrix": {},
    }

    for bn_mode in ["static", "adapt"]:
        matrix = {c: [] for c in CORRUPTIONS_REGISTRY.keys()}
        print(f"  Evaluating {bn_mode.upper()} BN across 15 corruptions x 5 severities...")

        for c_name, c_fn in CORRUPTIONS_REGISTRY.items():
            for sev in range(1, 6):
                eval_model = copy.deepcopy(model)
                set_bn_mode(eval_model, bn_mode)

                corrupt_ds = CorruptedTSRDataset(data_root, test_manifest, corruption_fn=c_fn, severity=sev)
                corrupt_loader = DataLoader(
                    corrupt_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers if sys.platform != "win32" else 0,
                    pin_memory=torch.cuda.is_available(),
                )

                c_preds, c_tgts = [], []
                with torch.no_grad():
                    for batch in corrupt_loader:
                        images = batch["image"].to(device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                            if is_cbm:
                                out = eval_model(images)
                                p = torch.argmax(out["logits"], dim=1).cpu().numpy()
                            else:
                                p = torch.argmax(eval_model(images), dim=1).cpu().numpy()
                        c_preds.extend(p)
                        c_tgts.extend(batch["label"].numpy())

                acc = float(accuracy_score(c_tgts, c_preds))
                matrix[c_name].append(acc)

        results[f"{bn_mode}_matrix"] = matrix
        all_vals = [v for vals in matrix.values() for v in vals]
        mca = float(np.mean(all_vals))
        rpd = float((clean_acc - mca) / clean_acc * 100.0) if clean_acc > 0 else 0.0
        results[f"{bn_mode}_mCA"] = mca
        results[f"{bn_mode}_RPD"] = rpd
        print(f"  [{bn_mode.upper()}] Clean: {clean_acc*100:.2f}% | mCA: {mca*100:.2f}% | RPD: {rpd:.2f}%")

    out_json = reports_dir / f"phase2_robustness_{dataset_key}_{model_type}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    f_static = reports_dir / f"phase2_matrix_{dataset_key}_{model_type}_static.csv"
    df_static = pd.DataFrame(results["static_matrix"], index=[f"sev_{s}" for s in range(1, 6)]).T
    df_static.to_csv(f_static)

    f_adapt = reports_dir / f"phase2_matrix_{dataset_key}_{model_type}_adapt.csv"
    df_adapt = pd.DataFrame(results["adapt_matrix"], index=[f"sev_{s}" for s in range(1, 6)]).T
    df_adapt.to_csv(f_adapt)

    # Live sync to Google Drive on the fly
    sync_reports_to_drive([out_json, f_static, f_adapt], custom_drive_dir=drive_dir)

    return results


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Robustness Benchmark Suite")
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "all"],
                        help="Benchmark phase to execute: '1', '2', or 'all'")
    parser.add_argument("--dataset", type=str, default="all", choices=["belgium", "china", "all"],
                        help="Target dataset: 'belgium', 'china', or 'all'")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for corruption passes")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--output-dir", type=str, default="reports/cross_dataset_robustness",
                        help="Directory to save evaluation reports and matrices")
    parser.add_argument("--drive-dir", type=str, default="", help="Optional custom Google Drive directory to sync on the fly")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = configure_gpu_performance()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Cross-Dataset Robustness Evaluation Suite | Phase: {args.phase}")
    print(f"Hardware: {device} | Precision: {amp_dtype} | Batch Size: {args.batch_size} | Workers: {args.num_workers}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Active GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | TF32: True | cuDNN Benchmark: True")
    print(f"Reports Output Directory: {out_dir}")
    print(f"On-the-Fly Drive Sync: Active (Auto-detecting /content/drive/MyDrive/ML_Project_Phase2_Results/reports)")
    print("=" * 80)

    datasets = ["belgium", "china"] if args.dataset == "all" else [args.dataset]

    # --- Phase 1 Execution ---
    if args.phase in ["1", "all"]:
        for d in datasets:
            for m in ["baseline", "cbm"]:
                run_phase1_robustness(
                    dataset_key=d,
                    model_name=m,
                    device=device,
                    reports_dir=out_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    amp_dtype=amp_dtype,
                    drive_dir=args.drive_dir,
                )

    # --- Phase 2 Execution ---
    if args.phase in ["2", "all"]:
        for d in datasets:
            for m in ["baseline", "freq_cbm"]:
                ckpt = PROJECT_ROOT / "checkpoints" / f"phase2_{d}_{m}" / f"best_{m}_{d}.pt"
                if ckpt.exists():
                    run_phase2_robustness(
                        dataset_key=d,
                        model_type=m,
                        ckpt_path=ckpt,
                        device=device,
                        reports_dir=out_dir,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        amp_dtype=amp_dtype,
                        drive_dir=args.drive_dir,
                    )
                else:
                    print(f"Warning: Phase 2 checkpoint not found at {ckpt}. Skipping Phase 2 for {d}_{m}.")

    print("\n" + "=" * 80)
    print(f"Robustness Benchmarking Complete! All matrices and JSON reports saved locally and to Google Drive!")
    print("=" * 80)


if __name__ == "__main__":
    main()
