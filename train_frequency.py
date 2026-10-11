"""
Frequency-Aware ResNet-18 Training Script (GTSRB Master v1)

Experiment 3: Spectral Data-Centric Robustness (Fourier Frequency Regularization)
- Trains ResNet-18 with FrequencyAugDataset (LFC / High-Frequency Spectral Dropout).
- Grounding: Wang et al. (CVPR 2020) High Frequency Component generalization.
- Optimization Protocol (Identical to Baseline & AugMix):
    * SGD (Momentum 0.9, Nesterov, No-Bias-Decay rule)
    * 3-epoch Linear Warmup + Cosine Annealing decay down to 1e-5 (25 epochs)
    * Automatic Mixed Precision (torch.cuda.amp)
    * Validation Macro-F1 model selection (checkpoint: best_frequency_resnet18.pt / .pth)
    * Comprehensive logging of training loss, validation loss, accuracy, and macro-F1.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations.frequency import FrequencyAugDataset
from src.dataset import GTSRBDataset
from src.models.baseline_cnn import BaselineResNet18, get_parameter_groups


def set_seed(seed: int = 42) -> None:
    """Freezes all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Runs one training epoch and returns average loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            else torch.cuda.amp.autocast(enabled=use_amp)
        )
        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, float, float]:
    """Evaluates the model on validation data.
    Returns: (val_loss, val_accuracy, val_macro_f1)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=use_amp)
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
        else torch.cuda.amp.autocast(enabled=use_amp)
    )

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        targets = labels.cpu().numpy().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets)

    val_loss = running_loss / total_samples
    val_acc = float(accuracy_score(all_targets, all_preds))
    val_macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    return val_loss, val_acc, val_macro_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frequency-Aware ResNet-18 Training on GTSRB Master v1")
    parser.add_argument("--data-root", type=str, default="german")
    parser.add_argument("--meta-dir", type=str, default="data/master_v1/metadata")
    parser.add_argument("--output-dir", type=str, default="checkpoints/frequency_resnet18")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--severity", type=float, default=3.0, help="Frequency dropout severity [1..10]")
    parser.add_argument("--mode", type=str, default="dropout", choices=["dropout", "lfc", "blend"])
    parser.add_argument("--radius", type=int, default=14, help="Frequency filter cutoff radius")
    parser.add_argument("--dropout-prob", type=float, default=0.50, help="P(apply frequency dropout)")
    parser.add_argument("--blend-prob", type=float, default=0.30, help="P(apply frequency blend)")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and torch.cuda.is_available()

    data_root = Path(args.data_root)
    meta_dir = Path(args.meta_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("GTSRB FREQUENCY-AWARE RESNET-18 TRAINING PIPELINE (EXPERIMENT 3)")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Data Root     : {data_root.resolve()}")
    print(f"Metadata Dir  : {meta_dir.resolve()}")
    print(f"Output Dir    : {out_dir.resolve()}")
    print(f"Batch Size    : {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"Spectral Mode : {args.mode} (Severity: {args.severity}, Radius: {args.radius})")
    print(f"Mixed Prec.   : {use_amp}")
    print("=" * 75)

    # 1. Datasets and Loaders
    train_dataset = FrequencyAugDataset(
        manifest_path=meta_dir / "train.csv",
        data_root=data_root,
        target_size=64,
        margin_ratio=0.10,
        severity=args.severity,
        mode=args.mode,
        radius=args.radius,
        is_training=True,
    )
    val_dataset = GTSRBDataset(
        manifest_path=meta_dir / "val.csv",
        data_root=data_root,
        target_size=64,
        margin_ratio=0.10,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples : {len(train_dataset):,} ({len(train_loader)} batches/epoch)")
    print(f"Val samples   : {len(val_dataset):,} ({len(val_loader)} batches)")

    # 2. Model & Optimization
    model = BaselineResNet18(num_classes=43, pretrained=(not args.no_pretrained), zero_init_residual=True)
    model.to(device)

    param_groups = get_parameter_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, nesterov=True)

    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-5)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    criterion = nn.CrossEntropyLoss()

    best_val_macro_f1 = 0.0
    history: list[dict[str, Any]] = []
    total_start_time = time.time()

    print("\nStarting Frequency-Aware ResNet-18 Training...\n")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            current_lr = warmup_lr
        else:
            cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, use_amp)
        epoch_duration = time.time() - epoch_start

        is_best = val_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_f1
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "config": vars(args),
            }
            # Save as both .pt and .pth for maximum Colab compatibility
            torch.save(ckpt_data, out_dir / "best_frequency_resnet18.pt")
            torch.save(ckpt_data, out_dir / "best_frequency_resnet18.pth")

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "duration_sec": epoch_duration,
            "is_best": is_best,
        }
        history.append(epoch_record)

        marker = " [*BEST*]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"LR: {current_lr:.5f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_f1*100:.2f}% ({epoch_duration:.1f}s){marker}"
        )

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 75)
    print("TRAINING COMPLETED!")
    print(f"Total Time        : {total_duration/60:.2f} minutes")
    print(f"Best Val Macro-F1 : {best_val_macro_f1*100:.2f}%")
    print(f"Best Checkpoint   : {out_dir / 'best_frequency_resnet18.pt'} (.pth)")

    history_path = out_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "best_val_macro_f1": best_val_macro_f1,
                "total_duration_minutes": total_duration / 60,
                "epochs": history,
            },
            f,
            indent=2,
        )
    print(f"Training History  : {history_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()

