"""AugMix ResNet-18 Training Script (GTSRB Experiment 2)

Implements Experiment 2 of the Controlled Robustness Ladder:
X -> AugMix (Dirichlet w + Beta m) -> ResNet-18 -> Jensen-Shannon Consistency Loss (lambda = 12)

Identical Experimental Controls to Baseline (Causality Guaranteed):
1. Backbone: ResNet-18 with Kaiming Normal head and Zero-gamma BN init.
2. Optimizer: SGD (momentum 0.9, Nesterov, lr=0.05, No-Bias-Decay rule).
3. Schedule: 3-epoch linear warmup + Cosine Annealing decay down to 1e-5.
4. Precision: Automatic Mixed Precision (AMP) with float32 JSD log-space protection.
5. Model Checkpointing: Best Validation Macro-F1 on frozen val.csv (7,830 samples, 0 leakage).
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
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations.augmix import AugMixDataset, jensen_shannon_loss
from src.dataset import GTSRBDataset
from src.models.baseline_cnn import BaselineResNet18, get_parameter_groups


def set_seed(seed: int = 42) -> None:
    """Freezes all random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch_augmix(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    lambda_jsd: float = 12.0,
    use_amp: bool = True,
) -> tuple[float, float, float]:
    """Runs one training epoch with concatenated AugMix forward pass and JSD loss."""
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_jsd = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="AugMix Training", unit="batch", leave=False)
    for batch in pbar:
        img_clean = batch["image_clean"].to(device, non_blocking=True)
        img_aug1 = batch["image_aug1"].to(device, non_blocking=True)
        img_aug2 = batch["image_aug2"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        batch_size = img_clean.size(0)

        # Concatenate all 3 views into a single GPU tensor: [3*B, 3, 64, 64]
        images_all = torch.cat([img_clean, img_aug1, img_aug2], dim=0)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_all = model(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, batch_size)

            loss, loss_ce, loss_jsd = jensen_shannon_loss(
                logits_clean=logits_clean,
                logits_aug1=logits_aug1,
                logits_aug2=logits_aug2,
                targets=labels,
                lambda_jsd=lambda_jsd,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        running_ce += loss_ce.item() * batch_size
        running_jsd += loss_jsd.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ce": f"{loss_ce.item():.4f}",
            "jsd": f"{loss_jsd.item():.4f}",
        })

    avg_loss = running_loss / total_samples
    avg_ce = running_ce / total_samples
    avg_jsd = running_jsd / total_samples
    return avg_loss, avg_ce, avg_jsd


@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, float, float]:
    """Evaluates on clean held-out validation set.
    Returns: (val_loss, val_accuracy, val_macro_f1)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    avg_loss = running_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, float(acc), float(macro_f1)


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 with AugMix and JSD Consistency (GTSRB Exp 2).")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB folder (default: german)")
    parser.add_argument("--meta-dir", type=str, default="data/master_v1/metadata", help="Path to master manifests")
    parser.add_argument("--output-dir", type=str, default="checkpoints/augmix_resnet18", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and eval (default: 128)")
    parser.add_argument("--lr", type=float, default=0.05, help="Base learning rate (default: 0.05)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay on weights (default: 1e-4)")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs (default: 3)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--lambda-jsd", type=float, default=12.0, help="Weight for JSD loss (default: 12.0)")
    parser.add_argument("--lambda-jsd", "--jsd-weight", dest="lambda_jsd", type=float, default=12.0, help="Weight for JSD loss (default: 12.0)")
    parser.add_argument("--severity", type=int, default=3, help="AugMix operation severity (default: 3)")
    parser.add_argument("--width", type=int, default=3, help="Number of parallel AugMix chains (default: 3)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root).resolve()
    meta_dir = Path(args.meta_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print("=" * 75)
    print("EXPERIMENT 2: AUGMIX RESNET-18 TRAINING (GTSRB)")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Mixed Precision (AMP) : {use_amp}")
    print(f"JSD Weight (lambda)   : {args.lambda_jsd}")
    print(f"AugMix Severity/Width : {args.severity} / {args.width}")
    print(f"Epochs / Batch Size   : {args.epochs} / {args.batch_size}")
    print(f"Output Directory      : {out_dir}")
    print("=" * 75)

    # Manifests
    train_manifest = meta_dir / "train.csv"
    val_manifest = meta_dir / "val.csv"

    # Datasets
    # Training uses AugMixDataset (yields clean + 2 augmented views)
    train_dataset = AugMixDataset(
        manifest_path=train_manifest,
        data_root=data_root,
        target_size=64,
        margin_ratio=0.10,
        severity=args.severity,
        width=args.width,
        depth=-1,
        alpha=1.0,
        is_training=True,
    )
    # Validation uses standard clean GTSRBDataset
    val_dataset = GTSRBDataset(
        manifest_path=val_manifest,
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Training Samples   : {len(train_dataset):,} (AugMix active)")
    print(f"Validation Samples : {len(val_dataset):,} (Clean evaluation)")

    # Model & Optimizers (Identical to Baseline)
    model = BaselineResNet18(num_classes=43, pretrained=True)
    model.to(device)

    param_groups = get_parameter_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, nesterov=True)

    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    val_criterion = nn.CrossEntropyLoss()

    best_val_macro_f1 = 0.0
    history: list[dict[str, Any]] = []
    total_start_time = time.time()

    print("\nStarting AugMix Training...\n")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Handle linear warmup
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            current_lr = warmup_lr
        else:
            cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_ce, train_jsd = train_one_epoch_augmix(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            lambda_jsd=args.lambda_jsd,
            use_amp=use_amp,
        )

        val_loss, val_acc, val_f1 = evaluate_clean(model, val_loader, val_criterion, device, use_amp)
        epoch_duration = time.time() - epoch_start

        is_best = val_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_f1
            best_model_path = out_dir / "best_augmix_resnet18.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "config": vars(args),
            }, best_model_path)

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_jsd": train_jsd,
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
            f"Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, JSD: {train_jsd:.4f}) | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_f1*100:.2f}% ({epoch_duration:.1f}s){marker}"
        )

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 75)
    print(f"AugMix Training Complete! Total time: {total_duration/60:.1f} minutes")
    print(f"Best Validation Macro-F1: {best_val_macro_f1*100:.2f}%")
    print(f"Best Model Saved to     : {out_dir / 'best_augmix_resnet18.pt'}")
    print("=" * 75)

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "best_val_macro_f1": best_val_macro_f1,
            "total_duration_sec": total_duration,
            "epochs": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()

