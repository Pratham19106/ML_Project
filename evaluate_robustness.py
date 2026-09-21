"""GTSRB Clean and Corruption Robustness Evaluation Benchmark

Evaluates a trained model checkpoint on:
1. Held-Out Clean Test Set (test.csv, 12,630 samples):
   - Top-1 Accuracy, Macro-F1, Expected Calibration Error (ECE).
2. Held-Out Corruption Suite (15 corruptions x 5 severities):
   - Per-corruption, per-severity accuracy.
   - Mean Corruption Accuracy (mCA) and Relative Performance Drop (RPD).
3. Exports clean CSV and JSON reports to reports/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.corruptions import CORRUPTIONS_REGISTRY
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, letterbox_roi_crop
from src.models.baseline_cnn import BaselineResNet18


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    test_df: pd.DataFrame,
    data_root: Path,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Evaluates the model on the clean held-out test split."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []

    # Batch iterator
    for start in tqdm(range(0, len(test_df), batch_size), desc="Clean Test Eval", unit="batch"):
        batch_rows = test_df.iloc[start : start + batch_size]
        tensors = []
        targets = []

        for _, row in batch_rows.iterrows():
            rel_p = str(row["source_relative_path"]).replace("\\", "/")
            with Image.open(data_root / rel_p) as raw_img:
                img = raw_img.convert("RGB")
            roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
            canvas = letterbox_roi_crop(img, roi, target_size=64)
            t = TF.normalize(TF.to_tensor(canvas), mean=IMAGENET_MEAN, std=IMAGENET_STD)
            tensors.append(t)
            targets.append(int(row["class_id"]))

        batch_tensor = torch.stack(tensors).to(device)
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()

        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_targets.extend(targets)
        all_probs.append(probs)

    all_targets_arr = np.array(all_targets)
    all_preds_arr = np.array(all_preds)
    all_probs_arr = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(all_targets_arr, all_preds_arr)
    macro_f1 = f1_score(all_targets_arr, all_preds_arr, average="macro", zero_division=0)
    ece = compute_ece(all_probs_arr, all_targets_arr)

    return {
        "clean_accuracy": float(acc),
        "clean_macro_f1": float(macro_f1),
        "clean_ece": float(ece),
        "total_test_samples": len(test_df),
    }


@torch.no_grad()
def evaluate_corruptions(
    model: nn.Module,
    test_df: pd.DataFrame,
    data_root: Path,
    device: torch.device,
    batch_size: int = 128,
    sample_limit: int | None = None,
) -> pd.DataFrame:
    """Evaluates the model across 15 corruptions and 5 severities."""
    model.eval()
    eval_df = test_df if sample_limit is None else test_df.sample(n=sample_limit, random_state=42)
    records = []

    # Pre-load clean canvases and targets once to maximize speed
    print(f"Pre-caching {len(eval_df):,} test canvas images...")
    clean_items = []
    for _, row in eval_df.iterrows():
        rel_p = str(row["source_relative_path"]).replace("\\", "/")
        with Image.open(data_root / rel_p) as raw_img:
            img = raw_img.convert("RGB")
        roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
        canvas = letterbox_roi_crop(img, roi, target_size=64)
        clean_items.append((canvas, int(row["class_id"])))

    targets_arr = np.array([item[1] for item in clean_items])

    for c_name, c_func in CORRUPTIONS_REGISTRY.items():
        row_dict = {"corruption": c_name}
        for severity in range(1, 6):
            all_preds: list[int] = []

            for start in range(0, len(clean_items), batch_size):
                batch_items = clean_items[start : start + batch_size]
                tensors = []
                for canvas, _ in batch_items:
                    corrupted_canvas = c_func(canvas, severity)
                    t = TF.normalize(TF.to_tensor(corrupted_canvas), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    tensors.append(t)

                batch_t = torch.stack(tensors).to(device)
                logits = model(batch_t)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

            acc = accuracy_score(targets_arr, np.array(all_preds))
            row_dict[f"severity_{severity}"] = float(acc)

        # Average across the 5 severities
        severities_accs = [row_dict[f"severity_{s}"] for s in range(1, 6)]
        row_dict["mean_acc"] = float(np.mean(severities_accs))
        records.append(row_dict)
        print(f"[{c_name:20s}] Mean Acc: {row_dict['mean_acc']*100:.2f}% (S1: {row_dict['severity_1']*100:.1f}% -> S5: {row_dict['severity_5']*100:.1f}%)")

    matrix_df = pd.DataFrame(records)
    return matrix_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate GTSRB Baseline ResNet-18 on Clean and Corrupted Test Sets.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_baseline_resnet18.pt")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB folder (default: german)")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv", help="Path to test.csv")
    parser.add_argument("--output-dir", type=str, default="reports/baseline_resnet18", help="Output directory for reports")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--quick-limit", type=int, default=None, help="Limit samples for quick corruption test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    test_df = pd.read_csv(args.meta_path)

    print("=" * 75)
    print("GTSRB ROBUSTNESS BENCHMARK EVALUATION")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Test Set   : {len(test_df):,} samples ({args.meta_path})")
    print(f"Report Dir : {out_dir}")
    print("=" * 75)

    # Load Model
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    model = BaselineResNet18(num_classes=43, pretrained=False)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)
    model.eval()

    # 1. Clean Evaluation
    print("\n--- 1. Clean Held-Out Test Evaluation ---")
    clean_results = evaluate_clean(model, test_df, data_root, device, batch_size=args.batch_size)
    print(f"Clean Test Accuracy : {clean_results['clean_accuracy']*100:.2f}%")
    print(f"Clean Test Macro-F1 : {clean_results['clean_macro_f1']*100:.2f}%")
    print(f"Clean Test ECE      : {clean_results['clean_ece']:.4f}")

    # 2. Corruption Evaluation
    print("\n--- 2. Corruption Robustness Benchmark (15 Corruptions x 5 Severities) ---")
    matrix_df = evaluate_corruptions(
        model, test_df, data_root, device, batch_size=args.batch_size, sample_limit=args.quick_limit
    )

    mca = float(matrix_df["mean_acc"].mean())
    clean_acc = clean_results["clean_accuracy"]
    rpd = float((clean_acc - mca) / clean_acc) if clean_acc > 0 else 0.0

    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"Clean Test Accuracy         : {clean_acc*100:.2f}%")
    print(f"Clean Test Macro-F1         : {clean_results['clean_macro_f1']*100:.2f}%")
    print(f"Clean Expected Calib Error  : {clean_results['clean_ece']:.4f}")
    print(f"Mean Corruption Acc (mCA)   : {mca*100:.2f}%")
    print(f"Relative Perf Drop (RPD)    : {rpd*100:.2f}%")
    print("=" * 75)

    # Save reports
    matrix_csv = out_dir / "robustness_matrix.csv"
    summary_json = out_dir / "evaluation_summary.json"

    matrix_df.to_csv(matrix_csv, index=False)
    summary_data = {
        **clean_results,
        "mean_corruption_accuracy": mca,
        "relative_performance_drop": rpd,
        "checkpoint_config": checkpoint_data.get("config", {}),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved Reports:")
    print(f"  - Robustness Matrix CSV : {matrix_csv}")
    print(f"  - Evaluation Summary    : {summary_json}")


if __name__ == "__main__":
    main()

