"""
Master 2x4 Factorial Benchmark: Test-Time Normalization Adaptation Across Models
================================================================================
Evaluates:
  1. Baseline ResNet-18
  2. AugMix ResNet-18
  3. Frequency ResNet-18
  4. Concept Bottleneck Model (CBM) ResNet-18

Across:
  - Mode A: Static Inference (standard running statistics)
  - Mode B: Shift-Gated BN-Adapt (prediction-time covariate shift adaptation)

Produces a unified comparative report and Markdown/CSV summary table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_checkpoint(model_name: str, explicit_path: str | None) -> Path | None:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        if (PROJECT_ROOT / p).exists():
            return PROJECT_ROOT / p

    candidates = [
        PROJECT_ROOT / f"checkpoints/{model_name}_resnet18/best_{model_name}_resnet18.pt",
        PROJECT_ROOT / f"checkpoints/{model_name}/best_{model_name}.pt",
        PROJECT_ROOT / f"reports/{model_name}_resnet18/best_{model_name}_resnet18.pt",
        Path(f"/content/drive/MyDrive/checkpoints/{model_name}_resnet18/best_{model_name}_resnet18.pt"),
        Path(f"/content/ML_Project/checkpoints/{model_name}_resnet18/best_{model_name}_resnet18.pt"),
        Path(f"/content/drive/MyDrive/checkpoints/best_{model_name}_resnet18.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_benchmark_for_model(
    model_name: str,
    ckpt_path: Path,
    output_dir: Path,
    seeds: str = "42",
    corruptions: str = "all",
    batch_size: int = 128,
) -> dict | None:
    out_json = output_dir / f"test_time_{model_name}.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_full_benchmark_suite.py"),
        "--checkpoint", str(ckpt_path),
        "--output-json", str(out_json),
        "--seeds", seeds,
        "--corruptions", corruptions,
        "--batch-size", str(batch_size),
        "--mode", "full_75",
    ]

    print("\n" + "=" * 80)
    print(f"RUNNING TEST-TIME ADAPTATION FOR: {model_name.upper()}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Output     : {out_json}")
    print("=" * 80)

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Error evaluating {model_name} (exit code {res.returncode})")
        return None

    if out_json.exists():
        with open(out_json, "r") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Run 2x4 Factorial BN-Adapt Benchmark Across All Models.")
    parser.add_argument("--baseline-ckpt", type=str, default=None, help="Path to Baseline ResNet-18 checkpoint")
    parser.add_argument("--augmix-ckpt", type=str, default=None, help="Path to AugMix ResNet-18 checkpoint")
    parser.add_argument("--freq-ckpt", type=str, default=None, help="Path to Frequency ResNet-18 checkpoint")
    parser.add_argument("--cbm-ckpt", type=str, default=None, help="Path to CBM ResNet-18 checkpoint")
    parser.add_argument("--output-dir", type=str, default="reports/diagnostics", help="Output directory")
    parser.add_argument("--seeds", type=str, default="42", help="Seeds for multi-seed corruption evaluation (e.g. '42' or '42,43,44')")
    parser.add_argument("--corruptions", type=str, default="all", help="Corruptions: 'all' or comma-separated")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = [
        ("baseline", args.baseline_ckpt),
        ("augmix", args.augmix_ckpt),
        ("frequency", args.freq_ckpt),
        ("cbm", args.cbm_ckpt),
    ]

    summary_rows = []

    for name, explicit_ckpt in models_to_run:
        ckpt = find_checkpoint(name, explicit_ckpt)
        if not ckpt:
            print(f"\n[SKIP] Could not find checkpoint for '{name}'. Provide via --{name}-ckpt.")
            continue

        data = run_benchmark_for_model(
            model_name=name,
            ckpt_path=ckpt,
            output_dir=out_dir,
            seeds=args.seeds,
            corruptions=args.corruptions,
            batch_size=args.batch_size,
        )

        if data and "sweep_75" in data:
            sw = data["sweep_75"]
            clean_static = sw.get("clean", {}).get("acc_static", 0.0) * 100
            clean_gated = sw.get("clean", {}).get("acc_gated", 0.0) * 100
            mca_static = sw.get("mca_static_75", 0.0) * 100
            mca_adapt = sw.get("mca_adapt_75", 0.0) * 100
            mca_gated = sw.get("mca_gated_75", 0.0) * 100
            gain_gated = sw.get("mca_gain_gated", 0.0) * 100

            summary_rows.append({
                "Model": name.upper(),
                "Clean Static (%)": f"{clean_static:.2f}%",
                "Clean Gated (%)": f"{clean_gated:.2f}%",
                "Static mCA (%)": f"{mca_static:.2f}%",
                "BN-Adapt mCA (%)": f"{mca_adapt:.2f}%",
                "Shift-Gated mCA (%)": f"{mca_gated:.2f}%",
                "Net Gain (Δ)": f"{gain_gated:+.2f}%",
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n" + "=" * 85)
        print("MASTER 2x4 FACTORIAL ROBUSTNESS GRID (STATIC VS. SHIFT-GATED BN-ADAPT)")
        print("=" * 85)
        print(df.to_markdown(index=False))
        print("=" * 85)

        master_csv = out_dir / "master_2x4_bn_benchmark.csv"
        df.to_csv(master_csv, index=False)
        print(f"Saved master summary to: {master_csv}")


if __name__ == "__main__":
    main()

