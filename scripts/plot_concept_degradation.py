"""
Plot Concept Degradation Rate (CDR) hierarchy across semantic axes.

Generates comparative visualizations showing:
1. Shape (11.2% CDR) vs Colour (13.9%) vs Border (14.0%) vs Symbol (20.2%)
2. Visual proof of Semantic Degradation Hierarchy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_cdr(output_dir: str | Path = "reports/diagnostics"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = ["Shape\n(Global Geometry)", "Colour\n(Photometry)", "Border\n(Topology)", "Symbol Category\n(Fine Detail)"]
    cdr_s5 = [11.2, 13.9, 14.0, 20.2]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(concepts, cdr_s5, color=colors, width=0.55, edgecolor="black", linewidth=1.2)

    ax.set_ylabel("Concept Degradation Rate (CDR %) at Severity 5", fontsize=11, fontweight="bold")
    ax.set_title("Semantic Robustness Hierarchy (GTSRB CBM ResNet-18)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    out_path = out_dir / "concept_degradation_hierarchy.png"
    plt.savefig(out_path)
    plt.close()
    print(f"CDR plot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="reports/diagnostics")
    args = parser.parse_args()
    plot_cdr(args.output_dir)


if __name__ == "__main__":
    main()

