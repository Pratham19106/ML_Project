"""
Generate Figure 16: Cross-Dataset Robustness & In-Domain Generalization Benchmark
================================================================================
Generates publication-quality 300 DPI multi-panel figure visualizing:
- Panel A: Clean Accuracy vs. Static Corruption Robustness (mCA) across Phase 1 & Phase 2
- Panel B: The Covariate Shift Paradox (Static BN vs. Test-Time Adapted BN divergence)
- Panel C: The Continental Invariance Phenomenon (German CBM transfer vs. Native Chinese Freq-CBM)
- Panel D: Relative Performance Drop (RPD%) Pareto Robustness Frontier
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Publication styling
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 0.8

# Empirical Data from Master Benchmark
DATA = {
    "P1_BE_Base": {"name": "P1 BE Baseline (DE)", "clean": 89.07, "static": 75.87, "adapt": 31.15, "rpd": 14.82, "ds": "Belgium", "phase": "P1"},
    "P1_BE_CBM":  {"name": "P1 BE CBM (DE)",      "clean": 88.79, "static": 77.65, "adapt": 32.40, "rpd": 12.54, "ds": "Belgium", "phase": "P1"},
    "P1_CN_Base": {"name": "P1 CN Baseline (DE)", "clean": 93.48, "static": 79.57, "adapt": 67.80, "rpd": 14.88, "ds": "China",   "phase": "P1"},
    "P1_CN_CBM":  {"name": "P1 CN CBM (DE)",      "clean": 94.38, "static": 79.24, "adapt": 65.37, "rpd": 16.04, "ds": "China",   "phase": "P1"},
    "P2_BE_Base": {"name": "P2 BE Baseline (Native)", "clean": 93.33, "static": 74.67, "adapt": 40.75, "rpd": 20.00, "ds": "Belgium", "phase": "P2"},
    "P2_BE_FCBM": {"name": "P2 BE Freq-CBM (Native)", "clean": 92.62, "static": 82.08, "adapt": 39.84, "rpd": 11.38, "ds": "Belgium", "phase": "P2"},
    "P2_CN_Base": {"name": "P2 CN Baseline (Native)", "clean": 93.18, "static": 63.07, "adapt": 62.44, "rpd": 32.31, "ds": "China",   "phase": "P2"},
    "P2_CN_FCBM": {"name": "P2 CN Freq-CBM (Native)", "clean": 94.88, "static": 79.78, "adapt": 67.67, "rpd": 15.92, "ds": "China",   "phase": "P2"},
}

def generate_figure16():
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

    labels = [
        "P1 BE\nBase", "P1 BE\nCBM", "P1 CN\nBase", "P1 CN\nCBM",
        "P2 BE\nBase", "P2 BE\nFreq-CBM", "P2 CN\nBase", "P2 CN\nFreq-CBM"
    ]
    keys = list(DATA.keys())
    clean_vals = [DATA[k]["clean"] for k in keys]
    static_vals = [DATA[k]["static"] for k in keys]
    adapt_vals = [DATA[k]["adapt"] for k in keys]
    rpd_vals = [DATA[k]["rpd"] for k in keys]

    # --------------------------------------------------------------------------
    # Panel A: Clean Accuracy vs. Static Robustness (mCA)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(keys))
    width = 0.35

    c_clean = "#2b5c8f"
    c_mca = "#27ae60"

    rects1 = ax1.bar(x - width/2, clean_vals, width, label="Clean Top-1 Accuracy", color=c_clean, alpha=0.88, edgecolor="#1a365d")
    rects2 = ax1.bar(x + width/2, static_vals, width, label="Static Corruption Robustness (mCA)", color=c_mca, alpha=0.90, edgecolor="#1e8449")

    # Annotate winning Phase 2 margins
    ax1.annotate("+7.41% mCA", xy=(5 + width/2, static_vals[5]), xytext=(5 + width/2, static_vals[5] + 5),
                 arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight="bold", color="#c0392b", ha="center")
    ax1.annotate("+16.71% mCA", xy=(7 + width/2, static_vals[7]), xytext=(7 + width/2, static_vals[7] + 5),
                 arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight="bold", color="#c0392b", ha="center")

    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Panel A: Clean Accuracy vs. Static Corruption Robustness (mCA)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(50, 102)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", framealpha=0.95, fontsize=10)

    # --------------------------------------------------------------------------
    # Panel B: The Covariate Shift Paradox (Static vs. Adapted BN)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    diffs = [DATA[k]["adapt"] - DATA[k]["static"] for k in keys]
    colors_b = ["#e74c3c" if d < 0 else "#2ecc71" for d in diffs]

    bars = ax2.bar(x, diffs, width=0.55, color=colors_b, alpha=0.85, edgecolor="#7f1d1d")
    ax2.axhline(0, color="#2c3e50", linewidth=1.2)

    for i, (b, d) in enumerate(zip(bars, diffs)):
        va = "top" if d < 0 else "bottom"
        offset = -2.5 if d < 0 else 1.5
        ax2.text(b.get_x() + b.get_width()/2, d + offset, f"{d:.1f}%",
                 ha="center", va=va, fontsize=9.5, fontweight="bold", color="#2c3e50")

    ax2.set_ylabel("Adaptation Delta: Adapt mCA - Static mCA (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Panel B: The Covariate Shift Paradox (Test-Time BN Collapse)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(-55, 10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.annotate("123:1 Class Imbalance\nInduces Severe Variance Collapse", xy=(0.5, -45), xytext=(1.5, -52),
                 arrowprops=dict(facecolor="#34495e", shrink=0.08, width=1.0, headwidth=5),
                 fontsize=9.5, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#f8d7da", ec="#f5c6cb"))

    # --------------------------------------------------------------------------
    # Panel C: Continental Invariance (German CBM vs. Native Chinese Freq-CBM)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    comp_labels = ["German CBM (P1)\nZero Target Training", "Chinese Freq-CBM (P2)\nNative Training", "Chinese Baseline (P2)\nNative Training"]
    comp_clean = [DATA["P1_CN_CBM"]["clean"], DATA["P2_CN_FCBM"]["clean"], DATA["P2_CN_Base"]["clean"]]
    comp_mca = [DATA["P1_CN_CBM"]["static"], DATA["P2_CN_FCBM"]["static"], DATA["P2_CN_Base"]["static"]]

    x_c = np.arange(3)
    w_c = 0.32
    ax3.bar(x_c - w_c/2, comp_clean, w_c, label="Clean Accuracy", color="#3498db", alpha=0.88, edgecolor="#1f618d")
    ax3.bar(x_c + w_c/2, comp_mca, w_c, label="Corruption Robustness (mCA)", color="#e67e22", alpha=0.90, edgecolor="#b9770e")

    for i in range(3):
        ax3.text(x_c[i] - w_c/2, comp_clean[i] + 0.8, f"{comp_clean[i]:.2f}%", ha="center", fontsize=9.5, fontweight="bold")
        ax3.text(x_c[i] + w_c/2, comp_mca[i] + 0.8, f"{comp_mca[i]:.2f}%", ha="center", fontsize=9.5, fontweight="bold")

    # Highlighting the 0.54% delta
    ax3.annotate("Negligible Gap:\nΔ = 0.54% mCA\n(Universal Concept Invariance)", xy=(0.5, 79.5), xytext=(0.5, 68),
                 arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=1.5),
                 fontsize=9.5, fontweight="bold", color="#8e44ad", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f4ecf7", ec="#d2b4de"))

    ax3.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax3.set_title("Panel C: Continental Invariance on Chinese Roadways", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(comp_labels, fontsize=10)
    ax3.set_ylim(55, 102)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.legend(loc="lower left", framealpha=0.95, fontsize=10)

    # --------------------------------------------------------------------------
    # Panel D: Relative Performance Drop (RPD%) Pareto Robustness Frontier
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Scatter plot: Clean Acc vs RPD% (Lower RPD is better)
    markers = {"Belgium": "o", "China": "s"}
    colors_d = {
        "P1_BE_Base": "#7f8c8d", "P1_BE_CBM": "#2980b9",
        "P1_CN_Base": "#95a5a6", "P1_CN_CBM": "#8e44ad",
        "P2_BE_Base": "#e74c3c", "P2_BE_FCBM": "#27ae60",
        "P2_CN_Base": "#d35400", "P2_CN_FCBM": "#16a085",
    }

    for k in keys:
        d = DATA[k]
        m = markers[d["ds"]]
        c = colors_d[k]
        ax4.scatter(d["clean"], d["rpd"], s=180, color=c, marker=m, edgecolor="#2c3e50", linewidth=1.2, zorder=4)
        ax4.text(d["clean"] + 0.15, d["rpd"] + 0.5, d["name"], fontsize=9, fontweight="bold", color="#2c3e50")

    # Ideal frontier zone (High Clean, Low RPD)
    ax4.axvspan(92.0, 95.5, ymin=0.0, ymax=0.35, color="#2ecc71", alpha=0.12, label="Optimal Robustness Frontier")
    ax4.text(93.8, 12.0, "Optimal Pareto Region\n(High Clean + Low RPD)", fontsize=10, fontweight="bold", color="#1e8449", ha="center")

    ax4.set_xlabel("Clean Top-1 Accuracy (%) [Higher is Better]", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Relative Performance Drop (RPD %) [Lower is Better]", fontsize=12, fontweight="bold")
    ax4.set_title("Panel D: Relative Performance Drop vs. Clean Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax4.set_xlim(88.0, 95.5)
    ax4.set_ylim(8.0, 35.0)
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", framealpha=0.95, fontsize=10)

    # Save figure
    out_path = Path("figs/fig16_cross_dataset_robustness_benchmark.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated {out_path} (300 DPI)")

if __name__ == "__main__":
    generate_figure16()

