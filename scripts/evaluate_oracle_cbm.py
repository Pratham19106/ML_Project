"""
evaluate_oracle_cbm.py
======================
Oracle Concept Bottleneck Model analysis.

Evaluates whether the 22-dim ground-truth concept bottleneck (shape + colour +
border + symbol_category) is *sufficient* to uniquely identify all 43 GTSRB
classes ? i.e., can a linear probe on perfect concept vectors achieve 100%
accuracy?

This script has ZERO torch dependency and runs in < 5 seconds on CPU.

Usage
-----
    python scripts/evaluate_oracle_cbm.py
    python scripts/evaluate_oracle_cbm.py --csv-path /custom/path/class_concepts.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Allow import of src/ from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.concept_encoder import (
    BOTTLENECK_DIM,
    N_BORDER,
    N_COLOUR,
    N_SHAPE,
    N_SYMCAT,
    encode_all_classes,
    get_oracle_concept_matrix,
    load_class_concepts,
)


# ??????????????????????????????????????????????????????????????????????????????

def find_degenerate_pairs(X: np.ndarray, y: np.ndarray):
    """Return list of (class_id_a, class_id_b) pairs that share identical rows."""
    degenerate = []
    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            if np.array_equal(X[i], X[j]):
                degenerate.append((int(y[i]), int(y[j])))
    return degenerate


def analyse_axis_separability(X: np.ndarray, axis_name: str, start: int, end: int):
    """Count unique values in the one-hot block [start:end]."""
    block = X[:, start:end]
    indices = np.argmax(block, axis=1)
    unique_vals, counts = np.unique(indices, return_counts=True)
    return {
        "axis": axis_name,
        "n_values_present": int(len(unique_vals)),
        "n_values_vocab": int(end - start),
        "distribution": {str(int(v)): int(c) for v, c in zip(unique_vals, counts)},
    }


def run_oracle_probe(X: np.ndarray, y: np.ndarray):
    """
    Train a logistic regression on the Oracle concept matrix.
    Since the X matrix has only 43 unique rows (at most), we train on all and
    evaluate on all ? the Oracle setting assumes perfect concept access.
    """
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    return clf, y_pred, acc


def main(args):
    print("=" * 70)
    print("Oracle CBM Analysis ? GTSRB 43-Class Concept Bottleneck")
    print("=" * 70)

    # 1. Load data
    rows = load_class_concepts(args.csv_path)
    enc  = encode_all_classes(args.csv_path)
    X, y = get_oracle_concept_matrix(args.csv_path)

    print(f"\nOK Loaded {len(rows)} class entries from class_concepts.csv")
    print(f"OK Bottleneck dimension: {BOTTLENECK_DIM}  "
          f"(shape={N_SHAPE}, colour={N_COLOUR}, border={N_BORDER}, symcat={N_SYMCAT})")
    print(f"OK Oracle matrix shape: X={X.shape}, y={y.shape}")

    # 2. Uniqueness analysis
    print("\n?? Concept Vector Uniqueness ??????????????????????????????????????")
    unique_rows = np.unique(X, axis=0)
    n_unique = len(unique_rows)
    n_total  = len(y)
    degenerate_pairs = find_degenerate_pairs(X, y)

    if degenerate_pairs:
        print(f"  WARN  {n_total - n_unique} degenerate collision(s) detected!")
        for a, b in degenerate_pairs:
            name_a = rows[a]["class_name"]
            name_b = rows[b]["class_name"]
            print(f"     Class {a} ({name_a}) == Class {b} ({name_b})")
        vocabulary_sufficient = False
    else:
        print(f"  OK All {n_total} classes have DISTINCT concept vectors.")
        vocabulary_sufficient = True

    # 3. Axis-level analysis
    print("\n?? Axis Coverage ??????????????????????????????????????????????????")
    axes = [
        ("shape",            0,  N_SHAPE),
        ("colour_profile",   N_SHAPE,           N_SHAPE + N_COLOUR),
        ("border_type",      N_SHAPE + N_COLOUR, N_SHAPE + N_COLOUR + N_BORDER),
        ("symbol_category",  N_SHAPE + N_COLOUR + N_BORDER, BOTTLENECK_DIM),
    ]
    axis_stats = []
    for name, start, end in axes:
        stat = analyse_axis_separability(X, name, start, end)
        axis_stats.append(stat)
        coverage = stat["n_values_present"] / stat["n_values_vocab"]
        print(f"  {name:20s}: {stat['n_values_present']}/{stat['n_values_vocab']} "
              f"vocabulary values used  ({coverage:.0%} coverage)")

    # 4. Oracle linear probe
    print("\n?? Oracle Linear Probe (Logistic Regression) ??????????????????????")
    clf, y_pred, acc = run_oracle_probe(X, y)

    print(f"  Oracle accuracy (ground-truth concepts -> class): {acc * 100:.2f}%")
    if acc == 1.0:
        print("  OK Perfect classification: the 22-dim vocabulary is SUFFICIENT to")
        print("    uniquely identify all 43 classes without symbol_content.")
        probe_result = "PERFECT (100%)"
    else:
        mismatches = np.where(y != y_pred)[0]
        print(f"  FAIL Imperfect classification: {len(mismatches)} misclassified classes")
        for m in mismatches:
            name_true = rows[int(y[m])]["class_name"]
            name_pred = rows[int(y_pred[m])]["class_name"]
            print(f"     True={y[m]} ({name_true})  ->  Pred={y_pred[m]} ({name_pred})")
        probe_result = f"{acc * 100:.2f}%"

    # 5. Theoretical ceiling note
    print("\n?? Theoretical Ceiling ????????????????????????????????????????????")
    print("  The Oracle CBM theoretical clean accuracy ceiling equals the Oracle")
    print("  probe accuracy above.  A Sequential/Joint CBM that learns to predict")
    print("  concepts from images can approach ? but not exceed ? this ceiling.")
    if vocabulary_sufficient:
        print("  -> Ceiling ? 100%  (vocabulary is degenerate-free).")
    else:
        print("  -> Ceiling < 100%  (vocabulary collisions create Bayes error).")

    # 6. Save report
    report = {
        "n_classes":                  n_total,
        "bottleneck_dim":             BOTTLENECK_DIM,
        "vocabulary_sufficient":      vocabulary_sufficient,
        "n_unique_concept_vectors":   n_unique,
        "degenerate_pairs":           degenerate_pairs,
        "oracle_probe_accuracy":      round(float(acc), 6),
        "oracle_probe_result":        probe_result,
        "axis_coverage":              axis_stats,
        "theoretical_clean_ceiling":  "100%" if vocabulary_sufficient else probe_result,
    }

    out_dir = PROJECT_ROOT / "reports" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oracle_cbm_analysis.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nOK Report saved -> {out_path}")
    print("\n" + "=" * 70)
    print("Oracle CBM Analysis Complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle CBM analysis for GTSRB")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to class_concepts.csv (default: auto-discovered)",
    )
    args = parser.parse_args()
    sys.exit(main(args))
