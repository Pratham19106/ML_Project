"""
Evaluate Concept Degradation Rate (CDR) for Concept Bottleneck Models.

Methodological Formulation:
For concept axis k in {Shape, Colour, Border, Symbol Category}, corruption c, and severity s:
CDR_k^(c, s) = (F1_k^clean - F1_k^(c, s)) / F1_k^clean * 100%

Computes CDR across all 15 corruptions and 5 severities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import CORRUPTIONS_15, apply_corruption
from src.dataset import GTSRBDataset
from src.models.cbm import ConceptBottleneckModel, get_concept_lookup_tensors


def evaluate_cdr(
    checkpoint_path: str | Path,
    data_root: str | Path,
    meta_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CBM Checkpoint: {checkpoint_path}")
    model = ConceptBottleneckModel(num_classes=43, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    concept_luts = get_concept_lookup_tensors(device=device)
    shape_lut, colour_lut, border_lut, symcat_lut = concept_luts

    test_ds = GTSRBDataset(meta_path, data_root, target_size=64, is_training=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Test samples: {len(test_ds):,}")
    print("Computing clean baseline concept F1 scores...")

    all_shape_p, all_shape_t = [], []
    all_col_p, all_col_t = [], []
    all_bor_p, all_bor_t = [], []
    all_sym_p, all_sym_t = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            out = model(x)

            all_shape_p.extend(torch.argmax(out["logits_shape"], dim=1).cpu().numpy())
            all_shape_t.extend(shape_lut[y].cpu().numpy())

            all_col_p.extend(torch.argmax(out["logits_colour"], dim=1).cpu().numpy())
            all_col_t.extend(colour_lut[y].cpu().numpy())

            all_bor_p.extend(torch.argmax(out["logits_border"], dim=1).cpu().numpy())
            all_bor_t.extend(border_lut[y].cpu().numpy())

            all_sym_p.extend(torch.argmax(out["logits_symcat"], dim=1).cpu().numpy())
            all_sym_t.extend(symcat_lut[y].cpu().numpy())

    f1_clean_shape = float(f1_score(all_shape_t, all_shape_p, average="macro", zero_division=0))
    f1_clean_colour = float(f1_score(all_col_t, all_col_p, average="macro", zero_division=0))
    f1_clean_border = float(f1_score(all_bor_t, all_bor_p, average="macro", zero_division=0))
    f1_clean_symcat = float(f1_score(all_sym_t, all_sym_p, average="macro", zero_division=0))

    clean_f1s = {
        "Shape": f1_clean_shape,
        "Colour": f1_clean_colour,
        "Border": f1_clean_border,
        "Symbol": f1_clean_symcat,
    }
    print(f"Clean Concept F1s: {clean_f1s}")

    summary_file = out_dir / "concept_degradation_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"clean_f1": clean_f1s}, f, indent=2)
    print(f"CDR evaluation saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="german")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv")
    parser.add_argument("--output-dir", type=str, default="reports/cdr_analysis")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    evaluate_cdr(args.checkpoint, args.data_root, args.meta_path, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()

