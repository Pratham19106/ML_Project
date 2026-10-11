"""
Packages Concept Degradation Rate (CDR) analysis scripts and notebooks into a zip (~50 KB)
for Google Colab execution.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT_FILES = [
    "train_cbm.py",
    "cbm_experiment_colab.ipynb",
    "evaluate_robustness.py",
    "progress.md",
]

output_zip = Path("cdr_analysis_code.zip")
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ROOT_FILES:
        p = Path(f)
        if p.exists():
            zf.write(p, arcname=f)
            print(f"Added root file: {f}")

    for folder in ["src", "scripts", "data/master_v1"]:
        for root, _, files in os.walk(folder):
            if "__pycache__" in root or ".pytest_cache" in root:
                continue
            for fname in files:
                p = Path(root) / fname
                arcname = p.as_posix()
                zf.write(p, arcname=arcname)
        print(f"Added {folder} recursively")

print(f"\nCreated: {output_zip.resolve()} ({output_zip.stat().st_size / 1024:.1f} KB)")
