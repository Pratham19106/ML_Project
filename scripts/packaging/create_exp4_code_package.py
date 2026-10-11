"""
Packages only the newly added Experiment 4 (CBM) code files into a tiny zip (~50 KB)
so it can be uploaded and unzipped directly onto Colab without re-uploading gigabytes of images.
"""

import os
import zipfile
from pathlib import Path

ROOT_FILES = [
    "train_cbm.py",
    "cbm_experiment_colab.ipynb",
    "evaluate_robustness.py",
    "progress.md",
]

output_zip = Path("exp4_cbm_code.zip")
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ROOT_FILES:
        p = Path(f)
        if p.exists():
            zf.write(p, arcname=f)
            print(f"Added root file: {f}")


    # Include all code files and directories
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
