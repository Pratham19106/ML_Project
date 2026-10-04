"""
Packages Experiment 5 (Hybrid CBM + AugMix) and full codebase into a compact zip (~2 MB)
so it can be uploaded and unzipped directly onto Colab without re-uploading gigabytes of images.
"""

import os
import zipfile
from pathlib import Path

ROOT_FILES = [
    "train_hybrid.py",
    "hybrid_experiment_colab.ipynb",
    "train_cbm.py",
    "cbm_experiment_colab.ipynb",
    "evaluate_robustness.py",
    "progress.md",
    "AGENT_PROJECT_CONTEXT.md",
    "research_content.md",
    "RESEARCH_GUIDELINES.md",
    "COLLABORATOR_AND_AGENT_GUIDE.md",
]

output_zip = Path("exp5_hybrid_code.zip")
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ROOT_FILES:
        p = Path(f)
        if p.exists():
            zf.write(p, arcname=f)
            print(f"Added root file: {f}")

    # Include all code files and metadata directories recursively
    for folder in ["src", "scripts", "data/master_v1"]:
        for root, _, files in os.walk(folder):
            if "__pycache__" in root or ".pytest_cache" in root:
                continue
            for fname in files:
                p = Path(root) / fname
                arcname = p.as_posix()
                zf.write(p, arcname=arcname)
        print(f"Added {folder} recursively")

print(f"\nCreated: {output_zip.resolve()} ({output_zip.stat().st_size / (1024 * 1024):.2f} MB)")
