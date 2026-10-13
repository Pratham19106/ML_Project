"""
Creates a lightweight zip package (~60 KB) for running the Test-Time BN-Adaptation
benchmark on Google Colab without re-uploading the dataset.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ROOT_FILES = [
    "notebooks/bn_adaptation_experiment_colab.ipynb",
    "scripts/run_all_bn_experiments.py",
    "scripts/evaluate_full_benchmark_suite.py",
    "scripts/evaluate_test_time_adaptation.py",
    "progress.md",
    "research_content.md",
]

output_zip = PROJECT_ROOT / "bn_adaptation_code.zip"

with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ROOT_FILES:
        p = PROJECT_ROOT / f
        if p.exists():
            zf.write(p, arcname=f)
            print(f"Added root/script file: {f}")

    for folder in ["src", "scripts", "data/master_v1"]:
        for root, _, files in os.walk(PROJECT_ROOT / folder):
            if "__pycache__" in root or ".pytest_cache" in root:
                continue
            for fname in files:
                p = Path(root) / fname
                arcname = p.relative_to(PROJECT_ROOT).as_posix()
                zf.write(p, arcname=arcname)
        print(f"Added folder: {folder}")

print(f"\nCreated: {output_zip.resolve()} ({output_zip.stat().st_size / 1024:.1f} KB)")

