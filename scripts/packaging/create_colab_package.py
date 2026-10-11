"""Packages the GTSRB master dataset, images, and project files into a single zip for Google Drive / Colab."""

import os
import zipfile
from pathlib import Path
from tqdm import tqdm

OUTPUT_ZIP = Path("ML_Project_Data.zip")

INCLUDE_DIRS = [
    Path("german"),
    Path("data/master_v1"),
    Path("src"),
    Path("notebooks"),
    Path("scripts"),
]

INCLUDE_FILES = [
    Path(".cursorrules"),
    Path("COLLABORATOR_AND_AGENT_GUIDE.md"),
    Path("RESEARCH_GUIDELINES.md"),
    Path("research_content.md"),
    Path("progress.md"),
    Path("AGENT_PROJECT_CONTEXT.md"),
    Path("Structured_Research_Proposal.md"),
    Path("train_baseline.py"),
    Path("train_augmix.py"),
    Path("train_frequency.py"),
    Path("train_cbm.py"),
    Path("train_freq_cbm.py"),
    Path("train_hybrid.py"),
    Path("evaluate_robustness.py"),
    Path("build_master_dataset.py"),
    Path("corruption_code.py"),
]


def collect_files() -> list[tuple[Path, str]]:
    """Gathers all files to be packaged with their relative archive names."""
    files_to_zip: list[tuple[Path, str]] = []

    # Include explicit root files
    for file_path in INCLUDE_FILES:
        if file_path.is_file():
            files_to_zip.append((file_path, file_path.name))

    # Include directories recursively
    for dir_path in INCLUDE_DIRS:
        if not dir_path.is_dir():
            print(f"Warning: Directory not found: {dir_path}")
            continue
        for root, _, filenames in os.walk(dir_path):
            for fname in filenames:
                full_path = Path(root) / fname
                # Skip temporary files, checkpoints, and archives
                if fname.endswith(".zip") or "__pycache__" in root or ".ipynb_checkpoints" in root:
                    continue
                arc_name = full_path.as_posix()
                files_to_zip.append((full_path, arc_name))

    return files_to_zip


def main():
    print("=" * 70)
    print("PACKAGING DATASET FOR GOOGLE DRIVE / COLAB")
    print(f"Output File: {OUTPUT_ZIP.resolve()}")
    print("=" * 70)

    files = collect_files()
    print(f"Total files to package: {len(files):,}")

    if OUTPUT_ZIP.exists():
        print(f"Removing existing {OUTPUT_ZIP.name}...")
        OUTPUT_ZIP.unlink()

    print("\nCompressing files into archive...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        pbar = tqdm(files, desc="Zipping", unit="file")
        for full_path, arc_name in pbar:
            zipf.write(full_path, arc_name)

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print("SUCCESSFULLY CREATED COLAB PACKAGE!")
    print(f"Archive Name : {OUTPUT_ZIP.name}")
    print(f"Archive Size : {zip_size_mb:.2f} MB")
    print(f"Full Path    : {OUTPUT_ZIP.resolve()}")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Upload 'ML_Project_Data.zip' to your Google Drive.")
    print("2. In Colab, run:")
    print("   !cp /content/drive/MyDrive/ML_Project_Data.zip /content/")
    print("   !unzip -q /content/ML_Project_Data.zip -d /content/ML_Project")


if __name__ == "__main__":
    main()

