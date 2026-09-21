"""Build Master Dataset Script (GTSRB Master v1)

Implements the frozen Master Dataset specification:
1. Track-aware, group-stratified train/val split (80/20) by (class_id + track_id) to eliminate video-frame leakage.
2. Preserves official Test.csv strictly untouched for final held-out clean testing.
3. Preprocessing: ROI crop with 10% context margin -> proportional letterbox resize -> 64x64 zero-padded canvas.
4. Schema-complete metadata: image_id, source_split, split, group_id, ROI metadata, concepts placeholder.
5. Saves manifests to data/master_v1/metadata/ and canonical images to data/master_v1/images/.
6. Generates configs/dataset_version.md with record counts and SHA-256 manifest hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW_REQUIRED_COLUMNS = [
    "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"
]


def load_raw_manifest(path: Path) -> pd.DataFrame:
    """Loads GTSRB raw CSV, handling header anomaly if present."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "German Traffic Sign" in first_line:
        df = pd.read_csv(path, skiprows=1, names=RAW_REQUIRED_COLUMNS)
    else:
        df = pd.read_csv(path)

    return df


def extract_train_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts sign_id, track_id, and frame_id from GTSRB Train Path.
    Expected pattern: Train/<class_dir>/<sign_id>_<track_id>_<frame_id>.png
    """
    pattern = r"Train/(\d+)/(\d+)_(\d+)_(\d+)\.png"
    extracted = df["Path"].str.extract(pattern)
    extracted.columns = ["dir_class", "sign_id", "track_id", "frame_id"]

    df = df.copy()
    df["dir_class"] = extracted["dir_class"]
    df["sign_id"] = extracted["sign_id"]
    df["track_id"] = extracted["track_id"]
    df["frame_id"] = extracted["frame_id"]
    df["group_id"] = df["ClassId"].astype(str) + "_" + df["track_id"].astype(str)
    return df


def stratified_group_train_val_split(
    train_df: pd.DataFrame, train_ratio: float = 0.80, seed: int = 42
) -> pd.DataFrame:
    """Splits training tracks into train and validation sets per class.
    Guarantees that no track has frames across both train and validation.
    """
    rng = np.random.default_rng(seed)
    train_df = train_df.copy()
    train_df["split"] = ""

    for class_id, group in train_df.groupby("ClassId"):
        unique_tracks = group["track_id"].unique()
        # Shuffle tracks deterministically for this class
        shuffled_tracks = rng.permutation(unique_tracks)
        n_tracks = len(shuffled_tracks)

        # For classes with multiple tracks, allocate at least 1 track to validation
        if n_tracks > 1:
            n_train_tracks = int(round(train_ratio * n_tracks))
            n_train_tracks = min(max(1, n_train_tracks), n_tracks - 1)
        else:
            n_train_tracks = 1

        train_tracks = set(shuffled_tracks[:n_train_tracks])
        val_tracks = set(shuffled_tracks[n_train_tracks:])

        is_train = group["track_id"].isin(train_tracks)
        is_val = group["track_id"].isin(val_tracks)

        train_df.loc[group[is_train].index, "split"] = "train"
        train_df.loc[group[is_val].index, "split"] = "val"

    return train_df


def preprocess_and_save_image(
    src_img_path: Path,
    dest_img_path: Path,
    roi: tuple[int, int, int, int],
    target_size: int = 64,
    margin_ratio: float = 0.10,
) -> None:
    """Crops to ROI + 10% margin, proportionally letterboxes, and zero-pads to target_size square."""
    with Image.open(src_img_path) as img:
        img = img.convert("RGB")
        w_img, h_img = img.size

        x1, y1, x2, y2 = roi
        roi_w = x2 - x1 + 1
        roi_h = y2 - y1 + 1

        margin_x = int(round(margin_ratio * roi_w))
        margin_y = int(round(margin_ratio * roi_h))

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(w_img - 1, x2 + margin_x)
        crop_y2 = min(h_img - 1, y2 + margin_y)

        # In PIL, crop box is (left, upper, right, lower) with exclusive upper bounds
        cropped = img.crop((crop_x1, crop_y1, crop_x2 + 1, crop_y2 + 1))
        crop_w, crop_h = cropped.size

        # Proportional scale factor
        scale = min(target_size / crop_w, target_size / crop_h)
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))

        resized = cropped.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        # Symmetric zero-padding onto target canvas
        canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))

        dest_img_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest_img_path, format="PNG", optimize=True)


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Construct GTSRB Master Dataset v1.")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB folder (default: german)")
    parser.add_argument("--output-dir", type=str, default="data/master_v1", help="Output directory for master dataset")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible train/val group split")
    parser.add_argument("--metadata-only", action="store_true", help="Generate manifests only without rendering images")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    meta_dir = out_dir / "metadata"
    img_dir = out_dir / "images"
    cfg_dir = out_dir / "configs"
    concept_dir = out_dir / "concepts"

    meta_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    concept_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("BUILDING GTSRB MASTER DATASET (v1)")
    print(f"Raw Source  : {data_root}")
    print(f"Master Root : {out_dir}")
    print(f"Master Seed : {args.seed}")
    print(f"Render Imgs : {'No (metadata only)' if args.metadata_only else 'Yes (64x64 letterboxed)'}")
    print("=" * 75)

    # 1. Load raw manifests
    train_raw = load_raw_manifest(data_root / "Train.csv")
    test_raw = load_raw_manifest(data_root / "Test.csv")
    print(f"Raw Train samples: {len(train_raw):,}")
    print(f"Raw Test samples : {len(test_raw):,}")

    # 2. Extract train track identifiers and perform group-stratified split
    train_identified = extract_train_identifiers(train_raw)
    train_split_df = stratified_group_train_val_split(train_identified, train_ratio=0.80, seed=args.seed)

    n_train = (train_split_df["split"] == "train").sum()
    n_val = (train_split_df["split"] == "val").sum()
    print(f"Track-stratified split -> Train: {n_train:,} | Validation: {n_val:,}")

    # 3. Load concepts template / mappings if available
    concept_csv = concept_dir / "class_concepts.csv"
    if concept_csv.is_file():
        concepts_df = pd.read_csv(concept_csv)
    else:
        concepts_df = pd.DataFrame({"class_id": list(range(43))})

    concept_cols = ["shape", "colour_profile", "border_type", "symbol_category", "symbol_content"]
    for c in concept_cols:
        if c not in concepts_df.columns:
            concepts_df[c] = ""
    if "class_name" not in concepts_df.columns:
        concepts_df["class_name"] = ""

    concept_map = concepts_df.set_index("class_id")[["class_name"] + concept_cols].to_dict("index")

    # 4. Construct standardized rows for Train and Val
    master_records: list[dict[str, Any]] = []

    for idx, row in train_split_df.iterrows():
        c_id = int(row["ClassId"])
        t_id = row["track_id"]
        f_id = row["frame_id"]
        split = row["split"]
        c_info = concept_map.get(c_id, {})

        image_id = f"train_c{c_id:02d}_t{t_id}_f{f_id}"
        canonical_rel = f"images/{split}/{image_id}.png"

        master_records.append({
            "image_id": image_id,
            "source_split": "train",
            "split": split,
            "class_id": c_id,
            "class_name": c_info.get("class_name", ""),
            "group_id": f"c{c_id:02d}_t{t_id}",
            "source_relative_path": str(row["Path"]).replace("\\", "/"),
            "canonical_image_path": canonical_rel,
            "shape": c_info.get("shape", ""),
            "colour_profile": c_info.get("colour_profile", ""),
            "border_type": c_info.get("border_type", ""),
            "symbol_category": c_info.get("symbol_category", ""),
            "symbol_content": c_info.get("symbol_content", ""),
            "roi_x1": int(row["Roi.X1"]),
            "roi_y1": int(row["Roi.Y1"]),
            "roi_x2": int(row["Roi.X2"]),
            "roi_y2": int(row["Roi.Y2"]),
            "roi_coordinate_convention": "inclusive_xyxy_zero_indexed",
            "orig_width": int(row["Width"]),
            "orig_height": int(row["Height"]),
            "preprocessing_version": "v1_roi_margin10_letterbox64",
        })

    # 5. Construct standardized rows for Test (Held-out benchmark)
    for idx, row in test_raw.iterrows():
        c_id = int(row["ClassId"])
        c_info = concept_map.get(c_id, {})
        image_id = f"test_{idx:05d}"
        canonical_rel = f"images/test/{image_id}.png"

        master_records.append({
            "image_id": image_id,
            "source_split": "test",
            "split": "test",
            "class_id": c_id,
            "class_name": c_info.get("class_name", ""),
            "group_id": f"test_{idx:05d}",
            "source_relative_path": str(row["Path"]).replace("\\", "/"),
            "canonical_image_path": canonical_rel,
            "shape": c_info.get("shape", ""),
            "colour_profile": c_info.get("colour_profile", ""),
            "border_type": c_info.get("border_type", ""),
            "symbol_category": c_info.get("symbol_category", ""),
            "symbol_content": c_info.get("symbol_content", ""),
            "roi_x1": int(row["Roi.X1"]),
            "roi_y1": int(row["Roi.Y1"]),
            "roi_x2": int(row["Roi.X2"]),
            "roi_y2": int(row["Roi.Y2"]),
            "roi_coordinate_convention": "inclusive_xyxy_zero_indexed",
            "orig_width": int(row["Width"]),
            "orig_height": int(row["Height"]),
            "preprocessing_version": "v1_roi_margin10_letterbox64",
        })

    master_df = pd.DataFrame(master_records)

    # 6. Save split CSVs
    train_manifest = meta_dir / "train.csv"
    val_manifest = meta_dir / "val.csv"
    test_manifest = meta_dir / "test.csv"
    master_manifest = meta_dir / "master_metadata.csv"

    master_df[master_df["split"] == "train"].to_csv(train_manifest, index=False)
    master_df[master_df["split"] == "val"].to_csv(val_manifest, index=False)
    master_df[master_df["split"] == "test"].to_csv(test_manifest, index=False)
    master_df.to_csv(master_manifest, index=False)

    print("\nSaved manifests to metadata/:")
    print(f"  - train.csv           : {len(master_df[master_df['split'] == 'train']):,} rows")
    print(f"  - val.csv             : {len(master_df[master_df['split'] == 'val']):,} rows")
    print(f"  - test.csv            : {len(master_df[master_df['split'] == 'test']):,} rows")
    print(f"  - master_metadata.csv : {len(master_df):,} rows")

    # 7. Render images if requested
    if not args.metadata_only:
        print("\nRendering canonical 64x64 letterboxed images...")
        pbar = tqdm(master_df.iterrows(), total=len(master_df), desc="Preprocessing images", unit="img")
        for _, r in pbar:
            src_p = data_root / r["source_relative_path"]
            dst_p = out_dir / r["canonical_image_path"]
            roi = (r["roi_x1"], r["roi_y1"], r["roi_x2"], r["roi_y2"])
            preprocess_and_save_image(src_p, dst_p, roi, target_size=64, margin_ratio=0.10)
        print("Canonical image generation complete!")

    # 8. Compute checksums and write dataset_version.md
    hashes = {
        "train.csv": compute_sha256(train_manifest),
        "val.csv": compute_sha256(val_manifest),
        "test.csv": compute_sha256(test_manifest),
        "master_metadata.csv": compute_sha256(master_manifest),
    }

    version_doc = f"""# Master Dataset Specification and Version Record (master_v1)

- **Creation Timestamp (UTC):** {datetime.now(timezone.utc).isoformat()}
- **Raw Data Source:** `{data_root}`
- **Master Random Seed:** `{args.seed}`
- **Split Protocol:** Stratified Group Split by `(ClassId + TrackId)`
- **Preprocessing Version:** `v1_roi_margin10_letterbox64`
  - ROI Crop: 10% contextual margin on all sides (clamped to image dimensions).
  - Resizing: Proportional bilinear scaling preserving aspect ratio.
  - Padding: Symmetric zero-padding to 64x64 canvas.

## Dataset Partition Summary

| Split | Number of Images | Description |
| :--- | :--- | :--- |
| **Train** | `{len(master_df[master_df['split'] == 'train']):,}` | 80% of tracks per class for training |
| **Validation** | `{len(master_df[master_df['split'] == 'val']):,}` | 20% of tracks per class (zero video-frame leakage) |
| **Test** | `{len(master_df[master_df['split'] == 'test']):,}` | Official GTSRB held-out test set (strictly untouched) |
| **Total** | `{len(master_df):,}` | Total annotated samples |

## Manifest SHA-256 Checksums

```text
train.csv:           {hashes['train.csv']}
val.csv:             {hashes['val.csv']}
test.csv:            {hashes['test.csv']}
master_metadata.csv: {hashes['master_metadata.csv']}
```
"""
    version_file = cfg_dir / "dataset_version.md"
    with open(version_file, "w", encoding="utf-8") as f:
        f.write(version_doc)

    print(f"\nVersion record generated: {version_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()

