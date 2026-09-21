"""GTSRB Dataset Audit Script (Local Runner)

Performs a read-only scientific audit of the GTSRB dataset manifests and images:
1. Missing values and duplicates in Train.csv and Test.csv.
2. Numeric column types, class IDs (0-42), and ROI validity.
3. Train path class folder vs ClassId consistency.
4. Missing, unreadable, or corrupted image files (Pillow verification + decoding).
5. Dimension agreement between CSV metadata and actual image dimensions.
6. Reports saved to reports/dataset_audit/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

REQUIRED_COLUMNS = [
    "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"
]
NUMERIC_COLUMNS = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]


def resolve_data_root(custom_path: str | None = None) -> Path:
    """Finds the GTSRB directory in the workspace or from an argument."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.is_dir() and (p / "Test.csv").is_file():
            return p
        raise FileNotFoundError(f"Provided data-dir does not contain GTSRB: {custom_path}")

    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "german",
        cwd / "archive",
        Path("d:/ML_Project/german"),
        Path("d:/ML_Project/archive"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "Test.csv").is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Could not locate GTSRB dataset folder. Checked: {[str(c) for c in candidates]}"
    )


def load_manifest(name: str, path: Path) -> pd.DataFrame:
    """Loads Train.csv or Test.csv, handling corrupted header lines gracefully."""
    if not path.is_file():
        raise FileNotFoundError(f"{name} manifest not found: {path}")

    # Inspect first line in case title replaced the CSV header
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "German Traffic Sign Recognition Benchmark" in first_line:
        print(f"[{name}] Detected non-header title in line 1. Loading with standard column names...")
        frame = pd.read_csv(path, skiprows=1, names=REQUIRED_COLUMNS)
    else:
        frame = pd.read_csv(path)

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    frame = frame.copy()
    frame["manifest"] = name
    frame["row_number"] = range(2, len(frame) + 2)
    return frame


def issue_rows(frame: pd.DataFrame, mask: pd.Series, issue_type: str, details: str) -> list[dict[str, Any]]:
    """Helper to collect audit issues from a boolean series mask."""
    cols = ["manifest", "row_number", "Path", "ClassId"]
    records = frame.loc[mask, cols].to_dict("records")
    for r in records:
        r["issue_type"] = issue_type
        r["details"] = details
    return records


def metadata_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Audits the CSV manifest metadata for integrity."""
    audited = frame.copy()
    issues: list[dict[str, Any]] = []

    # 1. Null check
    null_mask = audited[REQUIRED_COLUMNS].isna().any(axis=1)
    issues.extend(issue_rows(audited, null_mask, "missing_required_value", "At least one required CSV field is null."))

    # 2. Numeric conversion
    for col in NUMERIC_COLUMNS:
        num = pd.to_numeric(audited[col], errors="coerce")
        invalid_num = audited[col].notna() & num.isna()
        issues.extend(issue_rows(audited, invalid_num, "invalid_numeric_value", f"{col} is not numeric."))
        audited[f"_numeric_{col}"] = num

    width = audited["_numeric_Width"]
    height = audited["_numeric_Height"]
    x1 = audited["_numeric_Roi.X1"]
    y1 = audited["_numeric_Roi.Y1"]
    x2 = audited["_numeric_Roi.X2"]
    y2 = audited["_numeric_Roi.Y2"]
    class_id = audited["_numeric_ClassId"]

    # 3. Positive dimensions
    inv_size = (width <= 0) | (height <= 0)
    issues.extend(issue_rows(audited, inv_size.fillna(False), "invalid_image_dimensions", "Width and height must be > 0."))

    # 4. Valid ROI endpoints
    inv_roi = (x1 < 0) | (y1 < 0) | (x2 < x1) | (y2 < y1) | (x2 >= width) | (y2 >= height)
    issues.extend(issue_rows(audited, inv_roi.fillna(False), "invalid_roi", "ROI coordinates outside image dimensions or inverted."))

    # 5. ClassId validity (0 to 42)
    inv_class = class_id.isna() | (class_id < 0) | (class_id > 42) | (class_id % 1 != 0)
    issues.extend(issue_rows(audited, inv_class.fillna(False), "invalid_class_id", "ClassId must be integer in range [0, 42]."))

    # 6. Duplicate relative paths
    dup_paths = audited["Path"].duplicated(keep=False)
    issues.extend(issue_rows(audited, dup_paths, "duplicate_manifest_path", "Relative path occurs multiple times in manifest."))

    # 7. Safe relative path prefix
    manifest_type = audited["manifest"].iat[0]
    expected_prefix = "Train/" if manifest_type == "train" else "Test/"
    paths_str = audited["Path"].astype("string")
    unsafe = ~paths_str.str.startswith(expected_prefix) | paths_str.str.contains(r"(?:^|[\\/])\.\.(?:[\\/]|$)", regex=True)
    issues.extend(issue_rows(audited, unsafe.fillna(True), "unexpected_or_unsafe_path", f"Path must start with {expected_prefix} and avoid traversal."))

    # 8. Train folder class matches ClassId
    if manifest_type == "train":
        extracted_class = paths_str.str.extract(r"^Train/(\d+)/", expand=False)
        extracted_num = pd.to_numeric(extracted_class, errors="coerce")
        mismatch = extracted_num.ne(class_id)
        issues.extend(issue_rows(audited, mismatch.fillna(True), "path_class_mismatch", "Class folder in path does not match ClassId column."))

    return audited, issues


def image_audit(
    data_root: Path, frame: pd.DataFrame, max_images: int | None = None
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Verifies that image files exist, can be opened/decoded, and match declared dimensions."""
    rows = frame if max_images is None else frame.head(max_images)
    manifest_name = frame["manifest"].iat[0]
    issues: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    pbar = tqdm(rows.iterrows(), total=len(rows), desc=f"Auditing images ({manifest_name})", unit="img")
    for _, row in pbar:
        base = {
            "manifest": row["manifest"],
            "row_number": int(row["row_number"]),
            "Path": row["Path"],
            "ClassId": row["ClassId"],
        }
        rel_path = str(row["Path"]).replace("\\", "/")
        img_path = (data_root / rel_path).resolve()

        # Check path containment
        try:
            img_path.relative_to(data_root)
        except ValueError:
            issues.append({**base, "issue_type": "unsafe_resolved_path", "details": "Path escapes dataset root."})
            continue

        # Check existence
        if not img_path.is_file():
            issues.append({**base, "issue_type": "missing_image_file", "details": str(img_path)})
            continue

        # Check image decode
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img.load()
                actual_w, actual_h = img.size
                img_mode = img.mode
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as err:
            issues.append({**base, "issue_type": "unreadable_or_corrupt_image", "details": f"{type(err).__name__}: {err}"})
            continue

        exp_w = int(row["_numeric_Width"]) if pd.notna(row["_numeric_Width"]) else None
        exp_h = int(row["_numeric_Height"]) if pd.notna(row["_numeric_Height"]) else None
        observations.append({**base, "actual_width": actual_w, "actual_height": actual_h, "image_mode": img_mode})

        if (actual_w, actual_h) != (exp_w, exp_h):
            issues.append({
                **base,
                "issue_type": "csv_image_dimension_mismatch",
                "details": f"CSV={exp_w}x{exp_h} vs File={actual_w}x{actual_h}"
            })

    return issues, pd.DataFrame(observations)


def main():
    parser = argparse.ArgumentParser(description="Audit GTSRB dataset manifests and images locally.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to GTSRB folder (e.g. d:/ML_Project/german)")
    parser.add_argument("--limit", type=int, default=None, help="Audit only the first N images per split (for quick check)")
    parser.add_argument("--report-dir", type=str, default="reports/dataset_audit", help="Directory to save audit output")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_dir)
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GTSRB DATASET AUDIT (Local Runner)")
    print(f"Data Root   : {data_root}")
    print(f"Report Dir  : {report_dir}")
    print(f"Image Limit : {args.limit if args.limit else 'All (full audit)'}")
    print("=" * 70)

    manifest_paths = {
        "train": data_root / "Train.csv",
        "test": data_root / "Test.csv",
    }

    all_issues: list[dict[str, Any]] = []
    all_observations: list[pd.DataFrame] = []
    summary: dict[str, Any] = {"data_root": str(data_root)}

    for name, path in manifest_paths.items():
        print(f"\n--- Processing Manifest: {name} ---")
        df_raw = load_manifest(name, path)
        print(f"Loaded {len(df_raw):,} rows from {path.name}")

        df_audited, meta_issues = metadata_audit(df_raw)
        print(f"Metadata check complete: {len(meta_issues)} issues found.")

        img_issues, obs = image_audit(data_root, df_audited, max_images=args.limit)
        print(f"Image check complete: {len(img_issues)} issues found.")

        all_issues.extend(meta_issues)
        all_issues.extend(img_issues)
        all_observations.append(obs)

        summary[name] = {
            "total_rows": len(df_raw),
            "images_audited": len(obs),
            "metadata_issues": len(meta_issues),
            "image_issues": len(img_issues),
        }

    issues_df = pd.DataFrame(all_issues)
    obs_df = pd.concat(all_observations, ignore_index=True) if all_observations else pd.DataFrame()

    summary["total_issues"] = len(issues_df)
    summary["issues_by_type"] = issues_df["issue_type"].value_counts().to_dict() if not issues_df.empty else {}
    summary["image_modes"] = obs_df["image_mode"].value_counts().to_dict() if not obs_df.empty else {}

    # Save reports
    issues_csv = report_dir / "dataset_issues.csv"
    obs_csv = report_dir / "image_observations.csv"
    summary_json = report_dir / "audit_summary.json"

    issues_df.to_csv(issues_csv, index=False)
    obs_df.to_csv(obs_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Issues Detected: {len(issues_df)}")

    if not issues_df.empty:
        print("\nBreakdown of Issues:")
        for itype, count in issues_df["issue_type"].value_counts().items():
            print(f"  - {itype}: {count:,}")
        print(f"\nTop 10 sample issues saved in: {issues_csv}")
    else:
        print("\nAll checks passed! No corrupted images, broken paths, or metadata mismatches.")

    if not obs_df.empty:
        print("\nImage Statistics:")
        print(f"  - Color Modes: {obs_df['image_mode'].value_counts().to_dict()}")
        print(f"  - Width Range: min={obs_df['actual_width'].min()}, max={obs_df['actual_width'].max()}, median={obs_df['actual_width'].median()}")
        print(f"  - Height Range: min={obs_df['actual_height'].min()}, max={obs_df['actual_height'].max()}, median={obs_df['actual_height'].median()}")

    print("\nReports Generated:")
    print(f"  1. Issues CSV       : {issues_csv}")
    print(f"  2. Observations CSV : {obs_csv}")
    print(f"  3. Summary JSON     : {summary_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
