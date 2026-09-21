# Master Dataset Specification and Version Record (master_v1)

- **Creation Timestamp (UTC):** 2026-09-20T13:16:19.482840+00:00
- **Raw Data Source:** `D:\ML_Project\german`
- **Master Random Seed:** `42`
- **Split Protocol:** Stratified Group Split by `(ClassId + TrackId)`
- **Preprocessing Version:** `v1_roi_margin10_letterbox64`
  - ROI Crop: 10% contextual margin on all sides (clamped to image dimensions).
  - Resizing: Proportional bilinear scaling preserving aspect ratio.
  - Padding: Symmetric zero-padding to 64x64 canvas.

## Dataset Partition Summary

| Split | Number of Images | Description |
| :--- | :--- | :--- |
| **Train** | `31,378` | 80% of tracks per class for training |
| **Validation** | `7,830` | 20% of tracks per class (zero video-frame leakage) |
| **Test** | `12,630` | Official GTSRB held-out test set (strictly untouched) |
| **Total** | `51,838` | Total annotated samples |

## Manifest SHA-256 Checksums

```text
train.csv:           8b3ccc81296014751cc1ebbde1e9085076472f01e6704e420c32f25a31ac3aa4
val.csv:             5658566473412da8e4b3c2d251ee563ba894366f757d7830cf5abd03dcb2af0d
test.csv:            0a4029151060c5e748303aa6d88c7c7b53ec7e8fa0f77d5f66707837626ddc3b
master_metadata.csv: b6daad60ae5c7be0cda9fcfb26ad025592451af40b7c3ab6901c940a62673b40
```
