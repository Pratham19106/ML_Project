"""GTSRB PyTorch Dataset Module (Master v1 Preprocessing Contract)

Implements:
1. ROI crop with 10% context margin (clamped to image bounds).
2. Aspect-ratio preserving proportional letterbox resize to 64x64.
3. Symmetric center zero-padding on a 64x64 canvas.
4. ImageNet normalization for pretrained ResNet-18.
5. Returns image tensor, class label, and concept annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def letterbox_roi_crop(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    target_size: int = 64,
    margin_ratio: float = 0.10,
) -> Image.Image:
    """Applies ROI crop with margin, proportional letterbox scaling, and symmetric zero padding."""
    w_img, h_img = img.size
    x1, y1, x2, y2 = roi
    roi_w = x2 - x1 + 1
    roi_h = y2 - y1 + 1

    # 10% context margin
    margin_x = int(round(margin_ratio * roi_w))
    margin_y = int(round(margin_ratio * roi_h))

    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(w_img - 1, x2 + margin_x)
    crop_y2 = min(h_img - 1, y2 + margin_y)

    cropped = img.crop((crop_x1, crop_y1, crop_x2 + 1, crop_y2 + 1))
    crop_w, crop_h = cropped.size

    # Proportional scaling
    scale = min(target_size / crop_w, target_size / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized = cropped.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    # Center zero-padding
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas


def resolve_image_path(data_root: Path, rel_path: str) -> Path:
    """Robustly resolves an image path handling case sensitivity (Train vs train) and directory nesting."""
    rel_p = str(rel_path).replace("\\", "/")
    direct = data_root / rel_p
    if direct.is_file():
        return direct

    parts = rel_p.split("/")
    if len(parts) > 1:
        top = parts[0]
        rest = "/".join(parts[1:])
        alt_top = top.lower() if top[0].isupper() else top.capitalize()
        case_alt = data_root / alt_top / rest
        if case_alt.is_file():
            return case_alt

    parent_direct = data_root.parent / rel_p
    if parent_direct.is_file():
        return parent_direct

    if len(parts) > 1:
        alt_top = top.lower() if top[0].isupper() else top.capitalize()
        parent_case_alt = data_root.parent / alt_top / rest
        if parent_case_alt.is_file():
            return parent_case_alt

    sub_direct = data_root / "german" / rel_p
    if sub_direct.is_file():
        return sub_direct

    return direct


class GTSRBDataset(Dataset):
    """PyTorch Dataset loading from frozen master_v1 manifests."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        target_size: int = 64,
        margin_ratio: float = 0.10,
        transform: Callable | None = None,
        is_training: bool = False,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.data_root = Path(data_root).resolve()
        self.target_size = target_size
        self.margin_ratio = margin_ratio
        self.transform = transform
        self.is_training = is_training

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        # Resolve image file with case/nesting fallback
        img_path = resolve_image_path(self.data_root, row["source_relative_path"])

        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")

        roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
        canvas = letterbox_roi_crop(img, roi, target_size=self.target_size, margin_ratio=self.margin_ratio)

        # Base transform: PIL Image to Tensor normalized with ImageNet statistics
        if self.transform is not None:
            tensor = self.transform(canvas)
        else:
            tensor = TF.to_tensor(canvas)
            tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        class_id = int(row["class_id"])

        item = {
            "image": tensor,
            "label": torch.tensor(class_id, dtype=torch.long),
            "image_id": row["image_id"],
            "class_name": row.get("class_name", ""),
            "shape": row.get("shape", ""),
            "colour_profile": row.get("colour_profile", ""),
            "border_type": row.get("border_type", ""),
            "symbol_category": row.get("symbol_category", ""),
            "symbol_content": row.get("symbol_content", ""),
        }
        return item

