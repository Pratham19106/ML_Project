"""Frequency-Aware Augmentation Suite for GTSRB (Fourier Domain).

Methodological Grounding:
Wang, H., Wu, X., Yin, Z., & Xing, E. P. (2020).
High Frequency Component Helps Explain the Generalization of CNNs.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020).

Addresses the AugMix Noise Deficit by attenuating or stochastically dropping
high-frequency shortcuts in 2D Fourier space without spatial noise leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms.functional as TF
    from src.dataset import GTSRBDataset, IMAGENET_MEAN, IMAGENET_STD, letterbox_roi_crop, resolve_image_path
except ImportError:
    torch = None
    TF = None
    GTSRBDataset = object
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    letterbox_roi_crop = None


def low_frequency_component(img: Image.Image, radius: int = 14) -> Image.Image:
    """Extracts Low-Frequency Component (LFC) by filtering out frequencies beyond radius."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (radius ** 2)

    if arr.ndim == 2:
        f = np.fft.fftshift(np.fft.fft2(arr))
        filtered = f * mask
        spatial = np.fft.ifft2(np.fft.ifftshift(filtered))
        res = np.clip(np.real(spatial), 0, 255).astype(np.uint8)
        return Image.fromarray(res, mode="L")
    else:
        channels = []
        for c in range(arr.shape[2]):
            f = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
            filtered = f * mask
            spatial = np.fft.ifft2(np.fft.ifftshift(filtered))
            res = np.clip(np.real(spatial), 0, 255).astype(np.uint8)
            channels.append(res)
        res_arr = np.stack(channels, axis=-1)
        return Image.fromarray(res_arr, mode="RGB")


def high_frequency_component(img: Image.Image, radius: int = 14) -> Image.Image:
    """Extracts High-Frequency Component (HFC) by filtering out frequencies within radius."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx) ** 2 + (y - cy) ** 2) > (radius ** 2)

    if arr.ndim == 2:
        f = np.fft.fftshift(np.fft.fft2(arr))
        filtered = f * mask
        spatial = np.fft.ifft2(np.fft.ifftshift(filtered))
        res = np.clip(np.abs(np.real(spatial)), 0, 255).astype(np.uint8)
        return Image.fromarray(res, mode="L")
    else:
        channels = []
        for c in range(arr.shape[2]):
            f = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
            filtered = f * mask
            spatial = np.fft.ifft2(np.fft.ifftshift(filtered))
            res = np.clip(np.abs(np.real(spatial)), 0, 255).astype(np.uint8)
            channels.append(res)
        res_arr = np.stack(channels, axis=-1)
        return Image.fromarray(res_arr, mode="RGB")


def frequency_dropout(img: Image.Image, severity: float = 3.0, r_min: int = 10) -> Image.Image:
    """Stochastically drops high-frequency coefficients in Fourier space scaled by severity."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    hfc_mask = ((x - cx) ** 2 + (y - cy) ** 2) >= (r_min ** 2)

    p_drop = min(0.85, 0.08 * float(severity))
    drop_mask = (np.random.rand(h, w) > p_drop).astype(np.float32)
    filter_mask = np.where(hfc_mask, drop_mask, 1.0).astype(np.float32)

    if arr.ndim == 2:
        f = np.fft.fftshift(np.fft.fft2(arr))
        filtered = f * filter_mask
        spatial = np.fft.ifft2(np.fft.ifftshift(filtered))
        res = np.clip(np.real(spatial), 0, 255).astype(np.uint8)
        return Image.fromarray(res, mode="L")
    else:
        channels = []
        for c in range(arr.shape[2]):
            f = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
            filtered = f * filter_mask
            spatial = np.fft.ifft2(np.fft.ifftshift(filtered))
            res = np.clip(np.real(spatial), 0, 255).astype(np.uint8)
            channels.append(res)
        res_arr = np.stack(channels, axis=-1)
        return Image.fromarray(res_arr, mode="RGB")


def frequency_blend(img: Image.Image, radius: int = 14, alpha: float = 0.5) -> Image.Image:
    """Convex blend between the original image and its low-frequency component."""
    lfc = low_frequency_component(img, radius=radius)
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    return Image.blend(lfc, img, alpha_clamped)


class FrequencyAugDataset(GTSRBDataset):
    """PyTorch Dataset applying Fourier-domain spectral augmentations."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        target_size: int = 64,
        margin_ratio: float = 0.10,
        severity: float = 3.0,
        mode: str = "dropout",
        radius: int = 14,
        is_training: bool = True,
    ) -> None:
        super().__init__(
            manifest_path=manifest_path,
            data_root=data_root,
            target_size=target_size,
            margin_ratio=margin_ratio,
            is_training=is_training,
        )
        self.severity = severity
        self.mode = mode
        self.radius = radius

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = resolve_image_path(self.data_root, row["source_relative_path"])

        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")

        roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
        canvas = letterbox_roi_crop(img, roi, target_size=self.target_size, margin_ratio=self.margin_ratio)

        if self.is_training:
            if self.mode == "lfc":
                canvas = low_frequency_component(canvas, radius=self.radius)
            elif self.mode == "blend":
                canvas = frequency_blend(canvas, radius=self.radius, alpha=0.5)
            else:
                canvas = frequency_dropout(canvas, severity=self.severity, r_min=self.radius)

        tensor = TF.to_tensor(canvas)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        class_id = int(row["class_id"])
        return {
            "image": tensor,
            "label": torch.tensor(class_id, dtype=torch.long),
            "image_id": row["image_id"],
            "class_name": row.get("class_name", ""),
        }

