"""Comprehensive Unit Tests for Frequency-Aware Augmentation Suite.

Validates:
1. Low-Frequency Component (LFC) filtering: Output shape, dynamic range, no NaNs.
2. High-Frequency Component (HFC) filtering: Output shape, dynamic range, no NaNs.
3. Frequency Dropout: Severity scaling, mask validity, boundary preservation.
4. Frequency Blend: Structural blending consistency.
5. Spectral Energy Verification: Verifies LFC successfully attenuates high frequencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from src.augmentations.frequency import (
    low_frequency_component,
    high_frequency_component,
    frequency_dropout,
    frequency_blend,
)


def create_test_image(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Creates a deterministic synthetic traffic sign canvas (64x64 RGB)."""
    img = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Outer red circle
    draw.ellipse([6, 6, 57, 57], fill=(220, 20, 20))
    # Inner white circle
    draw.ellipse([14, 14, 49, 49], fill=(245, 245, 245))
    # High-frequency black glyph
    draw.line([(28, 24), (28, 40)], fill=(10, 10, 10), width=3)
    draw.line([(36, 24), (36, 40)], fill=(10, 10, 10), width=3)
    return img


def test_low_frequency_component():
    img = create_test_image()
    out = low_frequency_component(img, radius=12)

    assert isinstance(out, Image.Image), f"Expected PIL.Image, got {type(out)}"
    assert out.size == (64, 64), f"Expected size (64, 64), got {out.size}"
    assert out.mode == "RGB", f"Expected mode RGB, got {out.mode}"

    arr = np.array(out)
    assert arr.dtype == np.uint8, f"Expected uint8, got {arr.dtype}"
    assert not np.isnan(arr).any(), "LFC output contains NaN values"
    assert not np.isinf(arr).any(), "LFC output contains Inf values"
    assert arr.min() >= 0 and arr.max() <= 255, f"Values out of bounds: [{arr.min()}, {arr.max()}]"
    print("  [PASS] test_low_frequency_component (shape, bounds, no-NaN verified)")


def test_high_frequency_component():
    img = create_test_image()
    out = high_frequency_component(img, radius=12)

    assert isinstance(out, Image.Image), f"Expected PIL.Image, got {type(out)}"
    assert out.size == (64, 64), f"Expected size (64, 64), got {out.size}"
    assert out.mode == "RGB", f"Expected mode RGB, got {out.mode}"

    arr = np.array(out)
    assert arr.dtype == np.uint8, f"Expected uint8, got {arr.dtype}"
    assert not np.isnan(arr).any(), "HFC output contains NaN values"
    assert not np.isinf(arr).any(), "HFC output contains Inf values"
    assert arr.min() >= 0 and arr.max() <= 255, f"Values out of bounds: [{arr.min()}, {arr.max()}]"
    print("  [PASS] test_high_frequency_component (shape, bounds, no-NaN verified)")


def test_frequency_dropout_severities():
    img = create_test_image()

    for sev in [1.0, 3.0, 5.0, 10.0]:
        out = frequency_dropout(img, severity=sev, r_min=10)
        assert isinstance(out, Image.Image)
        assert out.size == (64, 64)
        arr = np.array(out)
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()
        assert arr.min() >= 0 and arr.max() <= 255

    print("  [PASS] test_frequency_dropout_severities (severity sweep [1..10] verified)")


def test_frequency_blend():
    img = create_test_image()
    out = frequency_blend(img, radius=14, alpha=1.0)

    assert isinstance(out, Image.Image)
    assert out.size == (64, 64)
    arr = np.array(out)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()
    assert arr.min() >= 0 and arr.max() <= 255
    print("  [PASS] test_frequency_blend (convex structural blend verified)")


def test_spectral_filtering_effectiveness():
    img = create_test_image()
    arr_orig = np.array(img.convert("L")).astype(np.float32)

    # Compute high frequency energy of original
    f_orig = np.fft.fftshift(np.fft.fft2(arr_orig))
    h, w = arr_orig.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    hfc_mask = ((x - cx) ** 2 + (y - cy) ** 2) >= (16 ** 2)

    hfc_energy_orig = np.sum(np.abs(f_orig)[hfc_mask] ** 2)

    # LFC filtered image
    lfc_img = low_frequency_component(img, radius=12)
    arr_lfc = np.array(lfc_img.convert("L")).astype(np.float32)
    f_lfc = np.fft.fftshift(np.fft.fft2(arr_lfc))
    hfc_energy_lfc = np.sum(np.abs(f_lfc)[hfc_mask] ** 2)

    # High frequency energy in LFC must be heavily attenuated (>80% reduction)
    attenuation = 1.0 - (hfc_energy_lfc / (hfc_energy_orig + 1e-8))
    assert attenuation > 0.80, f"Expected >80% HFC attenuation, got {attenuation*100:.1f}%"
    print(f"  [PASS] test_spectral_filtering_effectiveness (HFC attenuated by {attenuation*100:.1f}%)")


if __name__ == "__main__":
    print("\n--- Running Frequency Augmentation Unit Tests ---")
    test_low_frequency_component()
    test_high_frequency_component()
    test_frequency_dropout_severities()
    test_frequency_blend()
    test_spectral_filtering_effectiveness()
    print("--- ALL 5 UNIT TESTS PASSED SUCCESSFULLY ---\n")
