"""Traffic Sign Corruption Benchmark Suite (Resolution-Aware GTSRB-C)

Implements the 15 standard Hendrycks & Dietterich corruptions adapted for 64x64 inputs:
1. Noise: Gaussian Noise, Shot Noise, Impulse Noise, Speckle Noise
2. Blur: Defocus Blur, Glass Blur, Motion Blur, Zoom Blur, Gaussian Blur
3. Weather: Fog, Brightness, Contrast
4. Digital: Pixelate, JPEG Compression, Elastic Transform

Pure OpenCV / SciPy / NumPy implementation with zero fragile external C-library (Wand/ImageMagick) dependencies.
"""

from __future__ import annotations

from io import BytesIO
from typing import Callable

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates


# --- Helpers ---

def disk_kernel(radius: int) -> np.ndarray:
    """Generates an antialiased circular disk kernel for defocus blur."""
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((diameter, diameter), dtype=np.float32)
    kernel[mask] = 1.0
    return kernel / np.sum(kernel)


def motion_blur_kernel(kernel_size: int, angle: float) -> np.ndarray:
    """Generates a linear motion blur convolution kernel."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    cv2.line(kernel, (0, center), (kernel_size - 1, center), 1.0, thickness=1)
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    s = np.sum(kernel)
    return kernel / (s if s > 0 else 1.0)


def clipped_zoom(img: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Zooms in and center crops back to original dimensions."""
    h, w = img.shape[:2]
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))

    top = max(0, (h - ch) // 2)
    left = max(0, (w - cw) // 2)
    cropped = img[top : top + ch, left : left + cw]
    zoomed = scizoom(cropped, (zoom_factor, zoom_factor, 1), order=1)

    trim_top = (zoomed.shape[0] - h) // 2
    trim_left = (zoomed.shape[1] - w) // 2
    return zoomed[trim_top : trim_top + h, trim_left : trim_left + w]


# --- 15 Corruptions (Adapted for 64x64) ---

def gaussian_noise(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    c = [0.06, 0.10, 0.15, 0.22, 0.30][severity - 1]
    arr = np.array(x, dtype=np.float32) / 255.0
    noisy = arr + np.random.normal(scale=c, size=arr.shape)
    return PILImage.fromarray(np.uint8(np.clip(noisy, 0.0, 1.0) * 255))


def shot_noise(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    c = [50, 25, 12, 6, 3][severity - 1]
    arr = np.array(x, dtype=np.float32) / 255.0
    noisy = np.random.poisson(arr * c) / float(c)
    return PILImage.fromarray(np.uint8(np.clip(noisy, 0.0, 1.0) * 255))


def impulse_noise(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    c = [0.03, 0.06, 0.09, 0.15, 0.22][severity - 1]
    arr = np.array(x, dtype=np.float32) / 255.0
    prob = np.random.uniform(0.0, 1.0, size=arr.shape[:2])
    arr[prob < (c / 2)] = 0.0
    arr[(prob >= (c / 2)) & (prob < c)] = 1.0
    return PILImage.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255))


def speckle_noise(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    c = [0.10, 0.16, 0.24, 0.34, 0.45][severity - 1]
    arr = np.array(x, dtype=np.float32) / 255.0
    noisy = arr + arr * np.random.normal(scale=c, size=arr.shape)
    return PILImage.fromarray(np.uint8(np.clip(noisy, 0.0, 1.0) * 255))


def defocus_blur(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    radius = [1, 2, 3, 4, 5][severity - 1]
    kernel = disk_kernel(radius)
    arr = np.array(x, dtype=np.float32)
    blurred = cv2.filter2D(arr, -1, kernel)
    return PILImage.fromarray(np.uint8(np.clip(blurred, 0, 255)))


def motion_blur(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    ksize = [3, 5, 7, 9, 13][severity - 1]
    angle = np.random.uniform(-45, 45)
    kernel = motion_blur_kernel(ksize, angle)
    arr = np.array(x, dtype=np.float32)
    blurred = cv2.filter2D(arr, -1, kernel)
    return PILImage.fromarray(np.uint8(np.clip(blurred, 0, 255)))


def zoom_blur(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    c = [
        np.arange(1.0, 1.06, 0.01),
        np.arange(1.0, 1.11, 0.01),
        np.arange(1.0, 1.16, 0.02),
        np.arange(1.0, 1.21, 0.02),
        np.arange(1.0, 1.26, 0.03),
    ][severity - 1]
    arr = (np.array(x, dtype=np.float32) / 255.0)
    out = np.zeros_like(arr)
    for zoom_factor in c:
        out += clipped_zoom(arr, zoom_factor)
    combined = (arr + out) / (len(c) + 1)
    return PILImage.fromarray(np.uint8(np.clip(combined, 0.0, 1.0) * 255))


def gaussian_blur(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    sigma = [0.6, 1.0, 1.5, 2.0, 2.8][severity - 1]
    ksize = int(2 * np.ceil(2 * sigma) + 1)
    arr = cv2.GaussianBlur(np.array(x), (ksize, ksize), sigma)
    return PILImage.fromarray(arr)


def glass_blur(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    # sigma, max_delta, iterations
    c = [(0.5, 1, 1), (0.7, 1, 2), (0.9, 2, 1), (1.1, 2, 2), (1.3, 3, 2)][severity - 1]
    sigma, delta, iters = c
    arr = cv2.GaussianBlur(np.array(x), (0, 0), sigma)
    h, w = arr.shape[:2]
    for _ in range(iters):
        for y in range(delta, h - delta):
            for x_coord in range(delta, w - delta):
                dy, dx = np.random.randint(-delta, delta + 1, size=(2,))
                arr[y, x_coord], arr[y + dy, x_coord + dx] = arr[y + dy, x_coord + dx], arr[y, x_coord]
    return PILImage.fromarray(cv2.GaussianBlur(arr, (0, 0), sigma))


def brightness(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    arr = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2HSV).astype(np.float32)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + c * 255.0, 0, 255)
    return PILImage.fromarray(cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_HSV2RGB))


def contrast(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    c = [0.8, 0.6, 0.4, 0.25, 0.15][severity - 1]
    arr = np.array(x, dtype=np.float32) / 255.0
    mean = np.mean(arr, axis=(0, 1), keepdims=True)
    adjusted = (arr - mean) * c + mean
    return PILImage.fromarray(np.uint8(np.clip(adjusted, 0.0, 1.0) * 255))


def fog(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    c = [0.2, 0.35, 0.5, 0.65, 0.8][severity - 1]
    arr = np.array(x, dtype=np.float32) / 255.0
    # Homogeneous atmospheric scattering for small sign patches
    fogged = arr * (1.0 - c) + c * 0.95
    return PILImage.fromarray(np.uint8(np.clip(fogged, 0.0, 1.0) * 255))


def pixelate(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    factor = [0.65, 0.50, 0.38, 0.28, 0.18][severity - 1]
    w, h = x.size
    small_w = max(4, int(w * factor))
    small_h = max(4, int(h * factor))
    downsampled = x.resize((small_w, small_h), PILImage.Resampling.BOX)
    return downsampled.resize((w, h), PILImage.Resampling.BOX)


def jpeg_compression(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    quality = [40, 25, 18, 12, 7][severity - 1]
    buffer = BytesIO()
    x.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return PILImage.open(buffer).copy()


def elastic_transform(x: PILImage.Image, severity: int = 1) -> PILImage.Image:
    c = [(30, 2.5), (45, 2.0), (60, 1.8), (75, 1.5), (90, 1.2)][severity - 1]
    alpha, sigma = c
    arr = np.array(x, dtype=np.float32) / 255.0
    shape = arr.shape
    dx = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), sigma) * alpha

    y, x_grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    indices = (y + dy).reshape(-1, 1), (x_grid + dx).reshape(-1, 1)

    warped = np.zeros_like(arr)
    for ch in range(3):
        warped[..., ch] = map_coordinates(arr[..., ch], indices, order=1, mode="reflect").reshape(shape[:2])
    return PILImage.fromarray(np.uint8(np.clip(warped, 0.0, 1.0) * 255))


CORRUPTIONS_REGISTRY: dict[str, Callable[[PILImage.Image, int], PILImage.Image]] = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "speckle_noise": speckle_noise,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "gaussian_blur": gaussian_blur,
    "glass_blur": glass_blur,
    "brightness": brightness,
    "contrast": contrast,
    "fog": fog,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
    "elastic_transform": elastic_transform,
}

