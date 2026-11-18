"""Image loading and preprocessing for the quantum image pipeline.

Every image, whether from the built-in digits set, a ZIP of class folders, or a
single upload, goes through the same steps so what the dashboard shows is what
the model actually sees:

    original -> grayscale -> centre square crop -> (optional) invert so ink is
    bright -> resize to N x N by area averaging -> min-max normalise to [0, 1]
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
BASE_SIZE = 8


@dataclass
class PreprocessSteps:
    original: np.ndarray  # uint8 RGB, as uploaded
    gray: np.ndarray  # float 0..255
    cropped: np.ndarray  # square centre crop of gray
    oriented: np.ndarray  # after optional inversion (ink bright)
    resized: np.ndarray  # N x N, float 0..255
    normalized: np.ndarray  # N x N, float 0..1
    size: int
    inverted: bool


@dataclass
class ImageDataset:
    X: np.ndarray  # (N, BASE_SIZE, BASE_SIZE), each image min-max normalised to [0, 1]
    y: np.ndarray  # (N,) int labels 0..k-1
    class_names: list[str]
    source: str

    def at_size(self, size: int) -> np.ndarray:
        """All images resized to size x size, each re-normalised to [0, 1]."""
        return np.stack([normalize01(resize_array(img, size)) for img in self.X])

    def subsample_per_class(self, per_class: int, seed: int = 0) -> "ImageDataset":
        rng = np.random.RandomState(seed)
        keep = []
        for c in range(len(self.class_names)):
            idx = np.where(self.y == c)[0]
            if len(idx) > per_class:
                idx = rng.choice(idx, size=per_class, replace=False)
            keep.extend(idx.tolist())
        keep = np.array(sorted(keep))
        return ImageDataset(self.X[keep], self.y[keep], self.class_names, self.source)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def normalize01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def resize_array(a: np.ndarray, size: int) -> np.ndarray:
    """Resize a square float array to size x size by area averaging."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape == (size, size):
        return a
    if a.shape[0] % size == 0 and a.shape[1] % size == 0:
        fy, fx = a.shape[0] // size, a.shape[1] // size
        return a.reshape(size, fy, size, fx).mean(axis=(1, 3))
    resized = Image.fromarray(a, mode="F").resize((size, size), Image.BOX)
    return np.asarray(resized, dtype=np.float32)


def _flatten_alpha(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def preprocess_image(img: Image.Image, size: int, invert: bool) -> PreprocessSteps:
    """Run one image through the full preprocessing chain, keeping every
    intermediate result so the dashboard can show each step."""
    if not is_power_of_two(size):
        raise ValueError("Image size must be a power of two (2, 4, 8, 16).")
    rgb_img = _flatten_alpha(img)
    original = np.asarray(rgb_img, dtype=np.uint8)
    gray = np.asarray(rgb_img.convert("L"), dtype=float)

    h, w = gray.shape
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    cropped = gray[top:top + side, left:left + side]

    oriented = 255.0 - cropped if invert else cropped
    resized = resize_array(oriented, size).astype(float)
    normalized = normalize01(resized)
    return PreprocessSteps(original, gray, cropped, oriented, resized, normalized, size, invert)


def array_to_pil(normalized: np.ndarray, display_size: int = 256, ink_dark: bool = True) -> Image.Image:
    """Render a [0,1] ink-bright array as a paper-like image (dark ink on white)."""
    arr = (1.0 - normalized if ink_dark else normalized) * 255.0
    img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    return img.resize((display_size, display_size), Image.NEAREST)


def load_digits_subset(classes: list[int]) -> ImageDataset:
    """The scikit-learn 8x8 handwritten digits restricted to the chosen classes."""
    if len(classes) < 2:
        raise ValueError("Pick at least two digit classes.")
    digits = load_digits()
    classes = sorted(classes)
    mask = np.isin(digits.target, classes)
    remap = {c: i for i, c in enumerate(classes)}
    X = np.stack([normalize01(img) for img in digits.images[mask]])
    y = np.array([remap[t] for t in digits.target[mask]])
    return ImageDataset(X, y, [str(c) for c in classes], "sklearn handwritten digits (8x8)")


def load_zip_dataset(file_bytes: bytes, invert: bool, max_per_class: int = 200) -> ImageDataset:
    """A ZIP file containing one folder per class, each holding images."""
    images, labels = [], []
    per_class_count: dict[str, int] = {}
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in sorted(zf.namelist()):
            parts = name.split("/")
            if name.endswith("/") or "__MACOSX" in parts or parts[-1].startswith("."):
                continue
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            label = os.path.basename(os.path.dirname(name))
            if not label:
                continue
            if per_class_count.get(label, 0) >= max_per_class:
                continue
            try:
                img = Image.open(io.BytesIO(zf.read(name)))
                img.load()
            except Exception:
                continue
            steps = preprocess_image(img, BASE_SIZE, invert)
            images.append(steps.normalized)
            labels.append(label)
            per_class_count[label] = per_class_count.get(label, 0) + 1

    class_names = sorted(set(labels))
    if len(class_names) < 2:
        raise ValueError(
            "The ZIP needs at least two class folders containing images "
            "(e.g. cats/1.png, dogs/1.png). Found: " + (", ".join(class_names) or "none")
        )
    remap = {c: i for i, c in enumerate(class_names)}
    return ImageDataset(np.stack(images), np.array([remap[l] for l in labels]), class_names,
                        "uploaded ZIP dataset")
