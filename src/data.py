"""Data utilities for FREUID: paths, aspect-preserving letterbox, dataset."""
from __future__ import annotations
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Data root: hardcoded Kaggle competition path if present, else local download.
# (If Kaggle's folder layout differs, adjust _KAGGLE_DATA below.)
_KAGGLE_DATA = "/kaggle/input/the-freuid-challenge-2026-ijcai-ecai"
DATA = _KAGGLE_DATA if os.path.isdir(_KAGGLE_DATA) else os.path.join(ROOT, "the-freuid-challenge-dataset")
TRAIN_DIR = os.path.join(DATA, "train", "train")
TEST_DIR = os.path.join(DATA, "public_test", "public_test")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def train_path(img_id: str) -> str:
    return os.path.join(TRAIN_DIR, f"{img_id}.jpeg")


def test_path(img_id: str) -> str:
    return os.path.join(TEST_DIR, f"{img_id}.jpeg")


def letterbox(img: np.ndarray, out_h: int, out_w: int, pad_value: int = 0) -> np.ndarray:
    """Resize preserving aspect ratio, pad to (out_h, out_w). img is HxWx3 RGB uint8.

    Cards are ~1.585:1; choosing out_w/out_h near that keeps padding minimal.
    """
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((out_h, out_w, 3), pad_value, np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor_norm(img: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> 3xHxW float, ImageNet-normalized."""
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).contiguous()


class FreuidDataset(Dataset):
    """Returns (image_tensor, label, index). transform: optional albumentations.

    `paths` and `labels` are aligned lists. `labels` may be None (test).
    """
    def __init__(self, paths, labels, out_h, out_w, transform=None):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, np.float32)
        self.out_h, self.out_w = out_h, out_w
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = letterbox(img, self.out_h, self.out_w)
        x = to_tensor_norm(img)
        y = -1.0 if self.labels is None else float(self.labels[i])
        return x, y, i
