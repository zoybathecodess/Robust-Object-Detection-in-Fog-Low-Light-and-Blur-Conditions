"""Datasets. Labeled data is described by a small JSON *manifest* (see prepare_data.py):

{"root": "/abs/path", "classes": ["person", ...],
 "images": [{"file": "rel/path.png", "boxes": [[x1,y1,x2,y2], ...], "labels": [1, ...]}]}

Labels are 1-based (0 is background).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from degradations import MidasDepth, degrade, fallback_depth, random_degrade  # noqa: F401

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def read_rgb(path, max_side=1024):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = 1.0
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img, scale


def depth_key(rel_file: str) -> str:
    return os.path.splitext(rel_file)[0].replace("/", "__").replace("\\", "__")


def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


class DetManifest(Dataset):
    """Labeled detection dataset with optional on-the-fly degradation.

    degrade: None | "random" | "fog" | "blur" | "lowlight"
    fixed_seed: if True the degradation of image i is deterministic (for reproducible evaluation).
    """

    def __init__(self, manifest, degrade=None, depth_dir=None, train=False,
                 max_side=1024, limit=None, fixed_seed=False, seed=0):
        m = load_manifest(manifest)
        self.root, self.classes = Path(m["root"]), m["classes"]
        self.items = m["images"][:limit] if limit else m["images"]
        self.degrade, self.depth_dir, self.train = degrade, depth_dir, train
        self.max_side, self.fixed_seed, self.seed = max_side, fixed_seed, seed

    def __len__(self):
        return len(self.items)

    def _depth_fn(self, rel):
        def fn(img):
            if self.depth_dir:
                p = Path(self.depth_dir) / (depth_key(rel) + ".npy")
                if p.exists():
                    d = np.load(p).astype(np.float32)
                    if d.shape != img.shape[:2]:
                        d = cv2.resize(d, (img.shape[1], img.shape[0]))
                    return d
            return fallback_depth(*img.shape[:2])
        return fn

    def __getitem__(self, i):
        it = self.items[i]
        img, scale = read_rgb(self.root / it["file"], self.max_side)
        boxes = np.asarray(it["boxes"], np.float32).reshape(-1, 4) * scale
        labels = np.asarray(it["labels"], np.int64)

        rng = np.random.default_rng(self.seed + i) if self.fixed_seed else np.random.default_rng()
        if self.degrade == "random":
            img, _ = random_degrade(img, self._depth_fn(it["file"]), rng)
        elif self.degrade:
            img = degrade(img, self.degrade, self._depth_fn(it["file"]), rng)

        if self.train and random.random() < 0.5:  # horizontal flip
            img = img[:, ::-1].copy()
            w = img.shape[1]
            boxes = boxes[:, [2, 1, 0, 3]].copy()
            boxes[:, [0, 2]] = w - boxes[:, [0, 2]]

        keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        target = {"boxes": torch.from_numpy(boxes[keep]), "labels": torch.from_numpy(labels[keep]),
                  "image_id": torch.tensor(i)}
        return to_tensor(img), target


class UnlabeledImages(Dataset):
    """Recursively collects images from one or more folders; labels are never read.
    include: optional list of substrings a path must contain (e.g. ["Day2fog"])."""

    def __init__(self, folders, include=None, max_side=1024, limit=None, train=True):
        files = []
        for folder in folders:
            for p in sorted(Path(folder).rglob("*")):
                if p.suffix.lower() in IMG_EXT and (not include or any(s in str(p) for s in include)):
                    files.append(p)
        if not files:
            raise RuntimeError(f"No images found in {folders} (include={include})")
        random.Random(0).shuffle(files)
        self.files = files[:limit] if limit else files
        self.max_side, self.train = max_side, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img, _ = read_rgb(self.files[i], self.max_side)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1].copy()
        return to_tensor(img)


def det_collate(batch):
    return tuple(zip(*batch))


def img_collate(batch):
    return list(batch)


def strong_augment(img: torch.Tensor) -> torch.Tensor:
    """Photometric/noise/cutout augmentation for the student (geometry untouched so the
    teacher's pseudo-boxes stay valid)."""
    import torchvision.transforms.functional as TF

    x = img.clone()
    m = x.mean()
    x = ((x - m) * random.uniform(0.7, 1.3) + m) * random.uniform(0.7, 1.3)
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * random.uniform(0.01, 0.05)
    if random.random() < 0.3:
        x = TF.gaussian_blur(x, [5, 5], [random.uniform(0.5, 1.5)] * 2)
    _, h, w = x.shape
    for _ in range(random.randint(0, 3)):
        eh, ew = int(h * random.uniform(0.05, 0.15)), int(w * random.uniform(0.05, 0.15))
        y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
        x[:, y0:y0 + eh, x0:x0 + ew] = 0.5
    return x.clamp(0, 1)
