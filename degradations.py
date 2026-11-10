"""Physics-based synthetic degradations: depth-aware fog, directional motion blur,
sensor-realistic low-light noise. All functions take/return HxWx3 uint8 RGB arrays.

Depth convention: depth maps are float32 in [0, 1], 1 = farthest.
"""
from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------- depth
def fallback_depth(h: int, w: int) -> np.ndarray:
    """Ground-plane prior used when no depth map exists: far at the top, near at the bottom."""
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    d = 0.1 + 0.9 * (1.0 - y)
    return np.repeat(d, w, axis=1)


class MidasDepth:
    """Monocular relative depth with MiDaS (downloaded via torch.hub on first use)."""

    def __init__(self, device: str = "cpu", variant: str = "MiDaS_small"):
        import torch

        self.torch, self.device = torch, device
        self.model = torch.hub.load("intel-isl/MiDaS", variant, trust_repo=True).to(device).eval()
        tfs = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = tfs.small_transform if variant == "MiDaS_small" else tfs.dpt_transform

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = self.transform(rgb).to(self.device)
            p = self.model(x)
            p = torch.nn.functional.interpolate(
                p.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
        inv = (p - p.min()) / (p.max() - p.min() + 1e-6)  # inverse depth, 1 = near
        depth = 1.0 / (0.1 + inv)                          # depth ~ 1 / disparity
        return (depth / depth.max()).astype(np.float32)


# --------------------------------------------------------------------------- fog
def add_fog(img, depth01, beta=0.01, max_depth=250.0, airlight=0.9, rng=None):
    """Koschmieder model: I = J * t + A * (1 - t),  t = exp(-beta * d).

    beta: attenuation coefficient (1/m). Foggy Cityscapes uses 0.005 / 0.01 / 0.02.
    max_depth: metres corresponding to depth01 == 1.
    """
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    J = img.astype(np.float32) / 255.0
    d = depth01.astype(np.float32) * max_depth
    # spatially varying density (patchy fog)
    low = rng.uniform(0.75, 1.25, size=(4, 6)).astype(np.float32)
    density = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    t = np.exp(-beta * density * d)[..., None]
    out = J * t + airlight * (1.0 - t)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- blur
def motion_blur(img, length=15, angle=0.0):
    """Directional linear motion blur."""
    length = int(length) | 1  # odd
    k = np.zeros((length, length), np.float32)
    c = length // 2
    cv2.line(k, (0, c), (length - 1, c), 1.0, 1)
    rot = cv2.getRotationMatrix2D((c, c), angle, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum() + 1e-8
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------- low light
def low_light(img, gain=0.2, gamma=2.0, photons=120.0, read_sigma=0.015, rng=None):
    """Darken (gain, gamma) then add Poisson shot noise + Gaussian read noise
    (heteroscedastic sensor noise, which is what makes real low-light images so grainy)."""
    rng = rng or np.random.default_rng()
    I = img.astype(np.float32) / 255.0
    dark = gain * np.power(I, gamma)
    shot = rng.poisson(np.clip(dark, 0, 1) * photons).astype(np.float32) / photons
    read = rng.normal(0.0, read_sigma, size=I.shape).astype(np.float32)
    out = np.clip(shot + read, 0, 1)
    return (out * 255).astype(np.uint8)


# --------------------------------------------------------------------------- sampler
KINDS = ("fog", "blur", "lowlight")


def degrade(img, kind, depth_fn=None, rng=None):
    """Apply one named degradation with randomly sampled severity."""
    rng = rng or np.random.default_rng()
    if kind == "fog":
        depth = depth_fn(img) if depth_fn is not None else fallback_depth(*img.shape[:2])
        return add_fog(img, depth, beta=rng.uniform(0.005, 0.03),
                       airlight=rng.uniform(0.8, 0.95), rng=rng)
    if kind == "blur":
        return motion_blur(img, length=int(rng.integers(7, 31)), angle=rng.uniform(0, 180))
    if kind == "lowlight":
        return low_light(img, gain=rng.uniform(0.08, 0.35), gamma=rng.uniform(1.5, 2.5),
                         photons=rng.uniform(60, 300), read_sigma=rng.uniform(0.005, 0.03), rng=rng)
    raise ValueError(kind)


def random_degrade(img, depth_fn=None, rng=None, p_clean=0.25, p_combo=0.2):
    """Training-time augmentation: clean / single degradation / combination of two."""
    rng = rng or np.random.default_rng()
    if rng.random() < p_clean:
        return img, "clean"
    kinds = [str(rng.choice(KINDS))]
    if rng.random() < p_combo:
        kinds.append(str(rng.choice([k for k in KINDS if k != kinds[0]])))
    # fog first (needs the clean image for depth), noise/darkening last
    kinds.sort(key=lambda k: {"fog": 0, "blur": 1, "lowlight": 2}[k])
    for k in kinds:
        img = degrade(img, k, depth_fn, rng)
    return img, "+".join(kinds)
