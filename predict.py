"""
predict.py - run a trained detector on any image.

Put this file in the project folder (next to main.py), then:

    python predict.py my_photo.jpg --ckpt ckpt/self.pt
    python predict.py my_photo.jpg --ckpt ckpt/baseline.pt --thr 0.3 --out baseline_result.jpg

Prints every detection and saves the image with boxes drawn.
"""
import argparse
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from engine import get_device
from model import build_model


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = [str(c) for c in ck["classes"]]
    model = build_model(len(classes), ck["backbone"], pretrained=False,
                        min_size=ck["min_size"], max_size=ck["max_size"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default="ckpt/self.pt")
    ap.add_argument("--thr", type=float, default=0.5, help="minimum confidence to show")
    ap.add_argument("--out", default="prediction.jpg")
    a = ap.parse_args()

    device = get_device()
    model, classes = load(a.ckpt, device)

    img = np.asarray(Image.open(a.image).convert("RGB"))
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        out = model([x])[0]

    keep = out["scores"] >= a.thr
    boxes = out["boxes"][keep].cpu().numpy()
    labels = out["labels"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()

    vis = img.copy()
    scale = max(1, round(max(img.shape[:2]) / 600))
    for b, l, s in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
        name = classes[int(l) - 1] if 0 < int(l) <= len(classes) else str(int(l))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 60, 60), 2 * scale)
        cv2.putText(vis, f"{name} {s:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, (255, 60, 60), max(1, scale), cv2.LINE_AA)
        print(f"{name:12s} {s:.2f}  box=({x1},{y1},{x2},{y2})")
    if len(boxes) == 0:
        print(f"No detections above confidence {a.thr}. Try --thr 0.2")

    cv2.imwrite(a.out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved {a.out}")


if __name__ == "__main__":
    sys.exit(main())
