"""
app.py - upload an image in your browser, get the same image back with detected objects drawn on it.

Put this file in the project folder (next to main.py), then run:

    pip install gradio
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python app.py                          # picks the newest checkpoint in ckpt/
    python app.py --ckpt ckpt/r50_aug.pt   # or choose one
    python app.py --ckpt ckpt/full_aug.pt #to open app on browser

A browser tab opens at http://127.0.0.1:7860 . Everything runs locally on your Mac.
"""
import argparse
import glob
import os

import cv2
import gradio as gr
import numpy as np
import torch

from engine import get_device
from model import build_model

device = get_device()
_cache = {}
COLORS = [(230, 60, 60), (60, 140, 230), (60, 200, 90), (240, 180, 40),
          (170, 80, 220), (40, 200, 200), (240, 120, 30), (200, 200, 200)]


def load(path):
    if path not in _cache:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        classes = [str(c) for c in ck["classes"]]
        model = build_model(len(classes), ck["backbone"], pretrained=False,
                            min_size=ck["min_size"], max_size=ck["max_size"])
        model.load_state_dict(ck["model"])
        _cache[path] = (model.to(device).eval(), classes)
    return _cache[path]


def detect(image, ckpt, thr):
    if image is None:
        return None, "Upload an image first."
    if not ckpt:
        return None, "No checkpoint found in ckpt/. Train a model first."
    model, classes = load(ckpt)
    img = np.ascontiguousarray(image[..., :3]).astype(np.uint8)          # drop alpha if present
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        out = model([x])[0]

    keep = out["scores"] >= thr
    boxes = out["boxes"][keep].cpu().numpy()
    labels = out["labels"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()

    vis = img.copy()
    scale = max(1, round(max(img.shape[:2]) / 800))
    counts = {}
    for b, l, s in zip(boxes, labels, scores):
        li = int(l) - 1
        name = classes[li] if 0 <= li < len(classes) else str(int(l))
        counts[name] = counts.get(name, 0) + 1
        color = COLORS[li % len(COLORS)] if li >= 0 else COLORS[0]
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2 * scale)
        cv2.putText(vis, f"{name} {s:.2f}", (x1, max(14 * scale, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, color, max(1, scale), cv2.LINE_AA)

    if counts:
        summary = "Found: " + ", ".join(f"{n} x{c}" for n, c in sorted(counts.items()))
    else:
        summary = f"No objects above confidence {thr:.2f}. Try lowering the slider."
    return vis, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()

    ckpts = sorted(glob.glob("ckpt/*.pt"), key=os.path.getmtime, reverse=True)
    default = a.ckpt or (ckpts[0] if ckpts else None)
    if a.ckpt and a.ckpt not in ckpts:
        ckpts.insert(0, a.ckpt)

    demo = gr.Interface(
        fn=detect,
        inputs=[gr.Image(type="numpy", label="Upload an image"),
                gr.Dropdown(choices=ckpts, value=default, label="Model checkpoint"),
                gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Confidence threshold")],
        outputs=[gr.Image(label="Detections"), gr.Textbox(label="Summary")],
        title="Robust object detection in fog, blur and low light",
        description="Detects: person, rider, car, truck, bus, train, motorcycle, bicycle. Works best on street scenes.",
        flagging_mode="never",
    )
    demo.launch(server_port=a.port, inbrowser=True)


if __name__ == "__main__":
    main()
