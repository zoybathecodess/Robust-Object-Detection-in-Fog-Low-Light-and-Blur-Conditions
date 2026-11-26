"""
per_class_eval.py - per-class accuracy plus a direct measurement of what the model calls each true truck.

    python per_class_eval.py --ckpt ckpt/r50_aug.pt
    python per_class_eval.py --ckpt ckpt/r50_aug.pt --limit 100      # quick check
"""
import argparse
import collections
import json
import os

import numpy as np
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm

from engine import get_device
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--manifest", default="manifests/cityscapes_val.json")
ap.add_argument("--limit", type=int, default=500)
ap.add_argument("--thr", type=float, default=0.5, help="confidence needed to count a detection")
a = ap.parse_args()

device = get_device()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
classes = [str(c) for c in ck["classes"]]
model = build_model(len(classes), ck["backbone"], pretrained=False, min_size=ck["min_size"], max_size=ck["max_size"])
model.load_state_dict(ck["model"])
model = model.to(device).eval()

d = json.load(open(a.manifest))
truck_id = classes.index("truck") + 1
m_strict = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
m_50 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
called = collections.Counter()
by_size = collections.defaultdict(collections.Counter)


def size_bucket(h):
    return "small  (<60 px tall)" if h < 60 else ("medium (60-120 px)" if h < 120 else "large  (>120 px)")


for rec in tqdm(d["images"][:a.limit]):
    img = np.asarray(Image.open(os.path.join(d["root"], rec["file"])).convert("RGB"))
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        out = model([x])[0]
    pred = {k: v.cpu() for k, v in out.items()}
    gt = {"boxes": torch.tensor(rec["boxes"], dtype=torch.float32).reshape(-1, 4),
          "labels": torch.tensor(rec["labels"], dtype=torch.int64)}
    m_strict.update([pred], [gt])
    m_50.update([pred], [gt])

    keep = pred["scores"] >= a.thr
    pb, pl, ps = pred["boxes"][keep], pred["labels"][keep], pred["scores"][keep]
    for gi in torch.nonzero(gt["labels"] == truck_id).flatten():
        bucket = size_bucket(float(gt["boxes"][gi][3] - gt["boxes"][gi][1]))
        ok = box_iou(gt["boxes"][gi:gi + 1], pb)[0] >= 0.5 if len(pb) else torch.zeros(0, dtype=torch.bool)
        if not ok.any():
            key = "missed (nothing detected)"
        else:
            j = torch.nonzero(ok).flatten()[ps[ok].argmax()]   # best-scoring box on this truck
            key = "called " + classes[int(pl[j]) - 1]
        called[key] += 1
        by_size[bucket][key] += 1

def per_class(res):
    return {classes[int(c) - 1]: float(v) * 100 for c, v in zip(res["classes"], res["map_per_class"])}

r_strict, r_50 = m_strict.compute(), m_50.compute()
s, f = per_class(r_strict), per_class(r_50)
print(f"\nPer-class results on {min(a.limit, len(d['images']))} images")
print(f"{'class':12s}{'AP50':>8s}{'mAP':>8s}")
for c in classes:
    print(f"{c:12s}{f.get(c, float('nan')):>8.1f}{s.get(c, float('nan')):>8.1f}")
total = sum(called.values())
print(f"\nWhat the model calls each true truck (confidence >= {a.thr}, {total} trucks):")
for k, v in called.most_common():
    print(f"  {k:28s}{v:5d}  ({100 * v / max(total, 1):.0f}%)")

print("\nSame breakdown by truck size (height in the original 1024-px-tall image):")
for b in sorted(by_size):
    n = sum(by_size[b].values())
    parts = ", ".join(f"{k.replace('called ', '')} {100 * v / n:.0f}%" for k, v in by_size[b].most_common())
    print(f"  {b:22s} n={n:3d}: {parts}")
