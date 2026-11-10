"""Convert raw datasets into the manifest JSON format used by data.py."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CS_CLASSES = {24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus",
              31: "train", 32: "motorcycle", 33: "bicycle"}
CS_IDS = sorted(CS_CLASSES)


def prepare_cityscapes(cs_root, out_dir, foggy_root=None, beta="0.02", splits=("train", "val"), min_box=4):
    """Builds boxes from gtFine *_instanceIds.png (instance id = class_id * 1000 + k).

    Writes cityscapes_{split}.json and, if foggy images are found, foggy_{split}.json
    (same boxes, foggy image paths). Foggy boxes are identical because fog does not move objects."""
    cs_root, out_dir = Path(cs_root), Path(out_dir)
    foggy_root = Path(foggy_root) if foggy_root else cs_root
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = [CS_CLASSES[i] for i in CS_IDS]
    for split in splits:
        gts = sorted((cs_root / "gtFine" / split).glob("*/*_gtFine_instanceIds.png"))
        if not gts:
            print(f"[!] no gtFine annotations for split '{split}' under {cs_root}/gtFine/{split}")
            continue
        clear, foggy = [], []
        for n, g in enumerate(gts):
            inst = cv2.imread(str(g), cv2.IMREAD_UNCHANGED)
            boxes, labels = [], []
            for iid in np.unique(inst):
                if iid < 1000 or iid // 1000 not in CS_CLASSES:
                    continue
                ys, xs = np.where(inst == iid)
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                if x2 - x1 < min_box or y2 - y1 < min_box:
                    continue
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                labels.append(CS_IDS.index(int(iid // 1000)) + 1)
            base = g.name.replace("_gtFine_instanceIds.png", "")
            city = g.parent.name
            rec = {"boxes": boxes, "labels": labels}
            clear.append({"file": f"leftImg8bit/{split}/{city}/{base}_leftImg8bit.png", **rec})
            frel = f"leftImg8bit_foggy/{split}/{city}/{base}_leftImg8bit_foggy_beta_{beta}.png"
            if (foggy_root / frel).exists():
                foggy.append({"file": frel, **rec})
            if (n + 1) % 500 == 0:
                print(f"  {split}: {n + 1}/{len(gts)}")
        json.dump({"root": str(cs_root.resolve()), "classes": classes, "images": clear},
                  open(out_dir / f"cityscapes_{split}.json", "w"))
        print(f"cityscapes_{split}.json: {len(clear)} images")
        if foggy:
            json.dump({"root": str(foggy_root.resolve()), "classes": classes, "images": foggy},
                      open(out_dir / f"foggy_{split}.json", "w"))
            print(f"foggy_{split}.json: {len(foggy)} images (beta={beta})")
        else:
            print(f"[!] no foggy images found for '{split}' (looked for {foggy_root}/leftImg8bit_foggy/...)")


def prepare_yolo(images_dir, labels_dir, class_names, out_json):
    """YOLO txt labels (class cx cy w h, normalised) -> manifest. Matches files by stem."""
    from PIL import Image

    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    lab = {p.stem: p for p in labels_dir.rglob("*.txt")}
    items = []
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp") or p.stem not in lab:
            continue
        w, h = Image.open(p).size
        boxes, labels = [], []
        for line in lab[p.stem].read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            labels.append(c + 1)
        items.append({"file": str(p.relative_to(images_dir)), "boxes": boxes, "labels": labels})
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"root": str(images_dir.resolve()), "classes": list(class_names), "images": items},
              open(out_json, "w"))
    print(f"{out_json}: {len(items)} images")
