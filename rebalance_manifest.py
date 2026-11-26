"""
rebalance_manifest.py - repeat training images that contain rare classes (e.g. truck) so the model sees them more often.

    python rebalance_manifest.py                       # truck x3, bus x2, train x3
    python rebalance_manifest.py --boost truck=4 bus=2

Writes manifests/cityscapes_train_balanced.json (same format as the original manifest).
An image containing several boosted classes is repeated by the LARGEST boost among them.
"""
import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="manifests/cityscapes_train.json")
ap.add_argument("--out", default="manifests/cityscapes_train_balanced.json")
ap.add_argument("--boost", nargs="+", default=["truck=3", "bus=2", "train=3"])
a = ap.parse_args()

d = json.load(open(a.manifest))
classes = d["classes"]
boost = {}
for item in a.boost:
    name, k = item.split("=")
    boost[classes.index(name) + 1] = int(k)          # labels in the manifest are 1-based


def stats(images):
    inst, imgs = collections.Counter(), collections.Counter()
    for im in images:
        inst.update(im["labels"])
        imgs.update(set(im["labels"]))
    return inst, imgs


new_images = []
for im in d["images"]:
    r = max([boost.get(l, 1) for l in set(im["labels"])] or [1])
    new_images.extend([im] * r)

print(f"images: {len(d['images'])} -> {len(new_images)}")
print(f"{'class':12s}{'instances before':>18s}{'after':>10s}{'images before':>16s}{'after':>10s}")
bi, bm = stats(d["images"])
ai, am = stats(new_images)
for i, c in enumerate(classes, 1):
    print(f"{c:12s}{bi[i]:>18d}{ai[i]:>10d}{bm[i]:>16d}{am[i]:>10d}")

d2 = dict(d)
d2["images"] = new_images
json.dump(d2, open(a.out, "w"))
print(f"saved {a.out}  ({len(new_images) // 2} steps per epoch at batch size 2)")
