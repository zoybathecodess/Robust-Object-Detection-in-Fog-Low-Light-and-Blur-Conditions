"""CLI for the robust-detection pipeline. Run `python main.py <command> -h` for options."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DetManifest, UnlabeledImages, depth_key, det_collate, img_collate, load_manifest, read_rgb
from degradations import MidasDepth, degrade
from engine import evaluate, get_device, selftrain, train_supervised, tta_evaluate
from model import build_model, load_checkpoint, save_checkpoint


def dl(ds, bs, shuffle, workers, collate):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                      drop_last=shuffle, persistent_workers=workers > 0)


# --------------------------------------------------------------------------- commands
def cmd_prepare_cityscapes(a):
    from prepare_data import prepare_cityscapes
    prepare_cityscapes(a.cityscapes_root, a.out, a.foggy_root, a.beta)


def cmd_prepare_yolo(a):
    from prepare_data import prepare_yolo
    prepare_yolo(a.images, a.labels, a.classes.split(","), a.out)


def cmd_cache_depth(a):
    m = load_manifest(a.manifest)
    items = m["images"][:a.limit] if a.limit else m["images"]
    est = MidasDepth(str(get_device()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, it in enumerate(items):
        dst = out / (depth_key(it["file"]) + ".npy")
        if dst.exists():
            continue
        img, _ = read_rgb(Path(m["root"]) / it["file"], a.max_side)
        np.save(dst, est(img).astype(np.float16))
        if (n + 1) % 50 == 0:
            print(f"depth {n + 1}/{len(items)}", flush=True)


def cmd_preview(a):
    """Save a grid: original | fog | motion blur | low-light, to sanity-check the degradations."""
    ds = DetManifest(a.manifest, depth_dir=a.depth_dir, max_side=640, limit=a.n)
    rows = []
    for i in range(len(ds)):
        img = (ds[i][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rng = np.random.default_rng(i)
        fn = ds._depth_fn(ds.items[i]["file"])
        row = [img] + [degrade(img, k, fn, rng) for k in ("fog", "blur", "lowlight")]
        rows.append(np.hstack(row))
    cv2.imwrite(a.out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print("saved", a.out)


def cmd_train(a):
    device = get_device()
    degrade_mode = None if a.mode == "baseline" else "random"
    ds = DetManifest(a.train_manifest, degrade=degrade_mode, depth_dir=a.depth_dir, train=True,
                     max_side=a.max_side, limit=a.limit)
    model = build_model(len(ds.classes), a.backbone, pretrained=not a.no_pretrained,
                        min_size=a.min_size, max_size=a.max_side)
    train_supervised(model, dl(ds, a.batch_size, True, a.workers, det_collate), a.epochs, a.lr, device)
    save_checkpoint(a.out, model, ds.classes, a.backbone, a.min_size, a.max_side)
    print("saved", a.out)


def cmd_selftrain(a):
    device = get_device()
    model, ck = load_checkpoint(a.init, device)
    lab = DetManifest(a.train_manifest, degrade="random", depth_dir=a.depth_dir, train=True,
                      max_side=ck["max_size"], limit=a.limit)
    unl = UnlabeledImages(a.unlabeled, include=a.include, max_side=ck["max_size"], limit=a.unl_limit)
    print(f"labeled: {len(lab)}  unlabeled real-degraded: {len(unl)}")
    teacher = selftrain(model, dl(lab, a.batch_size, True, a.workers, det_collate),
                        dl(unl, a.batch_size, True, a.workers, img_collate),
                        device, a.steps, a.lr, a.thr, a.lam, a.ema)
    save_checkpoint(a.out, teacher, ck["classes"], ck["backbone"], ck["min_size"], ck["max_size"])
    print("saved", a.out)


def cmd_eval(a):
    device = get_device()
    model, ck = load_checkpoint(a.ckpt, device)
    suites = {"clean": (a.val_manifest, None), "synth_fog": (a.val_manifest, "fog"),
              "synth_blur": (a.val_manifest, "blur"), "synth_lowlight": (a.val_manifest, "lowlight")}
    for kv in a.extra or []:  # e.g. foggy_val=manifests/foggy_val.json
        name, path = kv.split("=", 1)
        suites[name] = (path, None)
    results = {}
    for name, (path, mode) in suites.items():
        ds = DetManifest(path, degrade=mode, depth_dir=a.depth_dir, max_side=ck["max_size"],
                         limit=a.limit, fixed_seed=True)
        loader = dl(ds, a.batch_size, False, a.workers, det_collate)
        r = {"static": evaluate(model, loader, device)}
        if a.tta:
            r["tta"] = tta_evaluate(model, loader, device, a.tta_lr, a.tta_thr, a.episodic)
        results[name] = r
        print(name, r, flush=True)
    if a.save:
        json.dump(results, open(a.save, "w"), indent=2)
    print("\n%-16s %s" % ("condition", "mAP / mAP50"))
    for name, r in results.items():
        print("%-16s %s" % (name, "   ".join(f"{k}: {v['mAP']:.1f}/{v['mAP50']:.1f}" for k, v in r.items())))


# --------------------------------------------------------------------------- parser
def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("prepare-cityscapes")
    s.add_argument("--cityscapes_root", required=True)
    s.add_argument("--foggy_root", help="defaults to cityscapes_root")
    s.add_argument("--beta", default="0.02", choices=["0.005", "0.01", "0.02"])
    s.add_argument("--out", default="manifests")
    s.set_defaults(f=cmd_prepare_cityscapes)

    s = sp.add_parser("prepare-yolo")
    s.add_argument("--images", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--classes", required=True, help="comma separated class names in YOLO id order")
    s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_prepare_yolo)

    s = sp.add_parser("cache-depth")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", default="cache/depth")
    s.add_argument("--max_side", type=int, default=1024)
    s.add_argument("--limit", type=int)
    s.set_defaults(f=cmd_cache_depth)

    s = sp.add_parser("preview")
    s.add_argument("--manifest", required=True)
    s.add_argument("--depth_dir")
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--out", default="preview.jpg")
    s.set_defaults(f=cmd_preview)

    def common(s):
        s.add_argument("--depth_dir", help="folder of cached depth maps (else a ground-plane prior)")
        s.add_argument("--batch_size", type=int, default=4)
        s.add_argument("--workers", type=int, default=2)
        s.add_argument("--limit", type=int, help="use only first N labeled images (debugging)")

    s = sp.add_parser("train")
    common(s)
    s.add_argument("--mode", choices=["baseline", "aug"], required=True,
                   help="baseline = clean only; aug = + synthetic degradations")
    s.add_argument("--train_manifest", required=True)
    s.add_argument("--backbone", default="mobilenet", choices=["mobilenet", "resnet50"])
    s.add_argument("--no_pretrained", action="store_true")
    s.add_argument("--epochs", type=int, default=10)
    s.add_argument("--lr", type=float, default=0.01)
    s.add_argument("--min_size", type=int, default=512)
    s.add_argument("--max_side", type=int, default=1024)
    s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_train)

    s = sp.add_parser("selftrain")
    common(s)
    s.add_argument("--init", required=True, help="checkpoint from `train --mode aug`")
    s.add_argument("--train_manifest", required=True)
    s.add_argument("--unlabeled", nargs="+", required=True, help="folder(s) of unlabeled degraded images")
    s.add_argument("--include", nargs="*", help="only paths containing these substrings")
    s.add_argument("--unl_limit", type=int)
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--lr", type=float, default=0.005)
    s.add_argument("--thr", type=float, default=0.7)
    s.add_argument("--lam", type=float, default=1.0)
    s.add_argument("--ema", type=float, default=0.999)
    s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_selftrain)

    s = sp.add_parser("eval")
    common(s)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--val_manifest", required=True, help="clean labeled val manifest")
    s.add_argument("--extra", nargs="*", help="name=manifest.json, e.g. foggy_val=manifests/foggy_val.json")
    s.add_argument("--tta", action="store_true")
    s.add_argument("--tta_lr", type=float, default=1e-3)
    s.add_argument("--tta_thr", type=float, default=0.8)
    s.add_argument("--episodic", action="store_true")
    s.add_argument("--save")
    s.set_defaults(f=cmd_eval)

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
