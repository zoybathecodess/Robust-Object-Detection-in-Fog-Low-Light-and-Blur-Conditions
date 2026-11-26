"""
finetune.py - continue training an existing checkpoint (with the synthetic-degradation augmentation) on a manifest.

    python finetune.py --init ckpt/r50_aug.pt --train_manifest manifests/cityscapes_train_balanced.json \
        --out ckpt/r50_aug_balanced.pt --epochs 2 --lr 0.003 --depth_dir cache/depth

Uses the same data loading, augmentation, training loop and checkpoint format as `main.py train`;
the only difference is that it starts from --init instead of from COCO-pretrained weights.
Optional: --min_size / --max_side fine-tune at a higher resolution (helps small, distant objects);
the new sizes are saved in the checkpoint, so eval and the app pick them up automatically.
"""
import argparse

from torch.utils.data import DataLoader

from data import DetManifest, det_collate
from engine import get_device, train_supervised
from model import load_checkpoint, save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="checkpoint to start from, e.g. ckpt/r50_aug.pt")
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--depth_dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debugging: use only the first N images")
    ap.add_argument("--min_size", type=int, default=None, help="shorter image side; default = same as --init")
    ap.add_argument("--max_side", type=int, default=None, help="longer image side; default = same as --init")
    a = ap.parse_args()

    device = get_device()
    model, ck = load_checkpoint(a.init, device)
    min_size, max_side = a.min_size or ck["min_size"], a.max_side or ck["max_size"]
    model.transform.min_size = (min_size,)          # torchvision resizes inputs inside the model
    model.transform.max_size = max_side
    ds = DetManifest(a.train_manifest, degrade="random", depth_dir=a.depth_dir, train=True,
                     max_side=max_side, limit=a.limit)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                        collate_fn=det_collate, drop_last=True, persistent_workers=a.workers > 0)
    print(f"fine-tuning {a.init} on {len(ds)} images at {min_size}/{max_side}: "
          f"{len(loader)} steps per epoch, {a.epochs} epochs")
    train_supervised(model, loader, a.epochs, a.lr, device)
    save_checkpoint(a.out, model, ck["classes"], ck["backbone"], min_size, max_side)
    print("saved", a.out)


if __name__ == "__main__":
    main()
