# Robust object detection under fog / blur / low light

Pipeline: synthetic degradations (depth-aware fog, motion blur, low-light sensor noise) →
augmentation-only training → teacher-student self-training on unlabeled real degraded images →
test-time adaptation (TTA) → benchmark.

```
degradations.py   fog (Koschmieder + MiDaS depth), motion blur, Poisson+Gaussian low-light
data.py           manifest dataset (on-the-fly degradation), unlabeled folder dataset, strong augmentation
model.py          Faster R-CNN builder, FPN scale/shift adapter used for TTA
engine.py         supervised loop, mean-teacher self-training, mAP evaluation, online TTA
prepare_data.py   Cityscapes / YOLO -> manifest JSON
main.py           CLI
```

## 1. Install

```bash
pip install torch torchvision opencv-python numpy pillow torchmetrics pycocotools timm
```

## 2. Dataset setup (step by step)

You never upload data into the code. You put it on disk, convert annotations to a small JSON
"manifest" that stores the root path, and pass manifests/folders on the command line.

```
project/
  datasets/
    cityscapes/   leftImg8bit/{train,val}/<city>/*.png   gtFine/{train,val}/<city>/*_instanceIds.png
                  leftImg8bit_foggy/{train,val}/<city>/*_foggy_beta_0.02.png
    IndiaScene365/  (from your download scripts)
    ExDark/         (from your download scripts)
  manifests/        (created below)
  cache/depth/      (created below)
```

### A. Labeled source data: Cityscapes (benchmark)
1. Register (free) at https://www.cityscapes-dataset.com and download `leftImg8bit_trainvaltest.zip`
   and `gtFine_trainvaltest.zip`.
2. Foggy Cityscapes: same site, download `leftImg8bit_trainval_foggy.zip` (contains beta 0.005/0.01/0.02).
3. Unzip all three into `datasets/cityscapes/` so the folder layout above is produced.
4. Build manifests (boxes are derived from the instance masks, 8 classes):
   ```bash
   python main.py prepare-cityscapes --cityscapes_root datasets/cityscapes --beta 0.02 --out manifests
   ```
   Creates `cityscapes_train.json`, `cityscapes_val.json`, `foggy_train.json`, `foggy_val.json`.

### B. Unlabeled real degraded data (for self-training)
Use the scripts you already have; no labels are needed for this step.
- `download_and_organize_indiascene365.py` -> `datasets/IndiaScene365/images/Day2fog/...` (real fog, also Day2Rainy).
- `download_and_organize_exdark.py` -> `datasets/ExDark/images/<Class>/*.jpg` (real low light).

Pass the folder(s) to `--unlabeled`; `--include Day2fog` restricts to paths containing that text.
Using **Foggy Cityscapes train images as unlabeled** (ignore their labels) is the standard
Cityscapes -> Foggy Cityscapes protocol: point `--unlabeled datasets/cityscapes/leftImg8bit_foggy/train`.

### C. Using your own dataset
- YOLO-format labels:
  ```bash
  python main.py prepare-yolo --images datasets/MyData/images --labels datasets/MyData/labels \
      --classes car,person,bus --out manifests/mydata_train.json
  ```
- Any other format: write a JSON with this shape (labels are 1-based, boxes are x1,y1,x2,y2 pixels):
  ```json
  {"root": "/abs/path/to/images", "classes": ["car","person"],
   "images": [{"file": "sub/img1.png", "boxes": [[10,20,200,180]], "labels": [1]}]}
  ```

### D. Optional: cache monocular depth (better fog, run once)
```bash
python main.py cache-depth --manifest manifests/cityscapes_train.json --out cache/depth
```
Uses MiDaS (downloads weights via torch.hub; needs internet once). Without `--depth_dir`, fog uses a
ground-plane prior (far at top) - fine for road scenes, worse otherwise. Cache the val set too if you
want depth-aware synthetic-fog evaluation.

### E. Look at the degradations before training
```bash
python main.py preview --manifest manifests/cityscapes_val.json --depth_dir cache/depth --n 4
```

## 3. Run the experiments

```bash
# baseline: clean only
python main.py train --mode baseline --train_manifest manifests/cityscapes_train.json --epochs 10 --out ckpt/baseline.pt
# augmentation-only: random fog / blur / low light on the fly
python main.py train --mode aug --train_manifest manifests/cityscapes_train.json --depth_dir cache/depth --epochs 10 --out ckpt/aug.pt
# + self-training on unlabeled real degraded images (starts from aug.pt)
python main.py selftrain --init ckpt/aug.pt --train_manifest manifests/cityscapes_train.json --depth_dir cache/depth \
    --unlabeled datasets/cityscapes/leftImg8bit_foggy/train --steps 3000 --out ckpt/self.pt
# evaluate (add --tta for test-time adaptation)
for c in baseline aug self; do
  python main.py eval --ckpt ckpt/$c.pt --val_manifest manifests/cityscapes_val.json \
      --extra foggy_val=manifests/foggy_val.json --depth_dir cache/depth --tta --save results_$c.json
done
```
The table printed per checkpoint gives mAP / mAP50 on clean, synthetic fog/blur/low-light, and real
Foggy Cityscapes, each without ("static") and with TTA. Quick debug run: add `--limit 50 --epochs 1`.
Use `--backbone resnet50` for higher accuracy (needs more GPU memory).

## Notes
- Do the self-training step from the augmentation model, and keep `--thr` high (0.7-0.9); lower values let
  wrong pseudo-labels reinforce themselves.
- TTA only updates ~2-3k FPN scale/shift parameters; use `--episodic` to reset after every batch if it drifts.
- Foggy Cityscapes is itself synthetic. For truly real conditions, evaluate on a labeled real set
  (e.g. IndiaScene365 YOLO labels via `prepare-yolo`, or ExDark's `imageclasslist`).
- Fog parameters: `beta` ~ 0.005 (light) to 0.03 (dense) per metre; `max_depth` maps relative depth to metres.
