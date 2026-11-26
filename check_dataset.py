"""Run from the project folder:  python check_dataset.py
Reports what is present under datasets/ and what to do about anything missing."""
from pathlib import Path

ROOT = Path("datasets")
IMG = {".jpg", ".jpeg", ".png", ".bmp"}


def count(folder, pattern="*"):
    return sum(1 for _ in Path(folder).rglob(pattern)) if Path(folder).exists() else 0


def count_img(folder):
    p = Path(folder)
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in IMG) if p.exists() else 0


def line(ok, label, n, fix=""):
    print(f"[{'OK' if ok else '!!'}] {label}: {n}" + ("" if ok else f"   -> {fix}"))
    return ok


cs = ROOT / "cityscapes"
print("== Cityscapes (labeled source data) ==")
a = line(count_img(cs / "leftImg8bit") > 0, "clear images (leftImg8bit)", count_img(cs / "leftImg8bit"),
         "unzip leftImg8bit_trainvaltest.zip into datasets/cityscapes/")
b = line(count(cs / "gtFine", "*_instanceIds.png") > 0, "labels (gtFine instanceIds)",
         count(cs / "gtFine", "*_instanceIds.png"), "unzip gtFine_trainvaltest.zip into datasets/cityscapes/")
c = line(count_img(cs / "leftImg8bit_foggy") > 0, "foggy images (leftImg8bit_foggy)",
         count_img(cs / "leftImg8bit_foggy"),
         "unzip leftImg8bit_trainvaltest_foggy.zip into datasets/cityscapes/ (needed only for the final real-fog test)")

print("\n== Unlabeled real degraded data (for self-training) ==")
d = line(count_img(ROOT / "IndiaScene365") > 0, "IndiaScene365 images", count_img(ROOT / "IndiaScene365"),
         "python download_and_organize_indiascene365.py")
e = line(count_img(ROOT / "ExDark") > 0, "ExDark images", count_img(ROOT / "ExDark"),
         "python download_and_organize_exdark.py")

print("\n== Generated files ==")
_m = Path("manifests/cityscapes_train.json").exists()
line(_m, "manifests/cityscapes_train.json", "found" if _m else "missing",
     "python main.py prepare-cityscapes --cityscapes_root datasets/cityscapes --out manifests")

print("\nMinimum to start: clear images + labels (first two lines OK).")
