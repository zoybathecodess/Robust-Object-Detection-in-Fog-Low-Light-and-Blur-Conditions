import os
import shutil
import urllib.request
import zipfile
import json
import csv

def download_file(url, dest):
    print(f"Downloading from {url} to {dest}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out_f:
        shutil.copyfileobj(resp, out_f)
    print(f"Downloaded {dest}. Size: {os.path.getsize(dest)} bytes")

def main():
    os.makedirs("datasets/ExDark/images", exist_ok=True)
    os.makedirs("datasets/ExDark/annotations", exist_ok=True)
    os.makedirs("datasets/ExDark/temp", exist_ok=True)

    zip_path = "datasets/ExDark/temp/ExDark.zip"
    url = "https://huggingface.co/datasets/SatwikKambham/ex-dark/resolve/main/ExDark.zip"
    
    # 1. Download zip
    download_file(url, zip_path)

    # 2. Extract zip
    print("Extracting ExDark.zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("datasets/ExDark/temp")
    print("Extracted ExDark.zip.")

    # 3. Organize images
    # Check what is inside datasets/ExDark/temp
    extracted_root = "datasets/ExDark/temp/ExDark"
    if not os.path.exists(extracted_root):
        extracted_root = "datasets/ExDark/temp"

    # Move class folders (Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table) into datasets/ExDark/images/
    classes = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
    for c in classes:
        src_c = os.path.join(extracted_root, c)
        dst_c = os.path.join("datasets/ExDark/images", c)
        if os.path.exists(src_c):
            shutil.move(src_c, dst_c)
            print(f"Moved class {c}: {len(os.listdir(dst_c))} images")

    # Move metadata.csv
    meta_src = os.path.join(extracted_root, "metadata.csv")
    if os.path.exists(meta_src):
        shutil.move(meta_src, "datasets/ExDark/annotations/metadata.csv")
        print("Moved metadata.csv to annotations/")

    # 4. Download official imageclasslist.txt from GitHub
    github_classlist = "https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image-Dataset/master/Groundtruth/imageclasslist.txt"
    try:
        download_file(github_classlist, "datasets/ExDark/annotations/imageclasslist.txt")
    except Exception as e:
        print("Error downloading imageclasslist.txt:", e)

    # 5. Download COCO format train.json and test.json from fuhao7i/ExDark
    coco_train = "https://huggingface.co/datasets/fuhao7i/ExDark/raw/main/JPEGImages/train.json"
    coco_test = "https://huggingface.co/datasets/fuhao7i/ExDark/raw/main/JPEGImages/test.json"
    try:
        download_file(coco_train, "datasets/ExDark/annotations/coco_train.json")
        download_file(coco_test, "datasets/ExDark/annotations/coco_test.json")
    except Exception as e:
        print("Error downloading COCO json annotations:", e)

    # 6. Clean up temp folder
    shutil.rmtree("datasets/ExDark/temp")
    print("Cleaned up temp directory.")

if __name__ == "__main__":
    main()
