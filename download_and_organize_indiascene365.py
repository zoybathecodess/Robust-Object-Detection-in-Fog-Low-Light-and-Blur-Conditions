import os
import json
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_folders():
    url = "https://data.mendeley.com/public-api/datasets/pwffhg6nhz/folders/2"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        folders = json.loads(resp.read().decode())
    folder_dict = {f["id"]: f for f in folders}
    def get_path(fid):
        parts = []
        curr = fid
        while curr and curr in folder_dict:
            parts.append(folder_dict[curr]["name"])
            curr = folder_dict[curr].get("parent_id")
        return "/".join(reversed(parts)) if parts else "root"
    
    id_to_path = {}
    for f in folders:
        id_to_path[f["id"]] = get_path(f["id"])
    return id_to_path

def get_all_files():
    all_files = []
    for start in range(0, 2800, 100):
        url = f"https://data.mendeley.com/api/datasets/pwffhg6nhz/files?version=2&$start={start}&$limit=100"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    batch = json.loads(resp.read().decode())
                    all_files.extend(batch)
                    break
            except Exception as e:
                time.sleep(1 + attempt)
    return all_files

def map_destination(item, id_to_path):
    fname = item["filename"]
    fid = item.get("folder_id")
    rel_path = id_to_path.get(fid, "other")
    # rel_path looks like: Indiascene365/Day2fog/Day/Train A
    # or Indiascene365/XML Files/Day
    # or Indiascene365/labels/Day
    
    parts = rel_path.split("/")
    if len(parts) > 1 and parts[0] == "Indiascene365":
        sub_type = parts[1] # "Day2fog", "Day2Rainy", "XML Files", "labels"
        rest = "/".join(parts[2:])
        if sub_type in ["Day2fog", "Day2Rainy"]:
            target_dir = os.path.join("datasets/IndiaScene365/images", sub_type, rest)
        elif sub_type == "XML Files":
            target_dir = os.path.join("datasets/IndiaScene365/annotations/pascal_voc", rest)
        elif sub_type == "labels":
            target_dir = os.path.join("datasets/IndiaScene365/annotations/yolo", rest)
        else:
            target_dir = os.path.join("datasets/IndiaScene365/other", rest)
    else:
        if fname.endswith(".png"):
            target_dir = "datasets/IndiaScene365/images"
        elif fname.endswith(".xml"):
            target_dir = "datasets/IndiaScene365/annotations/pascal_voc"
        elif fname.endswith(".txt"):
            target_dir = "datasets/IndiaScene365/annotations/yolo"
        else:
            target_dir = "datasets/IndiaScene365"
            
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, fname)

def download_item(args):
    item, dest = args
    fname = item["filename"]
    size = item.get("content_details", {}).get("size", 0)
    dl_url = item["content_details"]["download_url"]

    if os.path.exists(dest) and os.path.getsize(dest) == size and size > 0:
        return True, fname, "cached"

    for attempt in range(4):
        try:
            req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as in_f, open(dest, "wb") as out_f:
                out_f.write(in_f.read())
            return True, fname, "downloaded"
        except Exception as e:
            if attempt == 3:
                return False, fname, str(e)
            time.sleep(1 + attempt)

def main():
    print("Fetching folder structure...")
    id_to_path = get_folders()
    print("Fetching file list (28 pages)...")
    files = get_all_files()
    print(f"Total files in metadata: {len(files)}")

    tasks = []
    for f in files:
        dest = map_destination(f, id_to_path)
        tasks.append((f, dest))

    print(f"Prepared {len(tasks)} download tasks. Starting downloads...")
    success_count = 0
    fail_count = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_item, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            success, fname, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"Failed {fname}: {msg}")
            
            if i % 300 == 0 or i == total:
                print(f"Progress: {i}/{total} (Success: {success_count}, Failed: {fail_count})")

    print(f"Finished! Total: {total}, Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
