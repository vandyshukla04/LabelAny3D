"""
WildBox -> COCONUT format converter for the LabelAny3D pipeline.

For each frame path in --paths-file, this:
  1. Finds the parent seg<N>/ directory.
  2. Reads sam3_masks/metadata.json to get the category (text_prompt).
  3. For each sam3_masks/masks/obj_<i>/frame_<NNN>.png:
       - If non-empty, encodes the binary mask as a COCO RLE.
       - Emits one COCONUT annotation with the right category_id.
  4. Stages the frame as a symlink (or copy) under
     dataset/coco/images/val2017/ with a unique disambiguating name.
  5. Writes dataset/coco/annotations/coconut_val.json.

Run from src/:
    python tools/wildbox_to_coconut.py --paths-file ../wildbox_paths.txt
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# Category name -> id. Must match src/util.py COCO_CATEGORIES so the
# pipeline can resolve the name back from the id.
CATEGORY_NAME_TO_ID = {
    "elephant": 22,
    "zebra": 24,
    "giraffe": 25,
    "rhino": 201,
    "cow": 21,
    "sheep": 20,
    "horse": 19,
    "bear": 23,
}


def find_seg_dir(frame_path):
    seg = frame_path.parent
    meta = seg / "sam3_masks" / "metadata.json"
    if not meta.exists():
        raise FileNotFoundError(f"No sam3_masks/metadata.json next to {frame_path}")
    return seg


def get_category(seg_dir):
    meta = json.loads((seg_dir / "sam3_masks" / "metadata.json").read_text())
    name = meta["text_prompt"].strip().lower()
    if name not in CATEGORY_NAME_TO_ID:
        raise ValueError(
            f"Unknown category '{name}' from {seg_dir}. "
            f"Add it to CATEGORY_NAME_TO_ID here and to "
            f"src/util.py:COCO_CATEGORIES + src/tools/combine_results.py:COCO_CATEGORIES."
        )
    return name, CATEGORY_NAME_TO_ID[name]


def encode_mask(mask_path):
    arr = np.array(Image.open(mask_path))
    binary = (arr > 0).astype(np.uint8)
    if binary.sum() == 0:
        return None
    rle = mask_utils.encode(np.asfortranarray(binary))
    rle["counts"] = rle["counts"].decode("utf-8")
    bbox = mask_utils.toBbox(rle).tolist()
    area = int(mask_utils.area(rle))
    return rle, bbox, area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-file", required=True,
                    help="Text file with one absolute frame path per line. "
                         "Lines starting with # are ignored.")
    ap.add_argument("--repo-root", default=None,
                    help="LabelAny3D repo root. Default: parent of CWD if "
                         "running from src/, else CWD.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy frames into dataset/ instead of symlinking.")
    args = ap.parse_args()

    if args.repo_root is None:
        cwd = Path.cwd()
        repo_root = cwd.parent if cwd.name == "src" else cwd
    else:
        repo_root = Path(args.repo_root)
    repo_root = repo_root.resolve()

    img_dir = repo_root / "dataset/coco/images/val2017"
    ann_dir = repo_root / "dataset/coco/annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for line in Path(args.paths_file).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(Path(line))

    images = []
    annotations = []
    used_categories = {}
    ann_id = 1

    for idx, frame_path in enumerate(paths, start=1):
        if not frame_path.is_file():
            print(f"[{idx:02d}] SKIP missing file: {frame_path}", file=sys.stderr)
            continue

        seg_dir = find_seg_dir(frame_path)
        cat_name, cat_id = get_category(seg_dir)
        used_categories[cat_id] = cat_name

        stem = frame_path.stem
        staged_name = f"{idx:02d}_{cat_name}_{stem}.jpg"
        staged_path = img_dir / staged_name
        if staged_path.exists() or staged_path.is_symlink():
            staged_path.unlink()
        if args.copy:
            shutil.copy2(frame_path, staged_path)
        else:
            staged_path.symlink_to(frame_path.resolve())

        with Image.open(frame_path) as im:
            W, H = im.size

        image_id = idx
        images.append({
            "id": image_id,
            "file_name": staged_name,
            "width": W,
            "height": H,
        })

        masks_root = seg_dir / "sam3_masks" / "masks"
        n_added = 0
        for obj_dir in sorted(masks_root.glob("obj_*"), key=lambda p: int(p.name.split("_")[1])):
            mask_png = obj_dir / f"{stem}.png"
            if not mask_png.exists():
                continue
            res = encode_mask(mask_png)
            if res is None:
                continue
            rle, bbox, area = res
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(x) for x in bbox],
                "segmentation": rle,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1
            n_added += 1
        print(f"[{idx:02d}] {cat_name:9s} {staged_name}  ->  {n_added} objects")

    out = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": cname}
            for cid, cname in sorted(used_categories.items())
        ],
    }
    out_path = ann_dir / "coconut_val.json"
    out_path.write_text(json.dumps(out))
    print(f"\nWrote {len(images)} images, {len(annotations)} annotations -> {out_path}")
    print(f"Frames staged under {img_dir}")


if __name__ == "__main__":
    main()
