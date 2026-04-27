"""
Walk experimental_results/COCO/<split>/ and dump a flat CSV of every
detected object across all frames.

For each per-frame 3dbbox.json, emits one row per object with:
  frame, obj_id, category, length_m, height_m, width_m,
  cx, cy, cz (camera coords, meters), distance_m, has_glb, has_vis_png

Usage (from src/):
    python tools/summarize_results.py --split val
    python tools/summarize_results.py --split val --out /path/to/summary.csv
"""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--results-dir", default="../experimental_results/COCO")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <results_dir>/<split>_summary.csv)")
    args = ap.parse_args()

    root = Path(args.results_dir) / args.split
    if not root.is_dir():
        raise SystemExit(f"No directory: {root}")

    out_path = Path(args.out) if args.out else Path(args.results_dir) / f"{args.split}_summary.csv"

    rows = []
    frame_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    for fd in frame_dirs:
        bbox_json = fd / "3dbbox.json"
        if not bbox_json.exists():
            rows.append({"frame": fd.name, "status": "NO_3DBBOX"})
            continue
        boxes = json.loads(bbox_json.read_text())
        recon_dir = fd / "reconstruction"
        vis = fd / "vis_3dbox.png"
        for b in boxes:
            cx, cy, cz = b.get("center_cam", [None, None, None])
            d, h, w = b.get("dimensions", [None, None, None])
            obj_id = b.get("obj_id", "")
            cat = b.get("category_name", "")
            distance = math.sqrt(cx * cx + cy * cy + cz * cz) if cx is not None else None
            glb_path = recon_dir / f"{obj_id}_{cat}.glb"
            rows.append({
                "frame": fd.name,
                "obj_id": obj_id,
                "category": cat,
                "length_m": round(d, 3) if d is not None else "",
                "height_m": round(h, 3) if h is not None else "",
                "width_m": round(w, 3) if w is not None else "",
                "cx_m": round(cx, 3) if cx is not None else "",
                "cy_m": round(cy, 3) if cy is not None else "",
                "cz_m": round(cz, 3) if cz is not None else "",
                "distance_m": round(distance, 3) if distance is not None else "",
                "has_glb": glb_path.exists(),
                "has_vis_png": vis.exists(),
                "status": "OK",
            })

    fields = ["frame", "obj_id", "category",
              "length_m", "height_m", "width_m",
              "cx_m", "cy_m", "cz_m", "distance_m",
              "has_glb", "has_vis_png", "status"]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    print(f"Wrote {len(rows)} rows -> {out_path}")
    n_frames = len(frame_dirs)
    n_ok = sum(1 for r in rows if r.get("status") == "OK")
    n_fail = sum(1 for r in rows if r.get("status") == "NO_3DBBOX")
    print(f"Frames processed: {n_frames}    Objects with 3D box: {n_ok}    Frames missing 3dbbox: {n_fail}")

    # console table preview
    print("\nFirst 20 rows:")
    print(f"{'frame':30s}  {'obj':4s}  {'cat':9s}  {'l':>5s}  {'h':>5s}  {'w':>5s}  {'dist':>6s}")
    for r in rows[:20]:
        if r.get("status") != "OK":
            print(f"{r['frame']:30s}  --     {r['status']}")
            continue
        print(f"{r['frame']:30s}  {r['obj_id']:4s}  {r['category']:9s}  "
              f"{r['length_m']:>5}  {r['height_m']:>5}  {r['width_m']:>5}  {r['distance_m']:>6}")


if __name__ == "__main__":
    main()
