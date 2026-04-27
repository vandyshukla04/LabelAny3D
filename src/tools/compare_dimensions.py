"""
Side-by-side dimension comparison: VGGT vs LabelAny3D, per frame, per object.

For every frame in wildbox_paths.txt that produced LabelAny3D output, this:
  1. Reads VGGT's KITTI label file at <seg>/vggt_results/kitti_labels/<stem>.txt
     - Columns 8/9/10 = h/w/l in meters.
  2. Reads LabelAny3D's experimental_results/COCO/<split>/<idx>_<species>_<stem>/3dbbox.json
     - "dimensions" field: a 3-tuple (the order is implementation-defined; we
       sort each box's dims to compare longest/mid/shortest invariantly).
  3. Per box: extracts (longest, middle, shortest) extents.
  4. Per frame: aggregates count and mean longest/mid/shortest for each method.
  5. Prints a side-by-side table with a "real-world reference" row for the
     known animal size, so you can see how off each method is.

This is a *timing+sanity* comparison, not an accuracy benchmark — there's
no ground-truth 3D box for either method to be evaluated against.

Usage (from src/):
    python tools/compare_dimensions.py
    python tools/compare_dimensions.py --paths-file ../wildbox_paths.txt --split val
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


# Approx adult body dimensions (meters): longest, middle, shortest extent.
# Sources: standard wildlife biology refs; treat as orientation-agnostic.
REALITY_REF = {
    "elephant": (6.0, 3.0, 1.5),
    "zebra":    (2.4, 1.4, 0.8),
    "giraffe":  (5.5, 2.5, 1.0),  # incl. neck; "longest" axis is the body length
    "rhino":    (3.8, 1.8, 1.6),
    "cow":      (2.5, 1.5, 0.8),
    "sheep":    (1.3, 0.9, 0.5),
    "horse":    (2.4, 1.6, 0.9),
}


def load_kitti_dims(kitti_path):
    """Returns list of (longest, middle, shortest) tuples, one per object."""
    if not kitti_path.exists():
        return None
    out = []
    for line in kitti_path.read_text().strip().splitlines():
        cols = line.split()
        if len(cols) < 11:
            continue
        # KITTI: type trunc occl alpha x1 y1 x2 y2 h w l x y z ry [score]
        h, w, l = float(cols[8]), float(cols[9]), float(cols[10])
        dims = sorted((h, w, l), reverse=True)  # longest, mid, shortest
        out.append(tuple(dims))
    return out


def load_la3d_dims(bbox_json_path):
    """Returns list of (longest, middle, shortest) tuples, one per object."""
    if not bbox_json_path.exists():
        return None
    boxes = json.loads(bbox_json_path.read_text())
    out = []
    for b in boxes:
        dims = b.get("dimensions")
        if dims is None or len(dims) != 3:
            continue
        out.append(tuple(sorted([float(x) for x in dims], reverse=True)))
    return out


def fmt_dims(dims_tuple):
    if dims_tuple is None:
        return "-"
    return f"{dims_tuple[0]:.2f}/{dims_tuple[1]:.2f}/{dims_tuple[2]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-file", default="../wildbox_paths.txt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--results-dir", default="../experimental_results/COCO")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths_file = Path(args.paths_file)
    if not paths_file.exists():
        raise SystemExit(f"Paths file not found: {paths_file}")
    paths = []
    for line in paths_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(Path(line))

    val_root = Path(args.results_dir) / args.split
    rows = []

    print(f"\n{'frame':40s}  {'spec':9s}  {'method':9s}  {'n':>3s}  "
          f"{'longest/mid/short (m)':24s}  {'reality (m)':18s}")
    print("-" * 120)

    for idx, frame_path in enumerate(paths, start=1):
        stem = frame_path.stem
        seg_dir = frame_path.parent
        # Source: VGGT
        kitti = seg_dir / "vggt_results" / "kitti_labels" / f"{stem}.txt"
        vggt_dims = load_kitti_dims(kitti)

        # Find the LabelAny3D output dir by index prefix
        candidates = list(val_root.glob(f"{idx:02d}_*_{stem}"))
        if not candidates:
            print(f"[{idx:02d}] no LabelAny3D output found for stem {stem}, skip")
            continue
        scene_dir = candidates[0]
        species = scene_dir.name.split("_", 2)[1]
        bbox_json = scene_dir / "3dbbox.json"
        la3d_dims = load_la3d_dims(bbox_json)

        ref = REALITY_REF.get(species)
        ref_str = fmt_dims(ref) if ref else "-"

        def aggregate(dim_lists):
            if not dim_lists:
                return None, 0
            longs = [d[0] for d in dim_lists]
            mids = [d[1] for d in dim_lists]
            shorts = [d[2] for d in dim_lists]
            return (mean(longs), mean(mids), mean(shorts)), len(dim_lists)

        vggt_avg, vggt_n = aggregate(vggt_dims) if vggt_dims is not None else (None, 0)
        la3d_avg, la3d_n = aggregate(la3d_dims) if la3d_dims is not None else (None, 0)

        if vggt_dims is None and la3d_dims is None:
            continue

        print(f"{scene_dir.name:40s}  {species:9s}  {'VGGT':9s}  "
              f"{vggt_n:>3d}  {fmt_dims(vggt_avg):24s}  {ref_str:18s}")
        print(f"{'':40s}  {'':9s}  {'LabelAny3D':9s}  "
              f"{la3d_n:>3d}  {fmt_dims(la3d_avg):24s}  {ref_str:18s}")

        # Per-object rows for the CSV
        for i, d in enumerate(vggt_dims or []):
            rows.append({"scene": scene_dir.name, "species": species,
                         "method": "VGGT", "obj_idx": i,
                         "longest_m": round(d[0], 3), "middle_m": round(d[1], 3),
                         "shortest_m": round(d[2], 3),
                         "ref_longest_m": ref[0] if ref else "",
                         "ref_middle_m": ref[1] if ref else "",
                         "ref_shortest_m": ref[2] if ref else ""})
        for i, d in enumerate(la3d_dims or []):
            rows.append({"scene": scene_dir.name, "species": species,
                         "method": "LabelAny3D", "obj_idx": i,
                         "longest_m": round(d[0], 3), "middle_m": round(d[1], 3),
                         "shortest_m": round(d[2], 3),
                         "ref_longest_m": ref[0] if ref else "",
                         "ref_middle_m": ref[1] if ref else "",
                         "ref_shortest_m": ref[2] if ref else ""})

    # Per-species means (average of longest dim across all boxes of that species,
    # for each method)
    print("\nSpecies-level averages (longest extent, meters):")
    print(f"  {'species':10s}  {'VGGT':>8s} (n)   {'LabelAny3D':>11s} (n)   {'reality':>8s}   "
          f"{'VGGT/real':>9s}  {'LA3D/real':>9s}")
    species_set = sorted({r["species"] for r in rows})
    for sp in species_set:
        v = [r["longest_m"] for r in rows if r["species"] == sp and r["method"] == "VGGT"]
        l = [r["longest_m"] for r in rows if r["species"] == sp and r["method"] == "LabelAny3D"]
        ref = REALITY_REF.get(sp)
        ref_long = ref[0] if ref else None
        v_avg = mean(v) if v else None
        l_avg = mean(l) if l else None
        v_ratio = (v_avg / ref_long) if (v_avg is not None and ref_long) else None
        l_ratio = (l_avg / ref_long) if (l_avg is not None and ref_long) else None
        print(f"  {sp:10s}  "
              f"{(f'{v_avg:8.2f}' if v_avg is not None else '       -'):>8s}"
              f" ({len(v):>2d})   "
              f"{(f'{l_avg:11.2f}' if l_avg is not None else '          -'):>11s}"
              f" ({len(l):>2d})   "
              f"{(f'{ref_long:8.2f}' if ref_long else '       -'):>8s}   "
              f"{(f'{v_ratio:8.3f}x' if v_ratio else '         -'):>9s}  "
              f"{(f'{l_ratio:8.3f}x' if l_ratio else '         -'):>9s}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.results_dir) / "dimensions_comparison.csv"
    fields = ["scene", "species", "method", "obj_idx",
              "longest_m", "middle_m", "shortest_m",
              "ref_longest_m", "ref_middle_m", "ref_shortest_m"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} per-object rows -> {out_path}")
    print("\nNote: both methods estimate dimensions from monocular depth,")
    print("so absolute scale will be off (often 10-100x small for aerial drone footage).")
    print("The interesting question is *relative* — does one method get closer to reality?")


if __name__ == "__main__":
    main()
