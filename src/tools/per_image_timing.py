"""
Per-image, per-stage timing derived from filesystem mtimes of pipeline outputs.

For each scene directory under experimental_results/COCO/<split>/, locates a
known artifact for each stage and treats its mtime as that stage's completion
time for that scene. Per-stage time for scene N is then:

    artifact_mtime(N, stage)  -  artifact_mtime(N-prev, stage)

where N-prev is the scene processed just before N within the same stage
(determined by sorting scenes by mtime of that stage's artifact).

For the first scene processed in stage S, the prior reference is the LAST
scene's mtime in stage S-1 (i.e., when stage S started).

Caveats:
- This approximates compute time; it includes any I/O / GC pauses.
- Model load time at the start of a stage gets attributed to whichever
  scene was processed first in that stage.
- Stages with per-object loops (completion, elevation, reconstruction)
  use the LATEST per-object artifact mtime within the scene.

Usage (from src/):
    python tools/per_image_timing.py --split val
    python tools/per_image_timing.py --split val --out timing_per_image.csv
"""
import argparse
import csv
from pathlib import Path

# (stage_name, callable scene_dir -> artifact path or None) — order matches pipeline.
STAGES = [
    "depth",
    "enhance",
    "get_crops_enhanced",
    "completion",
    "elevation",
    "reconstruction",
    "whole",
]


def latest_in(glob_results):
    paths = [p for p in glob_results if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def stage_artifact(stage, scene_dir):
    """Return the path representing 'stage finished for this scene', or None."""
    if stage == "depth":
        p = scene_dir / "depth_map.npy"
        return p if p.exists() else None
    if stage == "enhance":
        p = scene_dir / "enhanced" / "input.png"
        return p if p.exists() else None
    if stage == "get_crops_enhanced":
        # last-written reproj crop OR bboxes.json (whichever later)
        return latest_in([
            *((scene_dir / "crops").glob("*_reproj.png") if (scene_dir / "crops").exists() else []),
            scene_dir / "bboxes.json",
        ])
    if stage == "completion":
        return latest_in(
            list((scene_dir / "crops").glob("*_rgba.png"))
            if (scene_dir / "crops").exists() else []
        )
    if stage == "elevation":
        return latest_in(
            list((scene_dir / "object_space").glob("*/estimated_elevation.npy"))
            if (scene_dir / "object_space").exists() else []
        )
    if stage == "reconstruction":
        return latest_in(
            [p for p in (scene_dir / "object_space").glob("*.glb")]
            if (scene_dir / "object_space").exists() else []
        )
    if stage == "whole":
        p = scene_dir / "3dbbox.json"
        return p if p.exists() else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--results-dir", default="../experimental_results/COCO")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results_dir) / args.split
    if not root.is_dir():
        raise SystemExit(f"No directory: {root}")

    scenes = sorted(d for d in root.iterdir() if d.is_dir())

    # For each stage, sort scenes by their artifact mtime (= processing order).
    stage_order = {}
    stage_mtimes = {}
    for stage in STAGES:
        rows = []
        for sc in scenes:
            art = stage_artifact(stage, sc)
            if art is not None:
                rows.append((sc.name, art.stat().st_mtime))
        rows.sort(key=lambda r: r[1])
        stage_order[stage] = [r[0] for r in rows]
        stage_mtimes[stage] = dict(rows)

    # Per-scene per-stage time.
    # Reference for the FIRST scene in stage S = mtime of LAST scene in stage S-1.
    rows = []
    for s_idx, stage in enumerate(STAGES):
        order = stage_order[stage]
        mtimes = stage_mtimes[stage]
        if not order:
            continue
        if s_idx == 0:
            # No prior stage; first scene gets time=NaN (we don't know stage start).
            ref = None
        else:
            prev_stage = STAGES[s_idx - 1]
            prev_order = stage_order.get(prev_stage, [])
            if prev_order:
                ref = stage_mtimes[prev_stage][prev_order[-1]]
            else:
                ref = None

        prev_mtime = ref
        for scene_name in order:
            t = mtimes[scene_name]
            elapsed = (t - prev_mtime) if prev_mtime is not None else None
            rows.append({
                "scene": scene_name,
                "stage": stage,
                "seconds": round(elapsed, 2) if elapsed is not None else "",
            })
            prev_mtime = t

    out_path = Path(args.out) if args.out else Path(args.results_dir) / f"{args.split}_timing_per_image.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "stage", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Pretty per-scene summary table.
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["scene"], {})[r["stage"]] = r["seconds"]
    by_scene = dict(sorted(by_scene.items()))

    header = ["scene"] + STAGES + ["TOTAL"]
    widths = [max(len(h), 9) for h in header]
    widths[0] = max(widths[0], max((len(s) for s in by_scene), default=10))

    print("\nPer-image timing (seconds):")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    totals = {s: [] for s in STAGES}
    for scene, m in by_scene.items():
        cells = [scene]
        scene_total = 0.0
        scene_total_known = True
        for s in STAGES:
            v = m.get(s, "")
            if isinstance(v, (int, float)):
                cells.append(f"{v:.1f}")
                totals[s].append(v)
                scene_total += v
            else:
                cells.append("-")
                scene_total_known = False
        cells.append(f"{scene_total:.1f}" if scene_total_known else "-")
        print("  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    # Per-stage averages
    print("\nPer-stage averages (seconds/image):")
    for s in STAGES:
        vals = totals[s]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {s:24s}  n={len(vals):2d}   avg {avg:8.2f}s   min {min(vals):8.2f}s   max {max(vals):8.2f}s")

    grand = sum(sum(v) for v in totals.values())
    n_with_data = sum(1 for v in totals.values() if v)
    if n_with_data == len(STAGES) and by_scene:
        per_image = grand / len(by_scene)
        print(f"\nEstimated end-to-end per image: ~{per_image:.0f}s ({per_image/60:.1f} min)")

    print(f"\nWrote raw rows -> {out_path}")


if __name__ == "__main__":
    main()
