#!/usr/bin/env bash
# Run the LabelAny3D pipeline on the staged frames.
# Assumes you've already run:
#   python tools/wildbox_to_coconut.py --paths-file ../wildbox_paths.txt
# from the src/ directory.
#
# Args (optional): START END   (defaults: 0 10)
set -euo pipefail

START="${1:-0}"
END="${2:-10}"

cd "$(dirname "$0")/src"

for stage in depth enhance get_crops_enhanced completion elevation reconstruction whole; do
    echo "============================================================"
    echo "stage: $stage   (--start_index=$START --end_index=$END)"
    echo "============================================================"
    python batch_scripts/$stage.py --start_index="$START" --end_index="$END" --split=val
done

echo "============================================================"
echo "combining results into Omni3D JSON"
echo "============================================================"
python tools/combine_results.py --split=val

echo
echo "Done. Outputs:"
echo "  experimental_results/COCO/val/<frame>/3dbbox.json   per-frame 3D boxes"
echo "  experimental_results/COCO/val/<frame>/reconstruction/full_scene.glb"
echo "  experimental_results/COCO/COCO3D_val.json           combined Omni3D"
