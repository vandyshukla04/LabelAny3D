#!/usr/bin/env bash
# Run the LabelAny3D pipeline on the staged frames, with per-stage wall-clock timing.
#
# Assumes you've already run:
#   python tools/wildbox_to_coconut.py --paths-file ../<paths-file>
# from the src/ directory, so dataset/coco/{images,annotations}/ are populated.
#
# Args (optional): START END   (defaults: 0 10)
set -euo pipefail

START="${1:-0}"
END="${2:-10}"
N=$((END - START))

cd "$(dirname "$0")/src"

LOG_DIR=../experimental_results/COCO
mkdir -p "$LOG_DIR"
TIMING_CSV="$LOG_DIR/timing.csv"
echo "stage,seconds,start_index,end_index,n_images,sec_per_image" > "$TIMING_CSV"

run_total_start=$(date +%s)

for stage in depth enhance get_crops_enhanced completion elevation reconstruction whole; do
    echo "============================================================"
    echo "stage: $stage   (--start_index=$START --end_index=$END)"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    stage_start=$(date +%s)
    python batch_scripts/$stage.py --start_index="$START" --end_index="$END" --split=val
    stage_end=$(date +%s)
    elapsed=$((stage_end - stage_start))
    per_img=$(awk -v t="$elapsed" -v n="$N" 'BEGIN { printf "%.2f", t/n }')
    echo "[time] $stage: ${elapsed}s total, ${per_img}s/image"
    echo "$stage,$elapsed,$START,$END,$N,$per_img" >> "$TIMING_CSV"
done

echo "============================================================"
echo "combine_results"
echo "============================================================"
combine_start=$(date +%s)
python tools/combine_results.py --split=val
combine_end=$(date +%s)
echo "$combine_results,$((combine_end - combine_start)),$START,$END,$N," >> "$TIMING_CSV"

run_total_end=$(date +%s)
total=$((run_total_end - run_total_start))
total_per_img=$(awk -v t="$total" -v n="$N" 'BEGIN { printf "%.2f", t/n }')

echo
echo "============================================================"
echo "DONE."
echo "Total wall time: ${total}s ($((total/60))m $((total%60))s)"
echo "Per-image average: ${total_per_img}s (${N} images)"
echo "Per-stage timing: $TIMING_CSV"
echo "Outputs: ../experimental_results/COCO/val/<frame>/"
echo "Combined: ../experimental_results/COCO/COCO3D_val.json"
echo "============================================================"
