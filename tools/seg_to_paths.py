"""
Emit a paths file (one absolute frame path per line) for every frame in a
WildBox segment directory. Use the output as input to wildbox_to_coconut.py.

Usage:
    python tools/seg_to_paths.py /path/to/segN
    python tools/seg_to_paths.py /path/to/segN --out segN_paths.txt
    python tools/seg_to_paths.py /path/to/segN --max 20
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg_dir", help="Path to a seg<N>/ directory containing frame_*.jpg")
    ap.add_argument("--out", default=None,
                    help="Output paths file (default: <seg_name>_paths.txt next to script)")
    ap.add_argument("--max", type=int, default=0,
                    help="If >0, take only the first N frames (sorted)")
    ap.add_argument("--every", type=int, default=1,
                    help="Stride: keep every Nth frame (default 1 = all). "
                         "Example: --every 10 on a 110-frame segment yields ~11 frames.")
    args = ap.parse_args()

    seg = Path(args.seg_dir).resolve()
    if not seg.is_dir():
        raise SystemExit(f"Not a directory: {seg}")

    frames = sorted(seg.glob("frame_*.jpg"))
    if args.every > 1:
        frames = frames[::args.every]
    if args.max > 0:
        frames = frames[: args.max]

    out_path = Path(args.out) if args.out else Path(f"{seg.name}_paths.txt")
    with out_path.open("w") as f:
        f.write(f"# {len(frames)} frames from {seg}\n")
        for p in frames:
            f.write(f"{p}\n")
    print(f"Wrote {len(frames)} frame paths -> {out_path}")
    print(f"\nNext:")
    print(f"  cd src && python tools/wildbox_to_coconut.py --paths-file ../{out_path.name}")
    print(f"  cd .. && bash run_wildbox.sh 0 {len(frames)}")


if __name__ == "__main__":
    main()
