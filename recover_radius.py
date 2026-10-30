#!/usr/bin/env python3
"""
Recover the true radius (in metres) from a camera_calibration_from_ellipse.py output CSV run with --radius 1, using the known
distance the camera travelled between two rows/frames.

WHY THIS EXISTS
---------------
Run with --radius 1, circle_center_a/circle_center_b come out in "radius
units" -- an arbitrary scale, since a single image can't tell size from
distance. If you separately know how far the camera moved (real metres)
between two frames, that known distance and the corresponding change in
(a, b) let you solve for what one radius unit is actually worth in metres:

    metres per radius = true_distance_m / movement_in_radii

where movement_in_radii = sqrt((a2-a1)^2 + (b2-b1)^2), and true_distance_m
comes from speed and the time between the two frames:

    true_distance_m = speed_m_s * (frame2 - frame1) / fps

USAGE
-----
    python recover_radius.py --csv calibration_results.csv

Then answer the prompts: start row, end row, speed (km/h), fps.
"""

import argparse
import csv
import sys
from pathlib import Path


def read_rows(csv_path):
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"[error] {csv_path} has no data rows.")
    for col in ("circle_center_a", "circle_center_b"):
        if col not in rows[0]:
            sys.exit(f"[error] {csv_path} has no '{col}' column -- is this a "
                     f"camera_calibration_from_ellipse.py or calibrate_known_f.py "
                     f"output CSV, run with --radius 1?")
    return rows


def prompt_row_index(rows, label):
    print(f"\nAvailable rows:")
    for i, r in enumerate(rows, start=1):
        a, b = r.get("circle_center_a", ""), r.get("circle_center_b", "")
        status = r.get("status", r.get("note", ""))
        ok = a != "" and b != ""
        flag = "" if ok else "  <-- no usable a/b (failed row)"
        print(f"  {i}: {r.get('filename','?')}   a={a} b={b}{flag}")
    while True:
        raw = input(f"\n{label} row number (1-{len(rows)}): ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if not 1 <= idx <= len(rows):
            print(f"  Must be between 1 and {len(rows)}.")
            continue
        a, b = rows[idx - 1].get("circle_center_a", ""), rows[idx - 1].get("circle_center_b", "")
        if a == "" or b == "":
            print(f"  Row {idx} ({rows[idx-1].get('filename','?')}) has no circle_center_a/b "
                 f"-- that row's calibration failed, pick a different row.")
            continue
        return idx


def prompt_float(label, allow_zero=False):
    while True:
        raw = input(f"{label}: ").strip()
        try:
            val = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if val < 0 or (val == 0 and not allow_zero):
            print("  Must be a positive number.")
            continue
        return val


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="calibration_results.csv / calibration_known_f.csv, "
                         "produced with --radius 1")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[error] not found: {csv_path}")
    rows = read_rows(csv_path)

    print(f"[info] loaded {len(rows)} row(s) from {csv_path}")
    print("[info] this only makes sense if the CSV was generated with --radius 1 "
          "(a/b in 'radius units', not real metres) -- if you used the true radius "
          "instead, this script's output will be meaningless.")

    i1 = prompt_row_index(rows, "START")
    i2 = prompt_row_index(rows, "END")
    if i1 == i2:
        sys.exit("[error] start and end rows are the same -- pick two different frames.")

    r1, r2 = rows[i1 - 1], rows[i2 - 1]
    a1, b1 = float(r1["circle_center_a"]), float(r1["circle_center_b"])
    a2, b2 = float(r2["circle_center_a"]), float(r2["circle_center_b"])

    speed_kmh = prompt_float("\nSpeed (km/h)")
    fps = prompt_float("Frames per second")
    frame_gap = prompt_float("Number of frames between start and end row "
                             "(e.g. frame 91 - frame 1 = 90)")

    speed_ms = speed_kmh / 3.6
    time_s = frame_gap / fps
    true_distance_m = speed_ms * time_s

    da, db = a2 - a1, b2 - b1
    movement_in_radii = (da ** 2 + db ** 2) ** 0.5

    print("\n" + "=" * 55)
    print("RESULT")
    print("=" * 55)
    print(f"Start row {i1} ({r1['filename']}): a={a1:.5f}  b={b1:.5f}")
    print(f"End row   {i2} ({r2['filename']}): a={a2:.5f}  b={b2:.5f}")
    print(f"\nChange in position: da={da:+.5f}  db={db:+.5f}")
    print(f"Movement in radii:   {movement_in_radii:.5f}")
    print(f"\nSpeed: {speed_kmh} km/h = {speed_ms:.4f} m/s")
    print(f"Time between frames: {frame_gap:.0f} frames / {fps} fps = {time_s:.4f} s")
    print(f"True distance travelled: {true_distance_m:.4f} m")

    if movement_in_radii < 1e-9:
        print("\n[error] movement in radii is ~0 -- the circle's position barely changed "
             "between these two rows, so a radius can't be recovered from them. Pick two "
             "frames further apart, or check the rows are actually different frames.")
        sys.exit(1)

    recovered_radius = true_distance_m / movement_in_radii
    print(f"\n{'RECOVERED RADIUS':>25}: {recovered_radius:.5f} m")
    print("=" * 55)
    print("\nCompare this against your circle's TRUE known radius. Close agreement means "
         "the ellipse fit, the calibration equations, and this scale-recovery step are all "
         "working correctly together.")


if __name__ == "__main__":
    main()
