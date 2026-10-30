#!/usr/bin/env python3
"""
Recover the true manhole radius (in metres) for KITTI, the same way recover_radius.py did for
the Blender simulations -- except that the distance the car drove comes from the KITTI OXTS
(GPS/IMU) speed instead of a speed you type in.
-------------------------
Run camera_calibration_from_ellipse.py with --radius 1 and --joint (or --joint-roll). Positions
then come out in "radius units": a single image cannot tell size from distance. If you know how
far the car really drove between two frames, you can work out what one radius unit is worth:

    radius (m) = distance_driven_m / movement_in_radii

    movement_in_radii = sqrt((a2 - a1)^2 + (b2 - b1)^2)      (circle_center_a / circle_center_b)
    distance_driven_m = sum over the frames of  speed x time between frames   (OXTS forward speed)

INPUTS
------
--csv    joint_frames.csv written by camera_calibration_from_ellipse.py --joint / --joint-roll
         (run with --radius 1). 
--oxts   the drive's oxts folder (data/0000000xxx.txt + timestamps.txt). The folder 'oxts' or
         'oxts/data' both work.

USAGE
-----
    python recover_radius_kitti.py --csv calib_out\\joint_frames.csv --oxts <drive>\\oxts

OUTPUT
------
A printed summary, and <csv name>_metric.csv next to the input: every row with its a / b /
forward position in metres, and the camera height in metres.

This is the simple two-frame method. It assumes the car drives straight between the two frames
(the heading change is printed so you can see it).

--turn-correct  
--sum-steps      
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

OXTS_YAW = 5   # yaw (rad)
OXTS_VF = 8    # forward velocity (m/s)


# --------------------------------------------------------------------------
# reading the CSV
# --------------------------------------------------------------------------

def read_rows(csv_path):
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"[error] {csv_path} has no data rows.")
    for col in ("filename", "circle_center_a", "circle_center_b"):
        if col not in rows[0]:
            sys.exit(f"[error] {csv_path} has no '{col}' column -- is this the joint_frames.csv written by "
                     f"camera_calibration_from_ellipse.py --joint, run with --radius 1?")
    return rows


def frame_number(filename):
    stem = Path(filename).stem
    if not stem.isdigit():
        sys.exit(f"[error] '{filename}': the file name must be the KITTI frame number (e.g. 0000000028.png) "
                 f"so the matching OXTS file can be found.")
    return int(stem)


def prompt_row(rows, label, default):
    while True:
        raw = input(f"\n{label} row number (1-{len(rows)}) [Enter = {default}]: ").strip()
        if raw == "":
            return default
        try:
            idx = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if 1 <= idx <= len(rows):
            return idx
        print(f"  Must be between 1 and {len(rows)}.")


# --------------------------------------------------------------------------
# reading OXTS
# --------------------------------------------------------------------------

def find_oxts(path):
    """Returns (folder with the numbered .txt files, timestamps.txt or None). 'oxts' or 'oxts/data' both work."""
    p = Path(path)
    data = p / "data" if (p / "data").is_dir() else p
    for cand in (p / "timestamps.txt", data / "timestamps.txt", data.parent / "timestamps.txt"):
        if cand.exists():
            return data, cand
    return data, None


def read_timestamps(path):
    """KITTI timestamps look like '2011-10-03 12:55:34.123456789' (nanoseconds). Returns seconds."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        day, clock = line.split(" ")
        hms, _, frac = clock.partition(".")
        base = datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S")
        out.append(base.timestamp() + (float("0." + frac) if frac else 0.0))
    return np.array(out)


def read_oxts_frames(data_dir, first, last):
    """Forward speed (m/s) and yaw (rad) for frames first..last inclusive."""
    vf, yaw = [], []
    for k in range(first, last + 1):
        f = Path(data_dir) / f"{k:010d}.txt"
        if not f.exists():
            sys.exit(f"[error] OXTS file not found: {f}")
        vals = [float(x) for x in f.read_text().split()]
        vf.append(vals[OXTS_VF])
        yaw.append(vals[OXTS_YAW])
    return np.array(vf), np.array(yaw)


# --------------------------------------------------------------------------
# turning correction (--turn-correct)
# --------------------------------------------------------------------------

def read_joint_height_tilt(csv_path, group):
    """(camera height in CSV units, tilt in degrees) from joint_results.csv next to the CSV, or (None, None)."""
    jr = Path(csv_path).parent / "joint_results.csv"
    if not jr.exists():
        return None, None, jr
    jrows = list(csv.DictReader(open(jr, newline="")))
    match = [j for j in jrows if group is None or j.get("group") == group] or jrows
    if not match:
        return None, None, jr
    h = float(match[0]["camera_height"]) if match[0].get("camera_height", "") != "" else None
    tilt = float(match[0]["tilt_deg"]) if match[0].get("tilt_deg", "") != "" else None
    return h, tilt, jr


def read_oxts_yaw(data_dir, frames):
    """OXTS heading (rad, 0 = east, counter-clockwise = left) for each frame number in `frames`."""
    out = {}
    for k in frames:
        f = Path(data_dir) / f"{k:010d}.txt"
        if not f.exists():
            sys.exit(f"[error] OXTS file not found: {f}")
        out[k] = float(f.read_text().split()[OXTS_YAW])
    return out


def rotate_all_frames(rows, f_start, data_dir, height_units, tilt_deg):
    """
    Rotate every CSV row's view of the circle so that all frames share the START frame's directions.
    Returns a list (sorted by frame number) of dicts: frame, filename, a, F, turn_deg, ahead, left
    (a, F, ahead, left in radius units; turn = heading minus the start frame's heading, left = positive).
    """
    items = []
    for r in rows:
        f = frame_number(r["filename"])
        a = float(r["circle_center_a"])                           # to the LEFT of the camera
        fw = r.get("forward_from_below_camera", "")
        if fw not in ("", None):
            F = float(fw)                                         # ahead of the point below the camera
        elif height_units is not None and tilt_deg is not None:
            F = float(r["circle_center_b"]) + height_units / np.tan(np.radians(tilt_deg))
        else:
            sys.exit("[error] --turn-correct needs the 'forward_from_below_camera' column in the CSV, or "
                     "joint_results.csv next to it (camera height and tilt) to work it out.")
        items.append((f, r["filename"], a, F))
    items.sort()
    yaw = read_oxts_yaw(data_dir, [it[0] for it in items] + [f_start])
    out = []
    for f, name, a, F in items:
        turn = (yaw[f] - yaw[f_start] + np.pi) % (2 * np.pi) - np.pi      # wrapped to -180..180 deg
        out.append(dict(frame=f, filename=name, a=a, F=F, turn_deg=float(np.degrees(turn)),
                        ahead=F * np.cos(turn) - a * np.sin(turn), left=F * np.sin(turn) + a * np.cos(turn)))
    return out


# --------------------------------------------------------------------------
# sum of the steps between consecutive frames (--sum-steps)
# --------------------------------------------------------------------------

def sum_of_steps(points, vf, dt, f_start):
    points = sorted(points)
    steps = []
    for (fa, xa, ya), (fb, xb, yb) in zip(points[:-1], points[1:]):
        radii = float(np.hypot(xb - xa, yb - ya))
        i0, i1 = fa - f_start, fb - f_start                       # indices into vf / dt
        metres = float(np.sum(0.5 * (vf[i0:i1] + vf[i0 + 1:i1 + 1]) * dt[i0:i1]))
        steps.append(dict(from_frame=fa, to_frame=fb, radii=radii, metres=metres,
                          radius_m=(metres / radii) if radii > 1e-9 else None))
    return steps


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="joint_frames.csv from camera_calibration_from_ellipse.py (run with --radius 1)")
    ap.add_argument("--oxts", required=True, help="the drive's oxts folder (oxts or oxts/data)")
    ap.add_argument("--fps", type=float, default=None,
                    help="OPTIONAL camera frames per second. If omitted, the real time between frames is read from timestamps.txt")
    ap.add_argument("--start", type=int, default=None, help="start row (1 = first row of the CSV); skips the prompt")
    ap.add_argument("--end", type=int, default=None, help="end row; skips the prompt")
    ap.add_argument("--radius-used", type=float, default=1.0,
                    help="the --radius you gave camera_calibration_from_ellipse.py. Default 1 (positions in radius units)")
    ap.add_argument("--true-height", type=float, default=None, help="OPTIONAL true camera height (m), only for the error printout")
    ap.add_argument("--turn-correct", action="store_true",
                    help="OPTIONAL: correct for the car turning between the frames, using the OXTS heading and speed")
    ap.add_argument("--sum-steps", action="store_true",
                    help="OPTIONAL: also work out the radius from the SUM of the circle's movement between consecutive "
                         "frames (follows the curve), printed next to the usual start-to-end result, which is not changed")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[error] not found: {csv_path}")
    rows = read_rows(csv_path)
    print(f"[info] loaded {len(rows)} row(s) from {csv_path}")
    print("[info] this only makes sense if the calibration was run with --radius 1 (positions in radius units).")

    print("\nAvailable rows:")
    for i, r in enumerate(rows, start=1):
        print(f"  {i}: {r['filename']}   a={r['circle_center_a']} b={r['circle_center_b']}")

    i1 = args.start if args.start is not None else prompt_row(rows, "START", 1)
    i2 = args.end if args.end is not None else prompt_row(rows, "END", len(rows))
    for i in (i1, i2):
        if not 1 <= i <= len(rows):
            sys.exit(f"[error] row {i} is outside 1-{len(rows)}.")
    if i1 == i2:
        sys.exit("[error] start and end rows are the same -- pick two different frames.")
    r1, r2 = rows[i1 - 1], rows[i2 - 1]
    f1, f2 = frame_number(r1["filename"]), frame_number(r2["filename"])
    if f1 > f2:                                          # the order of the two rows does not matter
        r1, r2, f1, f2, i1, i2 = r2, r1, f2, f1, i2, i1
    if f1 == f2:
        sys.exit("[error] the two rows are the same frame number.")
    a1, b1 = float(r1["circle_center_a"]), float(r1["circle_center_b"])
    a2, b2 = float(r2["circle_center_a"]), float(r2["circle_center_b"])

    # ---- the distance driven, from the OXTS speed ----
    data_dir, ts_path = find_oxts(args.oxts)
    if not data_dir.is_dir():
        sys.exit(f"[error] OXTS folder not found: {args.oxts}")
    vf, yaw = read_oxts_frames(data_dir, f1, f2)
    ts = read_timestamps(ts_path) if ts_path else None
    if ts is not None and len(ts) <= f2:
        sys.exit(f"[error] {ts_path} has {len(ts)} lines but frame {f2} was asked for -- is this the right drive?")
    if args.fps is not None:
        if args.fps <= 0:
            sys.exit("[error] --fps must be positive.")
        dt = np.full(f2 - f1, 1.0 / args.fps)
        time_txt = f"{1.0 / args.fps:.4f} s per frame from --fps {args.fps:g}"
        if ts is not None:
            real = float(np.mean(np.diff(ts[f1:f2 + 1])))
            time_txt += f"   (timestamps.txt says {real:.4f} s = {1.0 / real:.2f} fps over these frames)"
    elif ts is not None:
        dt = np.diff(ts[f1:f2 + 1])
        time_txt = f"{dt.mean():.4f} s per frame from timestamps.txt ({1.0 / dt.mean():.2f} fps)"
    else:
        sys.exit(f"[error] no timestamps.txt found next to {args.oxts} -- give the camera frame rate with --fps.")
    distance_m = float(np.sum(0.5 * (vf[:-1] + vf[1:]) * dt))
    heading_change = float(np.degrees(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0]))

    # ---- the movement in radii, and the radius ----
    da, db = a2 - a1, b2 - b1
    movement = float(np.hypot(da, db))
    print("\n" + "=" * 55)
    print("RESULT")
    print("=" * 55)
    print(f"Start row {i1} ({r1['filename']}): a={a1:.5f}  b={b1:.5f}")
    print(f"End row   {i2} ({r2['filename']}): a={a2:.5f}  b={b2:.5f}")
    print(f"\nChange in position: da={da:+.5f}  db={db:+.5f}")
    print(f"Movement in radii:   {movement:.5f}")
    print(f"\nSpeed from OXTS, frames {f1} to {f2} (m/s): " + "  ".join(f"{v:.2f}" for v in vf)
          + f"   (mean {vf.mean():.2f} m/s = {vf.mean() * 3.6:.1f} km/h)")
    print(f"Time between frames: {time_txt}")
    print(f"Distance driven, frames {f1} to {f2}: {distance_m:.4f} m")
    if args.turn_correct:
        print(f"[info] heading changed {heading_change:+.2f} deg over these frames (corrected with --turn-correct)")
    else:
        print(f"[info] heading changed {heading_change:+.2f} deg over these frames (this method assumes the car drives straight)")
    if movement < 1e-9:
        sys.exit("\n[error] movement in radii is ~0 -- the circle's position barely changed between these two rows, "
                 "so a radius cannot be recovered. Pick two frames further apart.")
    scale = distance_m / movement                                 # metres per unit of the CSV
    height_units, tilt_deg, jr = read_joint_height_tilt(csv_path, r1.get("group", None))
    if args.turn_correct:
        scale_simple = scale
        table = rotate_all_frames(rows, f1, data_dir, height_units, tilt_deg)
        print(f"\nTurning correction: every frame rotated into the directions of the start frame ({r1['filename']}):")
        print(f"  {'frame':>7s} {'a (left)':>10s} {'F (ahead)':>10s} {'turn vs start':>14s} {'ahead after':>12s} {'left after':>11s}")
        for t in table:
            mark = "   <- start" if t["filename"] == r1["filename"] else ("   <- end" if t["filename"] == r2["filename"] else "")
            print(f"  {t['frame']:>7d} {t['a']:>10.4f} {t['F']:>10.4f} {t['turn_deg']:>+11.2f} deg {t['ahead']:>12.4f} {t['left']:>11.4f}{mark}")
        ts_ = next(t for t in table if t["filename"] == r1["filename"])
        te_ = next(t for t in table if t["filename"] == r2["filename"])
        movement_rot = float(np.hypot(te_["ahead"] - ts_["ahead"], te_["left"] - ts_["left"]))
        if movement_rot < 1e-9:
            sys.exit("\n[error] after rotating, the circle's movement between the two rows is ~0. Pick two frames further apart.")
        print(f"Movement in radii after rotating: {movement_rot:.5f}   (change: ahead {te_['ahead'] - ts_['ahead']:+.5f}, "
              f"left {te_['left'] - ts_['left']:+.5f})")
        scale = distance_m / movement_rot
    radius = scale * args.radius_used
    print(f"\n{'RECOVERED RADIUS':>25}: {radius:.5f} m" + ("   (turning-corrected)" if args.turn_correct else ""))
    if args.turn_correct:
        diff = 100 * (scale_simple / scale - 1)
        print(f"{'':>25}  without the turning correction: {scale_simple * args.radius_used:.5f} m "
              f"({abs(diff):.1f}% {'higher' if diff > 0 else 'lower'})")
    if height_units is not None:
        height_m = height_units * scale
        print(f"{'CAMERA HEIGHT':>25}: {height_units:.4f} x {radius / args.radius_used:.5f} = {height_m:.4f} m   (from {jr.name})")
        if args.true_height:
            print(f"{'':>25}  error vs {args.true_height} m: {100 * (height_m - args.true_height) / args.true_height:+.1f}%")
    else:
        print("   (no joint_results.csv next to the CSV, so no camera height in metres)")
    print("=" * 55)

    if args.sum_steps:
        # positions of every CSV frame between START and END, in the same axes the main result used
        if args.turn_correct:
            pts = [(t["frame"], t["ahead"], t["left"]) for t in table if f1 <= t["frame"] <= f2]
            axes_txt = "after rotating into the start frame's directions"
        else:
            pts = [(frame_number(r["filename"]), float(r["circle_center_b"]), float(r["circle_center_a"]))
                   for r in rows if f1 <= frame_number(r["filename"]) <= f2]
            axes_txt = "from a and b, no turning correction"
        steps = sum_of_steps(pts, vf, dt, f1)
        print(f"\nSum of steps between consecutive frames, {f1} to {f2} ({len(pts)} frames in the CSV, {axes_txt}):")
        print(f"  {'from':>7s} {'to':>7s} {'step (radii)':>13s} {'driven (m)':>11s} {'radius from this step (m)':>26s}")
        for s in steps:
            per = f"{s['radius_m'] * args.radius_used:.5f}" if s["radius_m"] is not None else "n/a"
            print(f"  {s['from_frame']:>7d} {s['to_frame']:>7d} {s['radii']:>13.5f} {s['metres']:>11.4f} {per:>26s}")
        path_radii = sum(s["radii"] for s in steps)
        driven_m = sum(s["metres"] for s in steps)
        if len(pts) < 2 or path_radii < 1e-9:
            print("  [warn] not enough frames or no movement to add up steps -- skipped.")
        else:
            scale_sum = driven_m / path_radii
            radius_sum = scale_sum * args.radius_used
            diff = 100 * (scale_sum / scale - 1)
            print(f"  Path in radii (sum of steps): {path_radii:.5f}   (start-to-end line: "
                  f"{movement_rot if args.turn_correct else movement:.5f})")
            print(f"  {'RADIUS (sum of steps)':>25}: {radius_sum:.5f} m   ({diff:+.2f}% vs the start-to-end result above)")
            if height_units is not None:
                height_sum = height_units * scale_sum
                print(f"  {'CAMERA HEIGHT (sum)':>25}: {height_units:.4f} x {scale_sum:.5f} = {height_sum:.4f} m")
                if args.true_height:
                    print(f"  {'':>25}  error vs {args.true_height} m: "
                          f"{100 * (height_sum - args.true_height) / args.true_height:+.1f}%")
            if abs(driven_m - distance_m) > 1e-6:
                print(f"  [note] the steps add up to {driven_m:.4f} m, not {distance_m:.4f} m: some frames between "
                      f"START and END are not in the CSV, so the sum is over fewer, longer steps.")
            if len(pts) < f2 - f1 + 1:
                print(f"  [note] {f2 - f1 + 1 - len(pts)} frame(s) between START and END are not in the CSV.")

    # ---- the metric CSV ----
    out_path = csv_path.with_name(csv_path.stem + "_metric.csv")
    fields = list(rows[0].keys()) + ["circle_center_a_m", "circle_center_b_m"]
    has_fwd = "forward_from_below_camera" in rows[0]
    if has_fwd:
        fields.append("forward_m")
    if height_units is not None:
        fields.append("camera_height_m")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            o = dict(r)
            o["circle_center_a_m"] = round(float(r["circle_center_a"]) * scale, 4)
            o["circle_center_b_m"] = round(float(r["circle_center_b"]) * scale, 4)
            if has_fwd and r["forward_from_below_camera"] != "":
                o["forward_m"] = round(float(r["forward_from_below_camera"]) * scale, 4)
            if height_units is not None:
                o["camera_height_m"] = round(height_units * scale, 4)
            w.writerow(o)
    print(f"\nPer-frame positions in metres: {out_path}")


if __name__ == "__main__":
    main()
