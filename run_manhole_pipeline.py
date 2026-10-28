#!/usr/bin/env python3
"""
run_manhole_pipeline.py -- connects the five existing scripts for one KITTI raw sequence.


  1  batch_test_yolo.py                  detection                -> 01_yolo/detections.csv
  2  simple_ellipse_batch_rules.py       ellipse fit              -> 02_ellipse/ellipse_results.csv
  3  drop ellipses within --border px of the image edge, then group the ok ellipses by frame number (a gap of more than --gap frames starts a new group)
  4  tracks inside each group (centre_y must be GREATER than the track's last ellipse, and the centre
     must be within --max-jump x last major axis x frames-since-last-ellipse)
  5  growth check: keep the largest set of frames that grows on BOTH axes (tie-break: straight line
     through 1/major axis), at least --min-frames frames -> each surviving track is one manhole
  6  only with --frame-check perframe: camera_calibration_from_ellipse.py --radius 1 per frame, drop frames
     whose f / tilt / height is far from the median
  6b pitch-change check (only with --pitch-check, off by default): IMU pitch range over the frames used must be <= --max-pitch-change (default 0.75 deg)
  6c same-plane check (only with --same-plane, off by default): IMU pitch --plane-offset (default 5) frames after the last frame used (car over the
     manhole) vs the mean pitch of the frames used, must differ by <= --max-plane-diff (default 1.5 deg)
  7  camera_calibration_from_ellipse.py --radius 1 --joint. With --frame-check joint (default) a frame whose
     joint RMS is > --rms-factor x the median is not used to pick START/END
     A manhole is rejected if the joint fit's 1-sigma uncertainty is above --max-f-sd % of f (default 3),
     --max-h-sd % of the height (default 2) or --max-tilt-sd deg of tilt (default 0.2) or no valid closed form start.
     A manhole whose overall joint RMS (all frames together) is above --max-rms px is rejected (default 1, 0 = off).
  8  recover_radius_kitti.py --turn-correct [--sum-steps] --start S --end E -> radius (sum of steps, or the
     start-to-end radius when a flagged frame lies between START and END and --sum-steps is off)
  9  camera_calibration_from_ellipse.py --radius <recovered> --joint
 10  kitti_road_plane.py --frames <track frame numbers>     -> ground truth height / tilt / roll, and f
 11  accuracy per manhole and for the whole sequence -> results.xlsx

USAGE
    python run_manhole_pipeline.py --weights best.pt --images <drive>/image_00/data \
        --velodyne <drive>/velodyne_points --calib-dir <date folder> --oxts <drive>/oxts --output run1
    # re-use an existing detections.csv (skips YOLO only; the ellipse fit runs, e.g. with --edge-shift 1.0):
    python run_manhole_pipeline.py --detections-csv run3\\01_yolo\\detections.csv --edge-shift 1.0 --images ... \\
        --velodyne ... --calib-dir ... --oxts ... --output run5
    # re-use an existing ellipse_results.csv (skips stages 1-2, --weights not needed):
    python run_manhole_pipeline.py --ellipses-csv ellipse_results.csv --images ... --velodyne ... \
        --calib-dir ... --oxts ... --output run2
"""

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SCRIPTS = dict(yolo="batch_test_yolo.py", ellipse="simple_ellipse_batch_rules.py",
               calib="camera_calibration_from_ellipse.py", radius="recover_radius_kitti.py",
               road="kitti_road_plane.py")
NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
JOINT_MIN_TILT_DEG = 0.05           # lower tilt limit inside camera_calibration_from_ellipse.py's joint fit

LOG_PATH = None


# =====================================================================================
# helpers: logging, running the scripts, reading CSVs
# =====================================================================================
def log(msg=""):
    print(msg, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


def run(cmd, step_log=None):
    """Run one of the five scripts. Returns (exit code, printed output). The output is saved to step_log
    (if given) and to pipeline.log."""
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
    text = "$ " + " ".join(cmd) + "\n" + out + f"\n[exit code {p.returncode}]\n"
    if step_log:
        Path(step_log).write_text(text, encoding="utf-8")
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(out + f"[exit code {p.returncode}]\n")
    return p.returncode, out


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    """float or None for empty/invalid cells."""
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def frame_of(name):
    stem = Path(name).stem
    return int(stem) if stem.isdigit() else None


def image_size(images_dir):
    import cv2
    for p in sorted(Path(images_dir).iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(str(p))
            if img is not None:
                return img.shape[1], img.shape[0], p.name
    sys.exit(f"[error] no readable image found in {images_dir}")


# =====================================================================================
# stage 3 + 4: groups by frame number, tracks inside each group
# =====================================================================================
def load_ellipses(path, img_size=None, border=None, min_minor=0.0, max_fit_rms=0.0):
    """All rows of ellipse_results.csv. Returns (ok ellipses, failed rows). Accepts 'filename' or 'source_filename'."""
    rows = read_csv(path)
    ok, failed = [], []
    for i, r in enumerate(rows):
        name = r.get("filename") or r.get("source_filename") or ""
        fr = frame_of(name)
        status = r.get("ellipse_fit") or r.get("status") or ""
        rec = dict(filename=name, frame=fr, det=r.get("detection_index", i), status=status,
                   fail_reason=r.get("fail_reason", ""))
        if fr is None:
            rec["fail_reason"] = "file name is not a frame number"
            failed.append(rec)
            continue
        vals = [fnum(r.get(k)) for k in ("center_x", "center_y", "major_axis", "minor_axis", "angle")]
        if status != "ok" or any(v is None for v in vals):
            failed.append(rec)
            continue
        rec.update(cx=vals[0], cy=vals[1], major=vals[2], minor=vals[3], angle=vals[4],
                   rms_px=r.get("rms_px", ""))
        # fit-quality checks on the single ellipse (both off when 0): a very thin ellipse (small minor axis) and an
        # ellipse whose edge points sit far from it (high fitter RMS) are unreliable fits
        if min_minor > 0 and vals[3] < min_minor:
            rec["status"] = "minor_axis_too_small"
            rec["fail_reason"] = f"minor axis {vals[3]:.1f} px < {min_minor:g} px"
            failed.append(rec)
            continue
        fit_rms = fnum(r.get("rms_px"))
        if max_fit_rms > 0 and fit_rms is not None and fit_rms > max_fit_rms:
            rec["status"] = "fit_rms_too_high"
            rec["fail_reason"] = f"ellipse fitter RMS {fit_rms:.3f} px > {max_fit_rms:g} px"
            failed.append(rec)
            continue
        if img_size is not None and border is not None:
            # distance from the ellipse's bounding box to each image edge; a manhole cut off by the edge gives a
            # too-small / too-flat ellipse, so it is not used
            W, H = img_size
            a_, b_, t_ = vals[2] / 2, vals[3] / 2, math.radians(vals[4])
            hx, hy = math.hypot(a_ * math.cos(t_), b_ * math.sin(t_)), math.hypot(a_ * math.sin(t_), b_ * math.cos(t_))
            gaps = {"left": vals[0] - hx, "right": W - 1 - (vals[0] + hx),
                    "top": vals[1] - hy, "bottom": H - 1 - (vals[1] + hy)}
            side = min(gaps, key=gaps.get)
            if gaps[side] < border:
                rec["status"] = "near_image_border"
                rec["fail_reason"] = f"ellipse {gaps[side]:.1f} px from the {side} edge of the image (limit {border:g} px)"
                failed.append(rec)
                continue
        ok.append(rec)
    ok.sort(key=lambda e: (e["frame"], e["cx"]))
    return ok, failed


def group_by_frame(ok, gap):
    """Split the ok ellipses into groups: a jump of more than `gap` frame numbers starts a new group."""
    groups = []
    for e in ok:
        if groups and e["frame"] - groups[-1][-1]["frame"] <= gap:
            groups[-1].append(e)
        else:
            groups.append([e])
    return groups


def link_check(last, e, max_jump):
    """Position check of ellipse e against a track's last ellipse. Returns (passes, distance px, note)."""
    df = e["frame"] - last["frame"]
    d = math.hypot(e["cx"] - last["cx"], e["cy"] - last["cy"])
    limit = max_jump * last["major"] * df
    if e["cy"] <= last["cy"]:
        return False, d, f"center_y {e['cy']:.1f} not greater than {last['cy']:.1f} (frame {last['frame']})"
    if d > limit:
        return False, d, (f"centre jumped {d / (last['major'] * df):.2f} major axes per frame from frame "
                          f"{last['frame']} (limit {max_jump})")
    return True, d, ""


def build_tracks(group, max_jump, gap):
    """Walk forward through one group. Each frame's ellipses are linked to open tracks (closest pairs first,
    one ellipse per track per frame); the rest start new tracks. A track stays open while the frame gap to its
    last ellipse is <= gap."""
    tracks = []                                   # each: dict(items=[...], notes=[...])
    frames = sorted({e["frame"] for e in group})
    for f in frames:
        ells = [e for e in group if e["frame"] == f]
        open_t = [t for t in tracks if f - t["items"][-1]["frame"] <= gap]
        pairs, why = [], {}
        for ti, t in enumerate(open_t):
            for ei, e in enumerate(ells):
                ok_, d, note = link_check(t["items"][-1], e, max_jump)
                if ok_:
                    pairs.append((d, ti, ei))
                else:
                    why.setdefault(ei, []).append((d, f"track starting at frame {t['items'][0]['frame']}: {note}"))
        pairs.sort()
        used_t, used_e = set(), set()
        for d, ti, ei in pairs:
            if ti in used_t or ei in used_e:
                continue
            open_t[ti]["items"].append(ells[ei])
            used_t.add(ti)
            used_e.add(ei)
        for ei, e in enumerate(ells):
            if ei in used_e:
                continue
            note = ""
            if why.get(ei):
                note = "started a new track -- " + min(why[ei])[1]
            elif open_t:
                note = "started a new track -- the open track(s) already took a closer ellipse in this frame"
            tracks.append(dict(items=[e], start_note=note))
    return tracks


# =====================================================================================
# stage 5: growth check
# =====================================================================================
def grows(a, b):
    return b["major"] > a["major"] and b["minor"] > a["minor"]


def largest_growing_sets(items, cap=20000):
    """All largest sets of frames (as index lists) in which each frame is bigger on both axes than the one before."""
    n = len(items)
    L = [1] * n
    pred = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            if grows(items[j], items[i]):
                if L[j] + 1 > L[i]:
                    L[i], pred[i] = L[j] + 1, [j]
                elif L[j] + 1 == L[i]:
                    pred[i].append(j)
    best = max(L)
    out = []

    def back(i):
        if L[i] == 1:
            yield [i]
            return
        for j in pred[i]:
            for c in back(j):
                yield c + [i]

    for i in range(n):
        if L[i] == best:
            for c in back(i):
                out.append(c)
                if len(out) >= cap:
                    return out
    return out


def growth_check(items):
    """Returns (kept indices, {dropped index: reason})."""
    sets = largest_growing_sets(items)
    frames = np.array([e["frame"] for e in items], float)
    inv = np.array([1.0 / e["major"] for e in items])
    if len(items) >= 2 and len(set(frames)) >= 2:
        k, c = np.polyfit(frames, inv, 1)
        line = k * frames + c
        pred = np.where(line > 0, 1.0 / np.where(line > 0, line, 1), np.nan)
        resid = np.abs(np.array([e["major"] for e in items]) - pred)      # px from the 1/size trend
        resid = np.where(np.isfinite(resid), resid, 1e9)
    else:
        resid = np.zeros(len(items))
    scores = [float(np.sum(resid[s] ** 2)) for s in sets]
    order = np.argsort(scores)
    kept = sets[int(order[0])]
    tie = len(sets) > 1
    reasons = {}
    for i in range(len(items)):
        if i in kept:
            continue
        r = "not growing on both axes (dropped to keep the largest growing set of frames)"
        if tie:
            alt = sets[int(order[1])]
            alt_drop = [items[j]["frame"] for j in range(len(items)) if j not in alt]
            r += (f"; tie-break: {resid[i]:.1f} px from the 1/major-axis trend "
                  f"(alternative was to drop frame(s) {alt_drop}: "
                  f"{', '.join(f'{resid[j]:.1f}' for j in range(len(items)) if j not in alt)} px)")
        reasons[i] = r
    return kept, reasons


# =====================================================================================
# stages 6-10 for one manhole
# =====================================================================================
def write_calib_csv(path, items):
    fields = ["filename", "ellipse_fit", "center_x", "center_y", "major_axis", "minor_axis", "angle"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for e in items:
            w.writerow(dict(filename=e["filename"], ellipse_fit="ok", center_x=e["cx"], center_y=e["cy"],
                            major_axis=e["major"], minor_axis=e["minor"], angle=e["angle"]))


def calib_cmd(a, scr, csv_path, out_dir, radius, joint):
    cmd = [sys.executable, scr["calib"], "--ellipses", csv_path, "--image-width", a.img_w,
           "--image-height", a.img_h, "--radius", radius, "--output", out_dir]
    return cmd + (["--joint"] if joint else [])


def read_joint(out_dir):
    jr = read_csv(Path(out_dir) / "joint_results.csv")
    if not jr:
        return None, "joint_results.csv is empty"
    j = jr[0]
    if j.get("status") != "ok":
        return None, f"joint fit: {j.get('status')}"
    tilt = fnum(j.get("tilt_deg"))
    if tilt is not None and tilt <= JOINT_MIN_TILT_DEG * 1.001:
        return None, f"joint fit tilt at the {JOINT_MIN_TILT_DEG} deg lower limit (unreliable)"
    return j, ""


def perframe_median_check(m, a, scr, d, items, fail):
    """--frame-check perframe: closed-form calibration of each frame at R = 1, then drop frames whose f, tilt or
    height is far from the manhole's median. Returns the frames left, or None if the manhole fails."""
    fr = m["frames"]
    p6 = d / "ellipses_growth_kept.csv"
    write_calib_csv(p6, items)
    out6 = d / "calib_R1_perframe"
    code, _ = run(calib_cmd(a, scr, p6, out6, 1, False), d / "calib_R1_perframe.log")
    if code != 0 or not (out6 / "calibration_results.csv").exists():
        fail("6 per-frame calib R=1", f"calibration script exited with code {code}")
        return None
    res = {r["filename"]: r for r in read_csv(out6 / "calibration_results.csv")}
    good = []
    for e in items:
        r = res.get(e["filename"], {})
        rec = fr[e["frame"]]
        rec.update(r1_f=fnum(r.get("focal_length_px")), r1_tilt=fnum(r.get("tilt_deg")),
                   r1_h=fnum(r.get("camera_height")))
        if r.get("status") != "ok" or None in (rec["r1_f"], rec["r1_tilt"], rec["r1_h"]):
            rec["used"], rec["reason"] = "dropped", f"per-frame calib failed: {r.get('reason', 'no result')}"
        else:
            good.append(e)
    if good:
        mf = float(np.median([fr[e["frame"]]["r1_f"] for e in good]))
        mt = float(np.median([fr[e["frame"]]["r1_tilt"] for e in good]))
        mh = float(np.median([fr[e["frame"]]["r1_h"] for e in good]))
        kept = []
        for e in good:
            rec = fr[e["frame"]]
            why = []
            if abs(rec["r1_f"] - mf) > a.f_tol / 100 * abs(mf):
                why.append(f"f {100 * (rec['r1_f'] - mf) / abs(mf):+.1f}% from median")
            if abs(rec["r1_tilt"] - mt) > a.tilt_tol:
                why.append(f"tilt {rec['r1_tilt'] - mt:+.2f} deg from median")
            if abs(rec["r1_h"] - mh) > a.h_tol / 100 * abs(mh):
                why.append(f"height {100 * (rec['r1_h'] - mh) / abs(mh):+.1f}% from median")
            if why:
                rec["used"], rec["reason"] = "dropped", "median check: " + ", ".join(why)
            else:
                kept.append(e)
        good = kept
    if len(good) < 2:
        fail("6 median check", f"only {len(good)} frame(s) left after the per-frame check (need 2)")
        return None
    return good


NO_START = "no frame gave a valid closed-form starting point"


def joint_fallback(a, scr, csv_path, out_dir, radius):

    import importlib.util
    spec = importlib.util.spec_from_file_location("calib_fallback", scr["calib"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    orig = mod.calibrate_from_conic

    def abs_f(*args, **kw):
        q = dict(orig(*args, **kw))
        if q["focal_length_px"] < 0:
            q["focal_length_px"] = -q["focal_length_px"]
        return q

    mod.calibrate_from_conic = abs_f                   # only this in-memory copy of the module
    hw, hh = a.img_w / 2.0, a.img_h / 2.0               # principal point = image centre, as the script uses
    names, frames, conics = [], [], []
    for r in read_csv(csv_path):
        cx, cy = float(r["center_x"]), float(r["center_y"])
        major, minor, ang = float(r["major_axis"]), float(r["minor_axis"]), float(r["angle"])
        jx, jy = mod.points_from_ellipse(cx, cy, major, minor, ang, n=60)
        frames.append((jx - hw, jy - hh))
        conics.append(mod.ellipse_geom_to_conic(cx - hw, cy - hh, major / 2.0, minor / 2.0, np.radians(ang)))
        names.append(r["filename"])
    out_dir = Path(out_dir)
    jr = out_dir / "joint_results.csv"
    if jr.exists():
        jr.replace(out_dir / "joint_results_script.csv")
    try:
        J = mod.joint_fit(frames, conics, float(radius))
    except (ValueError, np.linalg.LinAlgError) as e:
        with open(jr, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["group", "n_frames", "status"])
            w.writeheader()
            w.writerow(dict(group="all", n_frames=len(names), status=f"failed (fallback |f| start): {e}"))
        return False, str(e)
    with open(jr, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "n_frames", "status", "with_roll", "focal_length_px",
                                           "sd_focal_length_px", "tilt_deg", "sd_tilt_deg", "roll_deg", "sd_roll_deg",
                                           "camera_height", "sd_camera_height", "rms_px"])
        w.writeheader()
        w.writerow({"group": "all", "n_frames": J["n_frames"], "status": "ok" if J["converged"] else "not converged",
                    "with_roll": False, "focal_length_px": round(J["focal_length_px"], 2),
                    "sd_focal_length_px": round(J["sd_focal_length_px"], 2),
                    "tilt_deg": round(np.degrees(J["tilt_rad"]), 4), "sd_tilt_deg": round(np.degrees(J["sd_tilt_rad"]), 4),
                    "roll_deg": 0.0, "sd_roll_deg": 0.0, "camera_height": round(J["height"], 5),
                    "sd_camera_height": round(J["sd_height"], 5), "rms_px": round(J["rms_px"], 3)})
    with open(out_dir / "joint_frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "filename", "circle_center_a", "circle_center_b",
                                           "forward_from_below_camera", "rms_px"])
        w.writeheader()
        for name, (ga, gb), rr in zip(names, J["ab"], J["frame_rms_px"]):
            w.writerow({"group": "all", "filename": name, "circle_center_a": round(float(ga), 4),
                        "circle_center_b": round(float(gb), 4),
                        "forward_from_below_camera": round(float(gb) + J["height"] / np.tan(J["tilt_rad"]), 4),
                        "rms_px": round(rr, 3)})
    return True, "fallback |f| start"


def joint_with_fallback(a, scr, m, csv_path, out_dir, radius, step_log):
    """Run the script's joint fit; if it had no valid starting point, use joint_fallback. Returns (row, why)."""
    code, _ = run(calib_cmd(a, scr, csv_path, out_dir, radius, True), step_log)
    if code != 0 or not (Path(out_dir) / "joint_results.csv").exists():
        return None, f"calibration script exited with code {code}"
    j, why = read_joint(out_dir)
    if j is None and NO_START in why:
        ok, msg = joint_fallback(a, scr, csv_path, out_dir, radius)
        m["start_method"] = "fallback |f| start"
        log(f"  [info] no valid closed-form start in any frame -> joint fit restarted with |f| ({'ok' if ok else msg})")
        if not ok:
            return None, f"fallback |f| start also failed: {msg}"
        j, why = read_joint(out_dir)
    return j, why


_OXTS_CACHE = {}


def oxts_reader(oxts):
    """(pitch_deg(frame), speed(frame), time(frame), n_frames) from the drive's OXTS folder (oxts or oxts/data)."""
    if oxts in _OXTS_CACHE:
        return _OXTS_CACHE[oxts]
    from datetime import datetime
    p = Path(oxts)
    data = p / "data" if (p / "data").is_dir() else p
    ts_path = next((c for c in (p / "timestamps.txt", data / "timestamps.txt", data.parent / "timestamps.txt")
                    if c.exists()), None)
    times = []
    if ts_path:
        for line in ts_path.read_text().splitlines():
            if line.strip():
                day, clock = line.strip().split(" ")
                hms, _, frac = clock.partition(".")
                times.append(datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S").timestamp()
                             + (float("0." + frac) if frac else 0.0))
    vals = {}

    def row(k):
        if k not in vals:
            f = data / f"{k:010d}.txt"
            vals[k] = [float(x) for x in f.read_text().split()] if f.exists() else None
        return vals[k]

    pitch = lambda k: math.degrees(row(k)[4]) if row(k) else None       # OXTS field 5: pitch (rad)
    speed = lambda k: row(k)[8] if row(k) else None                     # OXTS field 9: forward speed (m/s)
    t = lambda k: times[k] if k < len(times) else None
    _OXTS_CACHE[oxts] = (pitch, speed, t)
    return _OXTS_CACHE[oxts]


def pitch_change_check(m, a, frames):
    """IMU pitch-change check. The joint fit assumes one shared tilt for all frames, so the car's pitch should not
    change much while the manhole is being observed. Range (max - min) of the OXTS pitch over every frame from the
    first to the last frame used. Returns a fail reason, or '' if it passes / cannot be checked."""
    pitch, _, _ = oxts_reader(a.oxts)
    vals = [pitch(f) for f in range(frames[0], frames[-1] + 1)]
    if None in vals:
        m["pitch_change_note"] = "not checked (OXTS pitch missing)"
        return ""
    rng = float(max(vals) - min(vals))
    m.update(pitch_change=rng, pitch_change_note="checked")
    if rng > a.max_pitch_change:
        return (f"IMU pitch changed by {rng:.2f} deg over frames {frames[0]}-{frames[-1]} "
                f"(limit {a.max_pitch_change:g} deg): camera tilt not constant across the frames")
    return ""


def same_plane_check(m, a, frames):
    """IMU same-plane check. The car is taken to be over the manhole a fixed --plane-offset frames after the last
    frame used. Compare the IMU pitch there (mean of that frame and its neighbours) with the mean IMU pitch over the
    frames used. A big difference means the road between the car and the manhole is not one plane.
    Returns a fail reason, or '' if it passes / cannot be checked."""
    pitch, _, _ = oxts_reader(a.oxts)
    p_used = [pitch(f) for f in frames]
    k = frames[-1] + a.plane_offset
    p_on = [pitch(j) for j in (k - 1, k, k + 1) if pitch(j) is not None]
    if None in p_used or not p_on:
        m["plane_note"] = f"not checked (OXTS pitch missing for the frames used or around frame {k})"
        return ""
    diff = float(np.mean(p_on) - np.mean(p_used))
    m.update(plane_over_frame=k, plane_pitch_diff=diff, plane_note="checked")
    if abs(diff) > a.max_plane_diff:
        return (f"IMU pitch over the manhole (frame {k}, {a.plane_offset} frames after frame {frames[-1]}) differs by "
                f"{diff:+.2f} deg from the frames used (limit {a.max_plane_diff:g} deg): car and manhole not on one plane")
    return ""


def process_manhole(m, a, scr):
    """Runs stages 6-10 for manhole m (dict with 'items' = kept frames). Fills m in place."""
    d = a.out / f"manhole_{m['id']:02d}"
    d.mkdir(parents=True, exist_ok=True)
    m["folder"] = d.name
    fr = m["frames"]                      # frame -> per-frame record (already contains the ellipse)
    write_calib_csv(d / "ellipses_track.csv", m["track_items"])

    def fail(stage, reason):
        m["status"], m["fail_stage"], m["fail_reason"] = "failed", stage, reason
        log(f"  [fail] {stage}: {reason}")

    items = m["items"]

    # ---- stage 6 (only with --frame-check perframe): per-frame calibration R = 1, median check ----
    if a.frame_check == "perframe":
        joint_items = perframe_median_check(m, a, scr, d, items, fail)
        if joint_items is None:
            return
    else:
        joint_items = items
    m["used_frames"] = [e["frame"] for e in joint_items]

    # ---- stage 6b: IMU pitch-change check over the frames used (only with --pitch-check) ----
    if a.pitch_check:
        why = pitch_change_check(m, a, m["used_frames"])
        if why:
            fail("6b pitch change", why)
            return
    else:
        m["pitch_change_note"] = "not run (--pitch-check off)"

    # ---- stage 6c: IMU same-plane check (only with --same-plane) ----
    if a.same_plane:
        why = same_plane_check(m, a, m["used_frames"])
        if why:
            fail("6c same plane", why)
            return
    else:
        m["plane_note"] = "not run (--same-plane off)"

    # ---- stage 7: joint calibration R = 1 ----
    p7 = d / "ellipses_used.csv"
    write_calib_csv(p7, joint_items)
    out7 = d / "calib_R1_joint"
    m.setdefault("start_method", "script (closed form)")
    j1, why = joint_with_fallback(a, scr, m, p7, out7, 1, d / "calib_R1_joint.log")
    if j1 is None:
        fail("7 joint calib R=1", why)
        return
    m.update(r1_joint_f=fnum(j1["focal_length_px"]), r1_joint_tilt=fnum(j1["tilt_deg"]),
             r1_joint_h=fnum(j1["camera_height"]), r1_joint_rms=fnum(j1["rms_px"]),
             r1_joint_sd_f=fnum(j1.get("sd_focal_length_px")), r1_joint_sd_h=fnum(j1.get("sd_camera_height")),
             r1_joint_sd_tilt=fnum(j1.get("sd_tilt_deg")))
    # focal-length uncertainty check: f is only measurable when the ellipses rotate enough with their sideways
    # offset; a manhole almost straight ahead gives a huge 1-sigma uncertainty from the joint fit itself
    f1, sdf = m["r1_joint_f"], m["r1_joint_sd_f"]
    if f1 is None or sdf is None or f1 <= 0:
        fail("7 focal-length uncertainty", f"joint fit gave no usable f / sd (f={f1}, sd={sdf})")
        return
    m["r1_joint_sd_f_pct"] = 100 * sdf / f1
    if m["r1_joint_sd_f_pct"] > a.max_f_sd:
        fail("7 focal-length uncertainty", f"joint f = {f1:.1f} +/- {sdf:.1f} px ({m['r1_joint_sd_f_pct']:.1f}% "
                                          f"> {a.max_f_sd:g}%): f not measurable for this manhole")
        return
    # height and tilt uncertainty checks (joint fit 1-sigma). Height and its SD both scale with the radius, so
    # SD/height in % is the same at R = 1 as at the recovered radius; tilt does not depend on the radius.
    h1, sdh, sdt = m["r1_joint_h"], m["r1_joint_sd_h"], m["r1_joint_sd_tilt"]
    if h1 is None or sdh is None or sdt is None or h1 <= 0:
        fail("7 height/tilt uncertainty", f"joint fit gave no usable height / tilt sd (h={h1}, sd_h={sdh}, sd_tilt={sdt})")
        return
    m["r1_joint_sd_h_pct"] = 100 * sdh / h1
    if m["r1_joint_sd_h_pct"] > a.max_h_sd:
        fail("7 height uncertainty", f"joint height uncertainty {m['r1_joint_sd_h_pct']:.2f}% of height "
                                     f"> {a.max_h_sd:g}%")
        return
    if sdt > a.max_tilt_sd:
        fail("7 tilt uncertainty", f"joint tilt uncertainty {sdt:.3f} deg > {a.max_tilt_sd:g} deg")
        return
    # overall joint-fit RMS check: how well ONE shared camera explains all of this manhole's ellipses together
    rms1 = m["r1_joint_rms"]
    if a.max_rms > 0 and (rms1 is None or rms1 > a.max_rms):
        fail("7 joint RMS", f"joint fit RMS {rms1} px > {a.max_rms:g} px: the frames do not fit one shared camera well")
        return
    if a.frame_check == "joint":                  # per-frame closed form, for information only
        for r in read_csv(out7 / "calibration_results.csv"):
            rec = fr.get(frame_of(r["filename"]))
            if rec:
                rec.update(r1_f=fnum(r.get("focal_length_px")), r1_tilt=fnum(r.get("tilt_deg")),
                           r1_h=fnum(r.get("camera_height")))
    jrows = read_csv(out7 / "joint_frames.csv")   # same order as ellipses_used.csv
    for r in jrows:
        fr[frame_of(r["filename"])]["joint_r1_rms"] = fnum(r.get("rms_px"))

    # ---- choose START and END rows for the radius ----
    i1, i2, sum_steps = 0, len(jrows) - 1, True
    if a.frame_check == "joint":
        rms = np.array([fnum(r.get("rms_px")) or 0.0 for r in jrows])
        med = float(np.median(rms))
        m["median_joint_rms"] = med
        flagged = [i for i in range(len(jrows)) if rms[i] > a.rms_factor * med]
        for i in flagged:
            rec = fr[frame_of(jrows[i]["filename"])]
            rec["used"] = "joint fit only"
            rec["reason"] = (f"joint RMS {rms[i]:.3f} px = {rms[i] / med:.1f} x median ({med:.3f}) "
                             f"-> not used to pick start/end")
        good_idx = [i for i in range(len(jrows)) if i not in flagged]
        if len(good_idx) < 2:
            fail("7 joint RMS check", f"only {len(good_idx)} frame(s) with joint RMS <= {a.rms_factor} x median")
            return
        i1, i2 = good_idx[0], good_idx[-1]
        between = [frame_of(jrows[i]["filename"]) for i in flagged if i1 < i < i2]
        if between:
            sum_steps = False
            m["sum_steps_note"] = f"off: flagged frame(s) {between} lie between start and end"
            for f in between:
                fr[f]["reason"] += "; lies between start and end, so --sum-steps was turned off"
    m["flagged_frames"] = [f for f, rec in fr.items() if rec.get("used") == "joint fit only"]
    m["sum_steps"] = "on" if sum_steps else "off"
    m["start_frame"], m["end_frame"] = frame_of(jrows[i1]["filename"]), frame_of(jrows[i2]["filename"])

    # ---- stage 8: recover radius ----
    cmd = [sys.executable, scr["radius"], "--csv", out7 / "joint_frames.csv", "--oxts", a.oxts,
           "--start", i1 + 1, "--end", i2 + 1, "--turn-correct"] + (["--sum-steps"] if sum_steps else [])
    code, out = run(cmd, d / "recover_radius.log")
    if code != 0:
        last = [ln for ln in out.splitlines() if ln.strip()][-1:] or [""]
        fail("8 recover radius", f"script exited with code {code}: {last[0].strip()}")
        return
    ms = re.search(r"RADIUS \(sum of steps\)\s*:\s*" + NUM + r"\s*m", out)
    mm = re.search(r"RECOVERED RADIUS\s*:\s*" + NUM + r"\s*m", out)
    m["radius_start_end"] = float(mm.group(1)) if mm else None
    if sum_steps and ms:
        m["radius"], m["radius_method"] = float(ms.group(1)), "sum of steps (turn-corrected)"
    elif mm:
        m["radius"], m["radius_method"] = float(mm.group(1)), (
            "start-to-end (turn-corrected)" + ("" if not sum_steps else "; sum of steps not printed"))
    else:
        fail("8 recover radius", "could not read the radius from the script's output")
        return
    if not (m["radius"] > 0 and math.isfinite(m["radius"])):
        fail("8 recover radius", f"radius {m['radius']} is not positive")
        return

    # ---- stage 9: joint calibration with the recovered radius ----
    out9 = d / "calib_Rfinal_joint"
    j9, why = joint_with_fallback(a, scr, m, p7, out9, m["radius"], d / "calib_Rfinal_joint.log")
    if j9 is None:
        fail("9 joint calib R=radius", why)
        return
    m.update(f=fnum(j9["focal_length_px"]), sd_f=fnum(j9["sd_focal_length_px"]),
             tilt=fnum(j9["tilt_deg"]), sd_tilt=fnum(j9["sd_tilt_deg"]),
             h=fnum(j9["camera_height"]), sd_h=fnum(j9["sd_camera_height"]), rms=fnum(j9["rms_px"]))
    for r in read_csv(out9 / "joint_frames.csv"):
        rec = fr.get(frame_of(r["filename"]))
        if rec:
            rec.update(joint_a=fnum(r.get("circle_center_a")), joint_b=fnum(r.get("circle_center_b")),
                       joint_fwd=fnum(r.get("forward_from_below_camera")), joint_rms=fnum(r.get("rms_px")))
    for r in read_csv(out9 / "calibration_results.csv"):
        rec = fr.get(frame_of(r["filename"]))
        if rec:
            rec.update(fin_f=fnum(r.get("focal_length_px")), fin_tilt=fnum(r.get("tilt_deg")),
                       fin_h=fnum(r.get("camera_height")))

    # ---- stage 10: ground truth from the road plane ----
    all_frames = sorted(fr)
    code, out = run([sys.executable, scr["road"], "--velodyne", a.velodyne, "--calib-dir", a.calib_dir,
                     "--camera", a.camera, "--frames", *all_frames, "--output", d / "road_plane.csv"],
                    d / "road_plane.log")
    mf_ = re.search(r"focal length\s+" + NUM + r"\s*px", out)
    m["gt_f"] = float(mf_.group(1)) if mf_ else None
    if code != 0 or not (d / "road_plane.csv").exists():
        fail("10 road plane", f"road-plane script exited with code {code} (no frame could be fitted?)")
        return
    for r in read_csv(d / "road_plane.csv"):
        rec = fr.get(int(r["frame"]))
        if rec:
            rec.update(gt_h=fnum(r["camera_height_m"]), gt_tilt=fnum(r["tilt_deg"]), gt_roll=fnum(r["roll_deg"]))
    gt_used = [fr[f] for f in m["used_frames"] if fr[f].get("gt_h") is not None]
    if not gt_used:
        fail("10 road plane", "no ground truth for any of the frames used")
        return
    m["gt_frames"] = len(gt_used)
    m["gt_h"] = float(np.mean([r["gt_h"] for r in gt_used]))
    m["gt_tilt"] = float(np.mean([r["gt_tilt"] for r in gt_used]))
    m["gt_roll"] = float(np.mean([r["gt_roll"] for r in gt_used]))

    # ---- stage 11: accuracy ----
    m["h_err_pct"] = 100 * (m["h"] - m["gt_h"]) / m["gt_h"]
    m["h_acc"] = 100 - abs(m["h_err_pct"])
    m["tilt_err"] = m["tilt"] - m["gt_tilt"]
    m["tilt_abs_err"] = abs(m["tilt_err"])
    if m["gt_f"]:
        m["f_err_pct"] = 100 * (m["f"] - m["gt_f"]) / m["gt_f"]
        m["f_acc"] = 100 - abs(m["f_err_pct"])
    m["status"] = "passed"
    log(f"  [ok] radius {m['radius']:.4f} m | height {m['h']:.4f} m (GT {m['gt_h']:.4f}, acc {m['h_acc']:.1f}%) | "
        f"tilt {m['tilt']:.3f} deg (GT {m['gt_tilt']:.3f}, err {m['tilt_err']:+.3f}) | f {m['f']:.1f} px"
        + (f" (GT {m['gt_f']:.1f}, acc {m['f_acc']:.1f}%)" if m.get("f_acc") is not None else ""))


# =====================================================================================
# Excel
# =====================================================================================
def r_(v, n=4):
    return round(v, n) if isinstance(v, float) else v


def write_excel(path, a, summary, manholes, rejected):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    bold = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="D9E1F2")
    green = PatternFill("solid", fgColor="E2EFDA")
    red = PatternFill("solid", fgColor="FCE4E4")
    grey = PatternFill("solid", fgColor="EDEDED")
    yellow = PatternFill("solid", fgColor="FFF2CC")

    def sheet(ws, headers, rows, fill_fn=None):
        ws.append(headers)
        for c in ws[1]:
            c.font, c.fill = bold, head_fill
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for row in rows:
            if row is None:                         # blank separator row
                ws.append([])
                continue
            ws.append([r_(row.get(h)) if isinstance(row, dict) else None for h in headers])
            if fill_fn:
                f = fill_fn(row)
                if f:
                    for c in ws[ws.max_row]:
                        c.fill = f
        ws.freeze_panes = "A2"
        for i, h in enumerate(headers, 1):
            width = max([len(str(h))] + [len(str(r_(r.get(h)))) for r in rows if isinstance(r, dict)])
            ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 70)

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Sequence result (manholes that passed every stage)"])
    ws["A1"].font = Font(bold=True, size=13)
    for k, v in summary:
        ws.append([k, r_(v, 3)])
        if k.startswith(("Mean height", "Mean focal", "Mean absolute tilt", "Median height", "Median focal",
                          "Median absolute tilt")):
            ws.cell(ws.max_row, 1).font = bold
            ws.cell(ws.max_row, 2).font = bold
    ws.column_dimensions["A"].width = 52
    ws.column_dimensions["B"].width = 60

    mh_cols = ["manhole", "status", "fail_stage", "fail_reason", "start_method", "folder", "track_frames", "growth_dropped",
               "median_dropped", "used_frames", "flagged_frames", "start_frame", "end_frame",
               "sum_steps", "sum_steps_note", "pitch_change", "pitch_change_note", "plane_pitch_diff", "plane_over_frame", "plane_note", "radius", "radius_method", "radius_start_end", "median_joint_rms",
               "r1_joint_f", "r1_joint_sd_f", "r1_joint_sd_f_pct", "r1_joint_sd_h_pct", "r1_joint_sd_tilt", "r1_joint_tilt", "r1_joint_h", "r1_joint_rms",
               "f", "sd_f", "tilt", "sd_tilt", "h", "sd_h", "rms",
               "gt_f", "gt_h", "gt_tilt", "gt_roll", "gt_frames",
               "h_err_pct", "h_acc", "f_err_pct", "f_acc", "tilt_err", "tilt_abs_err"]
    rows = []
    for m in manholes:
        r = {k: m.get(k) for k in mh_cols}
        r["manhole"] = f"Manhole {m['id']}"
        for k in ("track_frames", "used_frames", "flagged_frames"):
            r[k] = ", ".join(map(str, m.get(k) or []))
        r["growth_dropped"] = ", ".join(str(f) for f, rec in m["frames"].items() if rec.get("stage") == "growth")
        r["median_dropped"] = ", ".join(str(f) for f, rec in m["frames"].items()
                                        if rec.get("used") == "dropped" and rec.get("stage") != "growth")
        rows.append(r)
    sheet(wb.create_sheet("Manholes"), mh_cols, rows,
          lambda r: green if r.get("status") == "passed" else red)

    fr_cols = ["manhole", "frame", "filename", "detection_index", "used", "reason",
               "center_x", "center_y", "major_axis", "minor_axis", "angle",
               "r1_f", "r1_tilt", "r1_h", "joint_r1_rms", "fin_f", "fin_tilt", "fin_h",
               "joint_a", "joint_b", "joint_fwd", "joint_rms", "gt_h", "gt_tilt", "gt_roll"]
    rows = []
    for m in manholes:
        if rows:
            rows.append(None)
        for f in sorted(m["frames"]):
            rec = m["frames"][f]
            e = rec["e"]
            rows.append(dict(rec, manhole=f"Manhole {m['id']}", frame=f, filename=e["filename"],
                             detection_index=e["det"], center_x=e["cx"], center_y=e["cy"],
                             major_axis=e["major"], minor_axis=e["minor"], angle=e["angle"]))
    sheet(wb.create_sheet("Frames"), fr_cols, rows,
          lambda r: green if r.get("used") == "used" else (yellow if r.get("used") == "joint fit only" else grey))

    sheet(wb.create_sheet("Rejected"), ["kind", "frames", "reason"], rejected, lambda r: red)
    wb.save(path)


# =====================================================================================
# main
# =====================================================================================
def main():
    global LOG_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="YOLO .pt file (not needed with --detections-csv or --ellipses-csv)")
    ap.add_argument("--images", required=True, help="folder of KITTI frames (file names = frame numbers)")
    ap.add_argument("--velodyne", required=True, help="the drive's velodyne_points folder")
    ap.add_argument("--calib-dir", required=True, help="date folder with calib_velo_to_cam.txt and calib_cam_to_cam.txt")
    ap.add_argument("--oxts", required=True, help="the drive's oxts folder")
    ap.add_argument("--output", required=True, help="results folder")
    ap.add_argument("--detections-csv", default=None,
                    help="OPTIONAL existing detections.csv from batch_test_yolo.py: skips YOLO only (stage 1), the "
                         "ellipse fit still runs (e.g. to try another --edge-shift)")
    ap.add_argument("--ellipses-csv", default=None,
                    help="OPTIONAL existing ellipse_results.csv: skips YOLO and the ellipse fit")
    ap.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parent),
                    help="folder holding the five scripts. Default: this script's folder")
    ap.add_argument("--camera", type=int, default=0, help="KITTI camera for the road plane. Default 0")
    ap.add_argument("--confidence", type=float, default=0.6, help="YOLO confidence. Default 0.6")
    ap.add_argument("--edge-shift", type=float, default=0.0, help="ellipse fit --edge-shift. Default 0 (off)")
    ap.add_argument("--border", type=float, default=3.0,
                    help="drop ellipses closer than this many px to the image edge (manhole may be cut off). Default 3")
    ap.add_argument("--min-minor", type=float, default=0.0,
                    help="drop ellipses whose minor axis is below this many px (thin, unreliable fit). Default 0 (off)")
    ap.add_argument("--max-fit-rms", type=float, default=0.0,
                    help="drop ellipses whose ellipse-fitter RMS is above this many px. Default 0 (off)")
    ap.add_argument("--gap", type=int, default=5, help="a frame gap larger than this starts a new group. Default 5")
    ap.add_argument("--max-jump", type=float, default=2.0,
                    help="max centre move per frame, in major axes of the previous ellipse. Default 2")
    ap.add_argument("--min-frames", type=int, default=3, help="frames needed after the growth check. Default 3")
    ap.add_argument("--frame-check", choices=["joint", "perframe"], default="joint",
                    help="joint (default): joint fit at R=1 on all growth-kept frames, then frames whose joint RMS is "
                         "> --rms-factor x median are not used for start/end (sum-steps off if one lies between). "
                         "perframe: closed-form per frame at R=1, drop frames far from the median f/tilt/height, "
                         "then the joint fit on the rest")
    ap.add_argument("--max-f-sd", type=float, default=3.0,
                    help="reject a manhole whose joint-fit focal length uncertainty (1 sigma) is above this %% of f. "
                         "Default 3")
    ap.add_argument("--max-h-sd", type=float, default=2.0,
                    help="reject a manhole whose joint-fit camera height uncertainty (1 sigma) is above this %% of the "
                         "height. Default 2")
    ap.add_argument("--max-rms", type=float, default=1.0,
                    help="reject a manhole whose overall joint-fit RMS (all frames together, R = 1 fit) is above this "
                         "(px). 0 turns the check off. Default 1")
    ap.add_argument("--max-tilt-sd", type=float, default=0.2,
                    help="reject a manhole whose joint-fit tilt uncertainty (1 sigma) is above this (deg). Default 0.2")
    ap.add_argument("--pitch-check", action="store_true",
                    help="turn ON the IMU pitch-change check (off by default)")
    ap.add_argument("--max-pitch-change", type=float, default=0.75,
                    help="pitch-change check: reject a manhole if the IMU pitch range over its frames (first to last "
                         "used) is above this (deg). Default 0.75")
    ap.add_argument("--max-plane-diff", type=float, default=1.5,
                    help="same-plane check: reject a manhole if the IMU pitch when the car is over it differs from the "
                         "mean pitch of the frames used by more than this (deg). Default 1.5")
    ap.add_argument("--same-plane", action="store_true",
                    help="turn ON the IMU same-plane check (off by default)")
    ap.add_argument("--plane-offset", type=int, default=5,
                    help="same-plane check: the car is taken to be over the manhole this many frames after the last "
                         "frame used. Default 5")
    ap.add_argument("--rms-factor", type=float, default=3.0,
                    help="--frame-check joint: flag a frame whose joint RMS is above this x the median. Default 3")
    ap.add_argument("--f-tol", type=float, default=15.0, help="median check: focal length tolerance %%. Default 15")
    ap.add_argument("--tilt-tol", type=float, default=2.0, help="median check: tilt tolerance (deg). Default 2")
    ap.add_argument("--h-tol", type=float, default=15.0, help="median check: height tolerance %%. Default 15")
    a = ap.parse_args()

    a.out = Path(a.output).resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    LOG_PATH = a.out / "pipeline.log"
    LOG_PATH.write_text("", encoding="utf-8")
    for k in ("images", "velodyne", "calib_dir", "oxts"):
        setattr(a, k, str(Path(getattr(a, k)).resolve()))
    scr = {k: Path(a.scripts_dir) / v for k, v in SCRIPTS.items()}
    missing = [str(p) for p in scr.values() if not p.exists()]
    if missing:
        sys.exit("[error] script(s) not found: " + ", ".join(missing) + "  (use --scripts-dir)")
    a.img_w, a.img_h, first = image_size(a.images)
    log(f"[info] image size {a.img_w} x {a.img_h} (from {first}); principal point = image centre")

    # ---- stages 1-2 ----
    if a.ellipses_csv and a.detections_csv:
        sys.exit("[error] give --ellipses-csv OR --detections-csv, not both")
    if a.ellipses_csv:
        ell_csv = Path(a.ellipses_csv).resolve()
        log(f"[info] using existing ellipse results: {ell_csv} (stages 1-2 skipped)")
    else:
        if a.detections_csv:
            det_csv = Path(a.detections_csv).resolve()
            if not det_csv.exists():
                sys.exit(f"[error] detections CSV not found: {det_csv}")
            log(f"[info] using existing detections: {det_csv} (stage 1 skipped)")
        else:
            if not a.weights:
                sys.exit("[error] --weights is required unless --detections-csv or --ellipses-csv is given")
            log("\n=== Stage 1: YOLO detection ===")
            code, _ = run([sys.executable, scr["yolo"], "--weights", a.weights, "--input", a.images,
                           "--output", a.out / "01_yolo", "--confidence", a.confidence])
            if code != 0:
                sys.exit(f"[error] batch_test_yolo.py failed (exit {code}); see pipeline.log")
            det_csv = a.out / "01_yolo" / "detections.csv"
        log("\n=== Stage 2: ellipse fit ===")
        code, _ = run([sys.executable, scr["ellipse"], "--images", a.images, "--detections", det_csv,
                       "--output", a.out / "02_ellipse", "--edge-shift", a.edge_shift])
        if code != 0:
            sys.exit(f"[error] simple_ellipse_batch_rules.py failed (exit {code}); see pipeline.log")
        ell_csv = a.out / "02_ellipse" / "ellipse_results.csv"

    ok, failed_rows = load_ellipses(ell_csv, (a.img_w, a.img_h), a.border, a.min_minor, a.max_fit_rms)
    log(f"\n[info] {len(ok)} ok ellipses, {len(failed_rows)} failed fits")

    # ---- stages 3-5 ----
    log("\n=== Stages 3-5: groups, tracks, growth check ===")
    groups = group_by_frame(ok, a.gap)
    rejected, manholes = [], []
    n_tracks = 0
    for g in groups:
        gtxt = f"{g[0]['frame']}-{g[-1]['frame']}"
        for t in build_tracks(g, a.max_jump, a.gap):
            n_tracks += 1
            items = t["items"]
            ftxt = ", ".join(str(e["frame"]) for e in items)
            kept, reasons = growth_check(items)
            if len(kept) < a.min_frames:
                why = (f"only {len(kept)} growing frame(s) (need {a.min_frames})"
                       + (f" out of {len(items)}" if len(items) > len(kept) else ""))
                if t.get("start_note"):
                    why += f"; {t['start_note']}"
                rejected.append(dict(kind=f"track in group {gtxt}", frames=ftxt, reason=why))
                log(f"  group {gtxt}: track [{ftxt}] rejected -- {why}")
                continue
            m = dict(id=len(manholes) + 1, track_items=items, items=[items[i] for i in kept],
                     track_frames=[e["frame"] for e in items], status="", start_note=t.get("start_note", ""))
            m["frames"] = {}
            for i, e in enumerate(items):
                rec = dict(e=e, used="used", reason="")
                if i in reasons:
                    rec.update(used="dropped", reason=reasons[i], stage="growth")
                m["frames"][e["frame"]] = rec
            if t.get("start_note") and m["frames"][items[0]["frame"]]["reason"] == "":
                m["frames"][items[0]["frame"]]["reason"] = t["start_note"]
            manholes.append(m)
            log(f"  group {gtxt}: Manhole {m['id']} = frames [{ftxt}]"
                + (f", growth dropped {[items[i]['frame'] for i in reasons]}" if reasons else ""))
    spans = [(g[0]["frame"], g[-1]["frame"]) for g in groups]
    for r in failed_rows:
        inside = next((f"{s}-{e}" for s, e in spans if r["frame"] is not None and s <= r["frame"] <= e), None)
        kind = "ellipse near image border" if r["status"] == "near_image_border" else "ellipse fit failed"
        rejected.append(dict(kind=kind + (f" (inside group {inside})" if inside else ""),
                             frames=r["frame"] if r["frame"] is not None else r["filename"],
                             reason=f"{r['status']}: {r['fail_reason']}".strip(": ")))

    # ---- stages 6-11 ----
    for m in manholes:
        log(f"\n=== Manhole {m['id']}: frames {m['track_frames'][0]}-{m['track_frames'][-1]} ===")
        try:
            process_manhole(m, a, scr)
        except Exception as ex:                         # keep going with the next manhole
            m.update(status="failed", fail_stage="pipeline error", fail_reason=repr(ex))
            log(f"  [fail] unexpected error: {ex!r}")
        if m["status"] != "passed":                     # frames still marked 'used' did not give a result
            for rec in m["frames"].values():
                if rec["used"] == "used":
                    rec["used"] = "not used (manhole failed)"
                    rec["reason"] = rec["reason"] or f"manhole failed at stage {m['fail_stage']}"

    # ---- summary ----
    passed = [m for m in manholes if m["status"] == "passed"]
    f_ok = [m for m in passed if m.get("f_acc") is not None]
    summary = [
        ("Mean height accuracy (%)", float(np.mean([m["h_acc"] for m in passed])) if passed else "n/a"),
        ("Median height accuracy (%)", float(np.median([m["h_acc"] for m in passed])) if passed else "n/a"),
        ("Mean focal length accuracy (%)", float(np.mean([m["f_acc"] for m in f_ok])) if f_ok else "n/a"),
        ("Median focal length accuracy (%)", float(np.median([m["f_acc"] for m in f_ok])) if f_ok else "n/a"),
        ("Mean absolute tilt error (deg)", float(np.mean([m["tilt_abs_err"] for m in passed])) if passed else "n/a"),
        ("Median absolute tilt error (deg)", float(np.median([m["tilt_abs_err"] for m in passed])) if passed else "n/a"),
        ("Mean signed tilt error (deg)", float(np.mean([m["tilt_err"] for m in passed])) if passed else "n/a"),
        ("Median signed tilt error (deg)", float(np.median([m["tilt_err"] for m in passed])) if passed else "n/a"),
        ("Manholes passed (used in the means/medians)", ", ".join(str(m["id"]) for m in passed) or "none"),
        ("", ""),
        ("Split by how the joint fit was started", ""),
    ]
    for label, grp in [("script closed-form start", [m for m in passed if m.get("start_method") != "fallback |f| start"]),
                       ("fallback |f| start", [m for m in passed if m.get("start_method") == "fallback |f| start"])]:
        gf = [m for m in grp if m.get("f_acc") is not None]
        summary += [
            (f"  {label}: manholes", ", ".join(str(m["id"]) for m in grp) or "none"),
            (f"  {label}: mean height accuracy (%)", float(np.mean([m["h_acc"] for m in grp])) if grp else "n/a"),
            (f"  {label}: mean focal length accuracy (%)", float(np.mean([m["f_acc"] for m in gf])) if gf else "n/a"),
            (f"  {label}: median height accuracy (%)", float(np.median([m["h_acc"] for m in grp])) if grp else "n/a"),
            (f"  {label}: median focal length accuracy (%)", float(np.median([m["f_acc"] for m in gf])) if gf else "n/a"),
            (f"  {label}: mean absolute tilt error (deg)", float(np.mean([m["tilt_abs_err"] for m in grp])) if grp else "n/a"),
            (f"  {label}: median absolute tilt error (deg)", float(np.median([m["tilt_abs_err"] for m in grp])) if grp else "n/a"),
        ]
    summary += [
        ("", ""),
        ("Ok ellipses", len(ok)),
        ("Failed ellipse fits", len(failed_rows)),
        ("Frame groups", len(groups)),
        ("Tracks", n_tracks),
        ("Manholes (passed growth check)", len(manholes)),
        ("Manholes passed all stages", len(passed)),
        ("Manholes failed after the growth check", len(manholes) - len(passed)),
        ("", ""),
        ("Settings", ""),
        ("image size", f"{a.img_w} x {a.img_h} (principal point = image centre)"),
        ("ellipse results", str(ell_csv)),
        ("camera (road plane)", a.camera),
        ("image border margin (px)", a.border),
        ("min ellipse minor axis (px)", a.min_minor if a.min_minor > 0 else "off"),
        ("max ellipse fitter RMS (px)", a.max_fit_rms if a.max_fit_rms > 0 else "off"),
        ("gap between groups (frames)", a.gap),
        ("max centre jump (major axes per frame)", a.max_jump),
        ("min frames after growth check", a.min_frames),
        ("max focal length uncertainty (% of f)", a.max_f_sd),
        ("max camera height uncertainty (% of height)", a.max_h_sd),
        ("max tilt uncertainty (deg)", a.max_tilt_sd),
        ("max overall joint RMS (px, 0 = off)", a.max_rms),
        ("pitch-change check: max IMU pitch range over the frames used (deg)", a.max_pitch_change),
        ("pitch-change check", "on" if a.pitch_check else "off"),
        ("same-plane check", "on" if a.same_plane else "off"),
        ("same-plane check: max IMU pitch difference (deg)", a.max_plane_diff),
        ("same-plane check: frames after the last frame used (car over manhole)", a.plane_offset),
        ("frame check", a.frame_check),
        ("joint RMS factor (frame check joint)", a.rms_factor),
        ("median check f / tilt / height (frame check perframe)", f"{a.f_tol}% / {a.tilt_tol} deg / {a.h_tol}%"),
        ("recover radius", "--turn-correct, --sum-steps unless a flagged frame lies between start and end"),
    ]
    xlsx = a.out / "results.xlsx"
    write_excel(xlsx, a, summary, manholes, rejected)
    log("\n=== Sequence result ===")
    for k, v in summary[:9]:
        log(f"  {k}: {r_(v, 3)}")
    log(f"\nwritten: {xlsx}")


if __name__ == "__main__":
    main()
