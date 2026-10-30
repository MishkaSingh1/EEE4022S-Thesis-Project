#!/usr/bin/env python3
"""
Build a PAIRED test suite of synthetic KITTI-like manhole images: a CLEAN set, then one set per
feature (occlusion, shadow, noise, blur, motion blur, low contrast, rim ring, painted stripe, JPEG, distance).

Pairing: there are --per-set scenes (default 100), numbered 0..99. A scene fixes the camera tilt,
manhole position and size, road texture, cover tone/pattern, noise pattern and detector-box jitter.
Every set uses the SAME scenes, so scene k in "occlusion" is the clean scene k with only the
occluder added.
  --mode full  (default): every scene is rendered at EVERY level of an effect, so a feature set has
                scenes x levels images (5 levels -> 500) and levels can be compared on the same scenes.
  --mode split           : scene k gets only level (k mod levels), so a set has just --per-set images
                (20 scenes per level for 5 levels). Smaller, but each level uses different scenes.
Ground truth is exact (see generate_kitti_manholes.py).

Layout written to --output:
    <set>/images/*.png  <set>/detections.csv  <set>/ground_truth.csv   (ground truth has columns
                                                                        scene_id, feature, level, level_value)
    suite_index.csv     one row per (set, level): the setting used

Usage:
    python make_test_suite.py --output suite
    python make_test_suite.py --output suite --fitter simple_ellipse_batch_rules.py
"""
import argparse
import csv
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Baseline = "clean": NO camera blur and NO sensor noise, no occlusion/shadow/paint/rim/JPEG.
# Every set starts from this clean image and adds only its own effect (levels are evenly spaced).
BASE_ARGS = ["--blur", "0", "--noise", "0"]
# Each feature: (generator option(s) that change per level, list of 5 levels).
# A level is a dict of generator arguments; "label" is the number written to level_value.
FEATURES = {
    "clean": [dict(label=0, args=[])],
    "occlusion": [dict(label=f, args=["--occlude", str(f)]) for f in (0.1, 0.2, 0.3, 0.4, 0.5)],
    "shadow": [dict(label=f, args=["--shadow", str(f)]) for f in (0.2, 0.35, 0.5, 0.65, 0.8)],
    "noise": [dict(label=s, args=["--noise", str(s)]) for s in (4, 6, 8, 10, 12)],
    "blur": [dict(label=s, args=["--blur", str(s)]) for s in (1, 1.5, 2, 2.5, 3)],
    # motion blur: label = streak length in px at the manhole, along its image motion direction
    "motion_blur": [dict(label=L, args=["--motion-blur", str(L)]) for L in (0.8, 1.6, 2.4, 3.2, 4.0)],
    "low_contrast": [dict(label=c, args=["--contrast", str(c), str(c)]) for c in (30, 24, 18, 12, 6)],
    "rim_ring": [dict(label=r, args=["--rim", str(r)]) for r in (0.04, 0.07, 0.10, 0.13, 0.16)],
    "paint": [dict(label=1, args=["--paint"])],
    "jpeg": [dict(label=q, args=["--jpeg", str(q)]) for q in (85, 70, 55, 40, 25)],
    "distance": [dict(label=hi, args=["--dist", str(lo), str(hi), "--min-minor", "3"])
                 for lo, hi in ((6, 8), (8, 10), (10, 12), (12, 14), (14, 16))],
}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def build_set(name, levels, out_dir, n_scenes, seed, extra, mode="full"):
    """mode full: every scene at every level. mode split: scene k gets level k mod len(levels).
    All scenes use the same --seed so they match across sets."""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    items, index_rows = [], []
    for li, lev in enumerate(levels, start=1):
        scenes = list(range(n_scenes)) if mode == "full" else [k for k in range(n_scenes) if k % len(levels) == li - 1]
        with tempfile.TemporaryDirectory() as tmp:
            run([sys.executable, str(HERE / "generate_kitti_manholes.py"), "--output", tmp,
                 "--seed", str(seed), "--scene-ids"] + [str(k) for k in scenes] + BASE_ARGS + extra + lev["args"])
            dets = list(csv.DictReader(open(Path(tmp) / "detections.csv", newline="")))
            gts = list(csv.DictReader(open(Path(tmp) / "ground_truth.csv", newline="")))
            for d, g in zip(dets, gts):
                new = f"{name}_L{li}_s{int(g['scene_id']):03d}.png"
                shutil.copy(Path(tmp) / "images" / d["filename"], out_dir / "images" / new)
                d["filename"] = new
                g.update(filename=new, feature=name, level=li, level_value=lev["label"])
                items.append((int(g["scene_id"]), d, g))
        index_rows.append(dict(set=name, level=li, level_value=lev["label"], images=len(dets),
                               settings=" ".join(lev["args"]) or "(baseline)"))
    items.sort(key=lambda t: (int(t[2]["level"]), t[0]))
    det_rows, gt_rows = [], []
    for i, (_, d, g) in enumerate(items):
        g["detection_index"] = i                 # row position in detections.csv
        det_rows.append(d)
        gt_rows.append(g)
    with open(out_dir / "detections.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(det_rows[0].keys()))
        w.writeheader()
        w.writerows(det_rows)
    with open(out_dir / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gt_rows[0].keys()))
        w.writeheader()
        w.writerows(gt_rows)
    return index_rows


def boot_ci(values, n=2000, seed=0):
    """95% bootstrap confidence interval of the median, as text 'lo..hi'."""
    if len(values) < 3:
        return ""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n, len(v)), replace=True), axis=1)
    return f"{np.percentile(meds, 2.5):.2f}..{np.percentile(meds, 97.5):.2f}"


def summarise(name, cmp_csv, n_truth_by_level, clean_rows):
    """Per level: unpaired medians, plus paired differences (feature - clean) for scenes fitted in both,
    and 'lost' = scenes fitted in clean but NOT fitted here."""
    rows = list(csv.DictReader(open(cmp_csv, newline=""))) if cmp_csv.exists() else []
    out = []
    for li, n_truth in sorted(n_truth_by_level.items()):
        sub = [r for r in rows if int(r["level"]) == li]
        def med(c, f=abs):
            v = [f(float(r[c])) for r in sub]
            return float(np.median(v)) if v else float("nan")
        pairs = [(r, clean_rows[r["scene_id"]]) for r in sub if r["scene_id"] in clean_rows] if clean_rows else []
        fitted_ids = {r["scene_id"] for r in sub}
        scene_ids = {r["scene_id"] for r in rows if int(r["level"]) == li} | {sid for sid in fitted_ids}
        d_rms = [float(a["boundary_rms"]) - float(b["boundary_rms"]) for a, b in pairs]
        d_dM = [float(a["dM_pct"]) - float(b["dM_pct"]) for a, b in pairs]
        d_dm = [float(a["dm_pct"]) - float(b["dm_pct"]) for a, b in pairs]
        d_c = [math.hypot(float(a["dx"]), float(a["dy"])) - math.hypot(float(b["dx"]), float(b["dy"])) for a, b in pairs]
        out.append(dict(
            set=name, level=li, level_value=sub[0]["level_value"] if sub else "", n=n_truth, fitted=len(sub),
            fitted_pct=round(100 * len(sub) / n_truth, 1),
            rms_median=round(med("boundary_rms"), 3),
            centre_err_median=round(float(np.median([math.hypot(float(r["dx"]), float(r["dy"])) for r in sub]))
                                    if sub else float("nan"), 3),
            major_pct_signed=round(med("dM_pct", lambda x: x), 3), minor_pct_signed=round(med("dm_pct", lambda x: x), 3),
            pairs=len(pairs),
            d_rms_median=round(float(np.median(d_rms)), 3) if d_rms else float("nan"),
            d_rms_ci95=boot_ci(d_rms),
            d_rms_worse_pct=round(100 * float(np.mean([x > 0 for x in d_rms])), 1) if d_rms else float("nan"),
            d_centre_median=round(float(np.median(d_c)), 3) if d_c else float("nan"),
            d_major_pct_median=round(float(np.median(d_dM)), 3) if d_dM else float("nan"),
            d_minor_pct_median=round(float(np.median(d_dm)), 3) if d_dm else float("nan")))
    return out


def lost_counts(name, out_dir, clean_ids):
    """Per level: scenes fitted in the clean set but not fitted in this set (paired failure count)."""
    n_by, fitted_by = {}, {}
    for r in csv.DictReader(open(out_dir / name / "ground_truth.csv", newline="")):
        n_by.setdefault(int(r["level"]), []).append(r["scene_id"])
    cmp_csv = out_dir / name / "compare.csv"
    ok = {}
    if cmp_csv.exists():
        for r in csv.DictReader(open(cmp_csv, newline="")):
            ok.setdefault(int(r["level"]), set()).add(r["scene_id"])
    return {li: sum(1 for sid in ids if sid in clean_ids and sid not in ok.get(li, set()))
            for li, ids in n_by.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-set", type=int, default=100, help="number of scenes")
    ap.add_argument("--mode", choices=["full", "split"], default="full",
                    help="full: every scene at every level; split: each scene at one level only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="build only these sets")
    ap.add_argument("--gen-extra", default="", help="extra options passed to every generator call")
    ap.add_argument("--fitter", help="fitting script to run on every set (e.g. simple_ellipse_batch_rules.py)")
    ap.add_argument("--fit-extra", default="", help="extra options for the fitting script")
    args = ap.parse_args()

    out = Path(args.output)
    extra = shlex.split(args.gen_extra)
    index, summary = [], []
    clean_rows, clean_ids = {}, set()
    for name, levels in FEATURES.items():
        if args.only and name not in args.only:
            continue
        print(f"building {name} ({len(levels)} level(s))")
        index += build_set(name, levels, out / name, args.per_set, args.seed, extra, args.mode)
        if args.fitter:
            fit_dir, cmp_csv = out / name / "fit", out / name / "compare.csv"
            run([sys.executable, args.fitter, "--images", str(out / name / "images"),
                 "--detections", str(out / name / "detections.csv"), "--output", str(fit_dir)]
                + shlex.split(args.fit_extra))
            run([sys.executable, str(HERE / "compare_to_truth.py"), "--results", str(fit_dir / "ellipse_results.csv"),
                 "--truth", str(out / name / "ground_truth.csv"), "--output", str(cmp_csv)])
            n_by = {}
            for r in csv.DictReader(open(out / name / "ground_truth.csv", newline="")):
                n_by[int(r["level"])] = n_by.get(int(r["level"]), 0) + 1
            if name == "clean":
                clean_rows = {r["scene_id"]: r for r in csv.DictReader(open(cmp_csv, newline=""))}
                clean_ids = set(clean_rows)
            rows_here = summarise(name, cmp_csv, n_by, clean_rows if name != "clean" else {})
            if name != "clean":
                lost = lost_counts(name, out, clean_ids)
                for r in rows_here:
                    r["lost_vs_clean"] = lost.get(r["level"], 0)
            else:
                for r in rows_here:
                    r["lost_vs_clean"] = 0
            summary += rows_here

    with open(out / "suite_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "level", "level_value", "images", "settings"])
        w.writeheader()
        w.writerows(index)
    if summary:
        with open(out / "suite_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print("\nPAIRED vs clean twin of the same scene (d_ = feature - clean; negative major/minor = smaller)")
        print("set            level value  n fit% lost | rms  d_rms [95% CI] worse% | d_centre d_major% d_minor%")
        for r in summary:
            print(f"{r['set']:<14} {r['level']:>5} {str(r['level_value']):>5} {r['n']:>3} {r['fitted_pct']:>5} "
                  f"{r['lost_vs_clean']:>4} | {r['rms_median']:>5} {r['d_rms_median']:>6} [{r['d_rms_ci95']:>10}] {r['d_rms_worse_pct']:>6} | "
                  f"{r['d_centre_median']:>8} {r['d_major_pct_median']:>8} {r['d_minor_pct_median']:>8}")
    print(f"\ndone -> {out}")


if __name__ == "__main__":
    main()
