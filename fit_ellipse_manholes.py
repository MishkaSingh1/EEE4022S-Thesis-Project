#!/usr/bin/env python3
"""
  1. Edge points      cv2.Canny on the blurred crop (whole-pixel positions).
  2. RANSAC           pick 5 edge points, fit an ellipse, count the points within `threshold` px
                      (Sampson distance) of it, keep the candidate with the most.
  3. Box rules        a candidate (and the final ellipse) must be plausible for its detector box.
  4. Final fit        least squares on the Sampson distance, using the inliers of the best candidate.

    python minimal_ellipse_batch.py --images IMG_DIR --detections dets.csv --output OUT_DIR
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


# ---- A. Edge points --------------------------------------------------------
def edge_points(gray, sigma=1.0, rel_thresh=0.3, min_strength=2.0):
    """Edge pixels of a grey crop as an (N, 2) array of (x, y) pixel coordinates, or None.

    Steps: Gaussian blur -> gradient strength (Sobel) -> pick the Canny thresholds from the
    strongest gradients in this crop -> cv2.Canny. cv2.Canny does not blur by itself, so the blur
    comes first. Its Sobel is not divided by 8, so thresholds are multiplied by 8 to be in grey
    levels per pixel like the old script."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    strongest = np.percentile(np.hypot(gx, gy), 99.5)
    high = max(min_strength, rel_thresh * strongest)        # strong edge: starts an edge line
    low = 0.5 * high                                        # weak edge: kept if joined to a strong one
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    edges = cv2.Canny(g8, 8 * low, 8 * high, apertureSize=3, L2gradient=True)
    ys, xs = np.nonzero(edges)
    if len(xs) < 10:
        return None
    return np.stack([xs, ys], axis=1).astype(np.float64)


# ---- B. Sampson distance ---------------------------------------------------
def sampson_distance(points, cx, cy, a, b, theta):
    """Approximate distance (px) from each point to the ellipse outline: F / |grad F|,
    with F = (u/a)^2 + (v/b)^2 - 1 in the ellipse's own axes (u along a, v along b)."""
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    u = dx * c + dy * s
    v = -dx * s + dy * c
    F = (u / a) ** 2 + (v / b) ** 2 - 1.0
    grad = np.hypot(2 * u / a ** 2, 2 * v / b ** 2) + 1e-12
    return F / grad


def cv_to_params(rect):
    (cx, cy), (w, h), ang = rect
    return cx, cy, w / 2.0, h / 2.0, math.radians(ang)


# ---- C. Box rules ----------------------------------------------------------
def ellipse_extent(cx, cy, a, b, theta):
    """Half-width and half-height of the ellipse's bounding rectangle."""
    c, s = math.cos(theta), math.sin(theta)
    return math.hypot(a * c, b * s), math.hypot(a * s, b * c)


def tilt_deg(p):
    """Angle of the MAJOR axis from horizontal, in (-90, 90] degrees."""
    cx, cy, a, b, th = p
    ang = math.degrees(th if a >= b else th + math.pi / 2)
    return (ang + 90.0) % 180.0 - 90.0


def area_fill(p, box):
    """Ellipse area / area of the ellipse inscribed in the box (1.0 = fills the box)."""
    cx, cy, a, b, th = p
    return 4 * a * b / ((box[2] - box[0]) * (box[3] - box[1]))


def box_failures(p, box, rules):
    """Names of the plausibility rules that ellipse p breaks (empty list = plausible)."""
    if not np.all(np.isfinite(p)):
        return ["not_finite"]
    cx, cy, a, b, th = p
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    ov = rules["overshoot"]
    hx, hy = ellipse_extent(cx, cy, a, b, th)
    out = []
    if cx - hx < bx1 - ov * bw: out.append("reach_left")
    if cx + hx > bx2 + ov * bw: out.append("reach_right")
    if cy - hy < by1 - ov * bh: out.append("reach_top")
    if cy + hy > by2 + ov * bh: out.append("reach_bottom")
    if 2 * hx < rules["min_fill_w"] * bw: out.append("too_narrow")
    if 2 * hy < rules["min_fill_h"] * bh: out.append("too_short")
    if min(a, b) / max(a, b) < 0.6 and abs(tilt_deg(p)) > rules["max_tilt"]: out.append("tilted")
    if area_fill(p, box) < rules["min_area_fill"]: out.append("low_area")
    if min(a, b) < 1.0: out.append("tiny_axis")
    return out


# ---- D. RANSAC -------------------------------------------------------------
def ransac_ellipse(points, box, rules, threshold=1.0, iterations=500, seed=0):
    """Return a boolean mask of the inliers of the best candidate, or None if no candidate
    passed the box rules."""
    rng = np.random.default_rng(seed)
    best_mask, best_count = None, 0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 5, replace=False)].astype(np.float32)
        try:
            p = cv_to_params(cv2.fitEllipse(sample))
        except cv2.error:
            continue
        if box_failures(p, box, rules):
            continue
        mask = np.abs(sampson_distance(points, *p)) < threshold
        if mask.sum() > best_count:
            best_count, best_mask = int(mask.sum()), mask
    return best_mask


# ---- E. Least squares on the Sampson distance ------------------------------
LOG_AXIS_MIN, LOG_AXIS_MAX = math.log(0.5), math.log(5000.0)     # semi-axes kept between 0.5 and 5000 px


def sampson_least_squares(points, start):
    """Refine (cx, cy, a, b, theta) by minimising the Sampson distances of `points`.
    Returns None if the optimiser diverges (a nearly straight point set can make an axis explode)."""
    cx, cy, a, b, th = start
    x0 = np.array([cx, cy, math.log(a), math.log(b), th])

    def residuals(q):
        if not (np.all(np.isfinite(q)) and LOG_AXIS_MIN <= q[2] <= LOG_AXIS_MAX
                and LOG_AXIS_MIN <= q[3] <= LOG_AXIS_MAX):
            return np.full(len(points), 1e6)               # huge but finite: the optimiser backs off
        return sampson_distance(points, q[0], q[1], math.exp(q[2]), math.exp(q[3]), q[4])

    try:
        q = least_squares(residuals, x0).x
    except (OverflowError, ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
        return None
    if not (np.all(np.isfinite(q)) and LOG_AXIS_MIN <= q[2] <= LOG_AXIS_MAX and LOG_AXIS_MIN <= q[3] <= LOG_AXIS_MAX):
        return None
    return q[0], q[1], math.exp(q[2]), math.exp(q[3]), q[4]


# ---- E2. Edge shift (--edge-shift) -------------------------------------------
def shift_inward(points, ellipse, shift):
    """Move each point `shift` px towards the inside of the ellipse, along the ellipse's normal at
    that point (the normal is the direction of the gradient of F = (u/a)^2 + (v/b)^2 - 1)."""
    cx, cy, a, b, th = ellipse
    c, s = math.cos(th), math.sin(th)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    u, v = dx * c + dy * s, -dx * s + dy * c                    # position in the ellipse's own axes
    gu, gv = 2 * u / a ** 2, 2 * v / b ** 2                     # gradient of F in those axes
    gx, gy = gu * c - gv * s, gu * s + gv * c                   # ... turned back into image axes
    norm = np.hypot(gx, gy) + 1e-12
    return points - shift * np.stack([gx / norm, gy / norm], axis=1)   # minus = inward


# ---- F. One detection ------------------------------------------------------
def fit_one(image, x1, y1, x2, y2, padding, rules, threshold, diag):
    h, w = image.shape[:2]
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None, "empty_crop"
    box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)          # the detector box in crop coordinates

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    points = edge_points(gray)
    if points is None:
        return None, "no_edges"
    diag["num_points"] = len(points)
    diag.update(_crop=crop, _box=box, _points=points)       # kept only so --save-failed can draw them

    mask = ransac_ellipse(points, box, rules, threshold)
    diag["_mask"] = mask
    if mask is None or mask.sum() < 6:
        diag["fail_reason"] = "no_candidate_passed_box_rule" if mask is None else "fewer_than_6_inliers"
        return None, "ransac_failed"
    inliers = points[mask]

    start = cv_to_params(cv2.fitEllipse(inliers.astype(np.float32)))
    diag["_ellipse"] = start
    final = sampson_least_squares(inliers, start)
    if final is None:
        diag["fail_reason"] = "least_squares_diverged"
        return None, "lsq_failed"
    diag["_ellipse"] = final
    fails = box_failures(final, box, rules)
    if fails:
        diag["fail_reason"] = ",".join(fails)
        return None, "final_outside_box"

    fitted_points = inliers
    shift = rules.get("edge_shift", 0.0)
    if shift > 0:
        moved = shift_inward(inliers, final, shift)
        shifted = sampson_least_squares(moved, final)
        if shifted is None:
            diag["fail_reason"] = "edge_shift_least_squares_diverged"
            return None, "lsq_failed"
        final, fitted_points = shifted, moved
        diag["_ellipse"] = final

    cx, cy, a, b, th = final
    rms = float(np.sqrt(np.mean(sampson_distance(fitted_points, cx, cy, a, b, th) ** 2)))
    if b > a:
        a, b, th = b, a, th + math.pi / 2
    return {"center_x": cx + cx1, "center_y": cy + cy1, "major_axis": 2 * a, "minor_axis": 2 * b,
            "angle": math.degrees(th) % 180.0, "num_inliers": int(mask.sum()),
            "num_points": len(points), "rms_px": rms}, "ok"


# ---- G. Picture of a failed detection (only with --save-failed) -------------
def draw_failed(path, diag, status):
    """Save the crop enlarged (about 800 px wide) with: the detector box (green), every edge point (cyan),
    the RANSAC inliers (red) and the ellipse that was rejected (yellow), plus the reason as text."""
    scale = int(min(12, max(4, 800 // diag["_crop"].shape[1])))
    img = cv2.resize(diag["_crop"], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    x1, y1, x2, y2 = diag["_box"]
    cv2.rectangle(img, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), (0, 200, 0), 1)
    mask = diag.get("_mask")
    for i, (x, y) in enumerate(diag["_points"]):
        inlier = mask is not None and mask[i]
        cv2.circle(img, (int((x + 0.5) * scale), int((y + 0.5) * scale)), 1,
                   (0, 0, 255) if inlier else (255, 255, 0), -1)     # pixel centre (x, y) sits at x + 0.5 after enlarging
    if "_ellipse" in diag:
        cx, cy, a, b, th = diag["_ellipse"]
        cv2.ellipse(img, (((cx + 0.5) * scale, (cy + 0.5) * scale), (2 * a * scale, 2 * b * scale),
                          math.degrees(th)), (0, 255, 255), 1)
    cv2.putText(img, f"{status}: {diag.get('fail_reason', '')}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


# ---- H. Batch loop ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--padding", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=1.0, help="RANSAC inlier distance in px")
    ap.add_argument("--overshoot", type=float, default=0.2,
                    help="how far past the box the ellipse may reach, as a fraction of the box size (was 0.1)")
    ap.add_argument("--min-fill-w", type=float, default=0.7,
                    help="ellipse width  >= this x box width")
    ap.add_argument("--min-fill-h", type=float, default=0.5,
                    help="ellipse height >= this x box height")
    ap.add_argument("--max-tilt", type=float, default=10.0)
    ap.add_argument("--min-area-fill", type=float, default=0.5,
                    help="ellipse area / box-inscribed-ellipse area >= this")
    ap.add_argument("--save-failed", action="store_true",
                    help="also save a picture of every rejected detection to OUT/failed_images")
    ap.add_argument("--edge-shift", type=float, default=0.0,
                    help="OPTIONAL: move the fitted edge points this many px INWARD before a final refit, to undo "
                         "the outward shift of a blurred thin ring's edge (default 0 = off; ~1.0 suits the KITTI tests)")
    args = ap.parse_args()
    if args.edge_shift < 0:
        ap.error("--edge-shift must be 0 or positive (it moves the edge inward)")
    rules = dict(overshoot=args.overshoot, min_fill_w=args.min_fill_w, min_fill_h=args.min_fill_h,
                 max_tilt=args.max_tilt, min_area_fill=args.min_area_fill, edge_shift=args.edge_shift)

    out = Path(args.output)
    (out / "ellipse_images").mkdir(parents=True, exist_ok=True)
    fields = ["filename", "detection_index", "status", "ellipse_fit", "center_x", "center_y", "major_axis", "minor_axis",
              "angle", "num_inliers", "num_points", "rms_px", "fail_reason"]
    rows = list(csv.DictReader(open(args.detections, newline="")))

    results, cache_name, cache_img = [], None, None
    for i, row in enumerate(rows):
        if row.get("x1", "") == "":
            continue
        name = row["filename"]
        if name != cache_name:
            cache_name, cache_img = name, cv2.imread(str(Path(args.images) / name))
        rec = {k: "" for k in fields}
        rec.update(filename=name, detection_index=i)
        if cache_img is None:
            rec["status"] = "unreadable_image"
        else:
            box = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
            diag = {}
            fit, status = fit_one(cache_img, *box, args.padding, rules, args.threshold, diag)
            rec["status"] = status
            rec.update({k: v for k, v in diag.items() if k in fields})
            if not fit and args.save_failed and "_crop" in diag:
                (out / "failed_images").mkdir(exist_ok=True)
                draw_failed(out / "failed_images" / f"{Path(name).stem}_det{i}_{status}.png", diag, status)
            if fit:
                rec.update({k: round(v, 3) if isinstance(v, float) else v for k, v in fit.items()})
                drawn = cache_img.copy()
                cv2.rectangle(drawn, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 200, 0), 1)
                cv2.ellipse(drawn, ((fit["center_x"], fit["center_y"]), (fit["major_axis"], fit["minor_axis"]),
                                    fit["angle"]), (0, 255, 255), 1)
                cv2.imwrite(str(out / "ellipse_images" / f"{Path(name).stem}_det{i}.png"), drawn)
        print(f"{name} det {i}: {rec['status']}")
        results.append(rec)

    for r in results:
        r["ellipse_fit"] = r["status"]   # same value, under the name camera_calibration_from_ellipse.py reads
    with open(out / "ellipse_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    ok = sum(r["status"] == "ok" for r in results)
    print(f"done: {ok}/{len(results)} fitted -> {out / 'ellipse_results.csv'}")


if __name__ == "__main__":
    main()
