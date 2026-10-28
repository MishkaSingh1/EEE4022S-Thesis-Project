
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def ellipse_geom_to_conic(cx, cy, semi_a, semi_b, theta):
    Auu = np.cos(theta) ** 2 / semi_a ** 2 + np.sin(theta) ** 2 / semi_b ** 2
    Avv = np.sin(theta) ** 2 / semi_a ** 2 + np.cos(theta) ** 2 / semi_b ** 2
    Auv = np.cos(theta) * np.sin(theta) * (1 / semi_a ** 2 - 1 / semi_b ** 2)

    H = Auv / Auu
    B = Avv / Auu
    G = -cx - H * cy
    F = -H * cx - B * cy
    E = cx ** 2 + 2 * H * cx * cy + B * cy ** 2 - 1 / Auu
    return H, B, G, F, E


def calibrate_from_conic(H, B, G, F, E, R, known_f=None):

    if abs(G) < 1e-9:
        raise ValueError("G is ~0 (degenerate conic for this formula, division by zero)")

    tilt_arg = B + (E * H * H) / (G * G) - ((2 * H * F) / G)
    if tilt_arg <= 0:
        raise ValueError(f"invalid domain for tilt: 1/sqrt(x) requires x>0, got {tilt_arg:.6g}")
    sqrt_term = np.sqrt(tilt_arg)
    asin_arg = 1.0 / sqrt_term
    if not (-1.0 <= asin_arg <= 1.0):
        raise ValueError(f"invalid domain for arcsin: got {asin_arg:.6g} (must be in [-1, 1])")
    phi = np.arcsin(asin_arg)

    if abs(H) < 1e-9:
        raise ValueError("H is ~0 (division by zero computing focal length)")
    if abs(np.tan(phi)) < 1e-9:
        raise ValueError("tan(phi) ~0 (division by zero computing focal length)")
    f_computed = (G / H) / np.tan(phi)
    f_used = known_f if known_f is not None else f_computed

    G2_minus_E = (G * G) - E
    denom = (H *H * (f_used *f_used )* G2_minus_E )+( ((F * G - H * E) * (F * G - H * E) )* (np.cos(phi) * np.cos(phi)))
    numer = R *R * (f_used *f_used )* (G *G) * (np.cos(phi) * np.cos(phi))
    if denom == 0:
        raise ValueError("denominator ~0 computing camera height")
    h_squared = numer / denom
    if h_squared < 0:
        raise ValueError(f"negative value under sqrt for height: {h_squared:.6g}")
    h = np.sqrt(h_squared)

    if abs(np.cos(phi)) < 1e-9:
        raise ValueError("cos(phi) ~0 (division by zero computing circle-centre a)")
    a = H * h / np.cos(phi)

    if abs(f_used) < 1e-9:
        raise ValueError("f ~0 (division by zero computing circle-centre b)")
    b = (F - (H * E) / G) * (h / f_used)

    if abs(b) < 1e-9:
        theta = np.pi / 2
    elif b > 0:
        theta = np.arctan(abs(a / b))
    else:
        theta = np.pi - np.arctan(abs(a / b))

    return {"tilt_rad": phi, "focal_length_px": f_computed, "focal_length_used_px": f_used,
            "height": h, "circle_center_a": a, "circle_center_b": b, "pan_rad": theta}


# ---------------------------------------------------------------------------
# JOINT FIT OVER SEVERAL FRAMES / CIRCLES (--joint)
#
# Several ellipses seen by ONE camera in ONE pose (e.g. the same manhole in consecutive frames
# from a car, or several circles in one photo) share the camera parameters; only each circle's
# ground position differs. Paper's model: shared (f, tilt, h), per-circle (a_k, b_k).
# 
#
# Projection: the paper's eq (4) with w_z = 0 and the paper's ground origin (where the optical
# axis meets the ground), as a 3x3 matrix acting on ground points (w_x, w_y, 1):
#     P = [[f, 0, 0], [0, f sin(phi), 0], [0, -cos(phi), -h/sin(phi)]]
# which gives exactly the paper's eq (5). Each circle becomes the image conic
# C_k = P^-T Circle_k P^-1, and the fit minimises the Sampson distance (approximately the pixel
# distance) of every edge point from its predicted ellipse, over all frames at once. 
# ---------------------------------------------------------------------------

JOINT_MIN_TILT = np.radians(0.05)


def _joint_projection(f, phi, h, roll=0.0):
    s, c = np.sin(phi), np.cos(phi)
    P = np.array([[f, 0.0, 0.0], [0.0, f * s, 0.0], [0.0, -c, -h / s]])
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]]) @ P


def _joint_unpack(p, n, with_roll):
    if with_roll:
        f, phi, roll, h = p[:4]
        rest = p[4:]
    else:
        f, phi, h = p[:3]
        roll = 0.0
        rest = p[3:]
    return f, phi, roll, h, rest.reshape(n, 2)


def _joint_residuals(p, frames, R, with_roll):
    f, phi, roll, h, ab = _joint_unpack(p, len(frames), with_roll)
    Pinv = np.linalg.inv(_joint_projection(f, phi, h, roll))
    out = []
    for (x, y), (a, b) in zip(frames, ab):
        Cg = np.array([[1.0, 0.0, -a], [0.0, 1.0, -b], [-a, -b, a * a + b * b - R * R]])
        C = Pinv.T @ Cg @ Pinv
        X = np.vstack([x, y, np.ones_like(x)])
        CX = C @ X
        out.append(np.sum(X * CX, axis=0) / (2.0 * np.hypot(CX[0], CX[1]) + 1e-12))
    return np.concatenate(out)


def joint_fit(frames, conics, R, with_roll=False):
    n = len(frames)
    if with_roll and n < 2:
        raise ValueError("--joint-roll needs at least 2 frames: one ellipse has 5 numbers, "
                         "fewer than the 6 unknowns (f, tilt, roll, h, a, b)")
    # starting values from the paper's closed form, eqs (10)-(15)
    seeds = []
    for con in conics:
        try:
            q = calibrate_from_conic(*con, R)
            if q["tilt_rad"] > 0 and q["focal_length_px"] > 0:   # skip the mirrored (negative-f) solution
                seeds.append(q)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            pass
    if not seeds:
        raise ValueError("no frame gave a valid closed-form starting point (eqs 10-15)")
    f0 = float(np.median([q["focal_length_px"] for q in seeds]))
    phi0 = float(np.clip(np.median([q["tilt_rad"] for q in seeds]), 2 * JOINT_MIN_TILT, np.radians(89.0)))
    h0 = float(np.median([q["height"] for q in seeds]))
    # each circle's starting position: its edge points' centre mapped onto the ground
    Pinv0 = np.linalg.inv(_joint_projection(f0, phi0, h0))
    ab0 = []
    for x, y in frames:
        g = Pinv0 @ np.array([np.mean(x), np.mean(y), 1.0])
        ab0 += [g[0] / g[2], g[1] / g[2]]
    p0 = np.array(([f0, phi0, 0.0, h0] if with_roll else [f0, phi0, h0]) + ab0)

    lo = np.full(p0.size, -np.inf)
    hi = np.full(p0.size, np.inf)
    lo[1], hi[1] = JOINT_MIN_TILT, np.radians(89.5)
    res = least_squares(_joint_residuals, p0, args=(frames, R, with_roll), x_scale="jac",
                        max_nfev=2000, bounds=(lo, hi))
    f, phi, roll, h, ab = _joint_unpack(res.x, n, with_roll)

    # standard deviations from the fit: cov = (J^T J)^-1 * residual variance
    m, k = res.fun.size, res.x.size
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * (res.fun @ res.fun) / max(m - k, 1)
        sd = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sd = np.full(k, np.nan)
    names = ["f", "phi", "roll", "h"] if with_roll else ["f", "phi", "h"]
    sd_named = dict(zip(names, sd[:len(names)]))

    per_rms, start = [], 0
    for x, _ in frames:
        r = res.fun[start:start + len(x)]
        per_rms.append(float(np.sqrt(np.mean(r * r))))
        start += len(x)
    return {"focal_length_px": f, "tilt_rad": phi, "roll_rad": roll, "height": h, "ab": ab,
            "sd_focal_length_px": sd_named["f"], "sd_tilt_rad": sd_named["phi"],
            "sd_roll_rad": sd_named.get("roll", 0.0), "sd_height": sd_named["h"],
            "rms_px": float(np.sqrt(np.mean(res.fun ** 2))), "frame_rms_px": per_rms,
            "converged": bool(res.success), "n_frames": n, "with_roll": with_roll,
            "tilt_at_limit": bool(phi <= JOINT_MIN_TILT * 1.001)}


def ground_distance_joint(p1, p2, phi, f, h, roll):

    cr, sr = np.cos(roll), np.sin(roll)
    ground = []
    for ix, iy in (p1, p2):
        u, v = cr * ix + sr * iy, -sr * ix + cr * iy
        if f * np.sin(phi) + v * np.cos(phi) <= 1e-9:
            raise ValueError(f"point ({ix:.1f}, {iy:.1f}) is at or above the horizon")
        ground.append(back_project(u, v, phi, f, h))
    (x1, y1), (x2, y2) = ground
    return float(np.hypot(x2 - x1, y2 - y1))


def back_project(ix, iy, phi, f, h):

    D = f * np.sin(phi) + iy * np.cos(phi)
    wx = -h * ix / D
    wy = -h * iy / (np.sin(phi) * D)
    return wx, wy


def ground_distance(p1, p2, phi, f, h):

    ground = []
    for ix, iy in (p1, p2):
        D = f * np.sin(phi) + iy * np.cos(phi)
        if D <= 1e-9:
            raise ValueError(f"point ({ix:.1f}, {iy:.1f}) is at or above the horizon for this "
                             f"camera pose, so it can't be on the ground")
        ground.append(back_project(ix, iy, phi, f, h))
    (x1, y1), (x2, y2) = ground
    return float(np.hypot(x2 - x1, y2 - y1))


def optimise_from_points(ix, iy, R, seed, known_f=None, bound_frac=0.5):
  
    fix_f = known_f is not None
    f_fixed = known_f if fix_f else None

    names = ["tilt_rad", "height", "circle_center_a", "circle_center_b"]
    if not fix_f:
        names.insert(1, "focal_length_px")
    p0 = np.array([seed[k] for k in names], dtype=float)

    lo = p0 - bound_frac * np.abs(p0)
    hi = p0 + bound_frac * np.abs(p0)
    tiny = (hi - lo) < 1e-9          # a parameter that is exactly 0 would get zero-width bounds
    lo[tiny] -= 1e-3
    hi[tiny] += 1e-3
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    # the paper's model needs a positive tilt
    lo[0] = max(lo[0], np.radians(0.5))
    hi[0] = min(hi[0], np.radians(89.5))
    if lo[0] >= hi[0]:
        raise ValueError("tilt bounds collapsed; cannot optimise")
    p0 = np.clip(p0, lo, hi)

    def unpack(p):
        if fix_f:
            phi, h, a, b = p
            f = f_fixed
        else:
            phi, f, h, a, b = p
        return phi, f, h, a, b

    def residuals(p):
        phi, f, h, a, b = unpack(p)
        wx, wy = back_project(ix, iy, phi, f, h)
        return (wx - a) ** 2 + (wy - b) ** 2 - R ** 2

    res = least_squares(residuals, p0, bounds=(lo, hi))
    phi, f, h, a, b = unpack(res.x)

    if abs(b) < 1e-9:
        pan = np.pi / 2
    elif b > 0:
        pan = np.arctan(abs(a / b))
    else:
        pan = np.pi - np.arctan(abs(a / b))

    # which parameters finished pressed against a +/-50% bound?
    at_bound = [names[i] for i in range(len(names))
                if np.isclose(res.x[i], lo[i], rtol=1e-6, atol=1e-9)
                or np.isclose(res.x[i], hi[i], rtol=1e-6, atol=1e-9)]

    return {"tilt_rad": phi, "focal_length_px": f, "height": h,
            "circle_center_a": a, "circle_center_b": b, "pan_rad": pan,
            "cost": float(np.sum(res.fun ** 2)), "n_points": len(ix),
            "converged": bool(res.success), "at_bound": at_bound}


def points_from_ellipse(cx, cy, major, minor, angle_deg, n=36):
    """Sample points around a fitted ellipse (fallback when no raw clicked points are given)."""
    th = np.radians(angle_deg)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + (major / 2) * np.cos(t) * np.cos(th) - (minor / 2) * np.sin(t) * np.sin(th)
    y = cy + (major / 2) * np.cos(t) * np.sin(th) + (minor / 2) * np.sin(t) * np.cos(th)
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ellipses", required=True,
                         help="Path to ellipse_results.csv (output of fit_ellipse_manholes.py)")
    parser.add_argument("--image-width", type=float, required=True,
                         help="Width (px) of the image that center_x/center_y in the CSV are "
                              "expressed in -- i.e. whatever image you actually fed into "
                              "fit_ellipse_manholes.py (a manual crop, if you made one).")
    parser.add_argument("--image-height", type=float, required=True,
                         help="Height (px) of the image that center_x/center_y in the CSV are "
                              "expressed in (same purpose as --image-width).")
    parser.add_argument("--original-width", type=float, default=None,
                         help="OPTIONAL: width (px) of the true original, uncropped photo. "
                              "Only needed if your --image-width/--image-height image is a "
                              "manual crop that was NOT centred on the original photo. "
                              "Must be given together with --original-height, --crop-x0, "
                              "--crop-y0.")
    parser.add_argument("--original-height", type=float, default=None,
                         help="OPTIONAL: height (px) of the true original, uncropped photo. "
                              "See --original-width.")
    parser.add_argument("--crop-x0", type=float, default=None,
                         help="OPTIONAL: x-pixel-coordinate, IN THE ORIGINAL PHOTO, of the "
                              "left edge of your manual crop. E.g. if you cropped starting "
                              "200px in from the left of the original, this is 200.")
    parser.add_argument("--crop-y0", type=float, default=None,
                         help="OPTIONAL: y-pixel-coordinate, IN THE ORIGINAL PHOTO, of the "
                              "top edge of your manual crop. See --crop-x0.")
    parser.add_argument("--radius", type=float, required=True,
                         help="Real-world radius of the manhole cover (e.g. metres). "
                              "Common circular covers are roughly 0.3-0.35 m radius, but verify "
                              "this for your actual covers rather than assuming a default.")
    parser.add_argument("--focal-length", type=float, default=None,
                         help="OPTIONAL: known focal length in pixels (e.g. from EXIF/lens specs). "
                              "If given, this value is used for the height and circle-centre "
                              "calculations (eqs. 12, 14) instead of the value recovered from the "
                              "ellipse (eq. 11). The recovered value is still computed and reported "
                              "(as focal_length_px) so you can compare it against this input.")
    parser.add_argument("--output", required=True, help="Folder to write calibration results to")
    parser.add_argument("--optimize", action="store_true",
                         help="ALSO run the paper's option two (Section 2.4): non-linear least-squares "
                              "refinement over the clicked edge points, starting from the direct "
                              "result, each parameter free to move +/-50%%.")
    parser.add_argument("--points", default=None,
                         help="Raw clicked points CSV (filename, point_index, ix, iy) -- the "
                              "*_points.csv that manual_ellipse_fit.py writes. Used by --optimize. "
                              "If missing for an image, points are sampled from its fitted ellipse "
                              "instead (the result will then barely differ from the direct method).")
    parser.add_argument("--distances", default=None,
                         help="OPTIONAL: CSV of ground point pairs (filename, label, u1, v1, u2, v2, "
                              "true_distance) to measure with each image's calibration. See the "
                              "'distance measurement' section of the docstring.")
    parser.add_argument("--method", choices=["paper"], default="paper",
                        help="Only Dinh & Tang's method is available (eqs (10)-(15), and eq (6) for "
                             "option two and distances). Kept so existing commands still run.")
    parser.add_argument("--principal-x", type=float, default=None,
                        help="OPTIONAL: principal point x (px) in the image the ellipses were fitted "
                             "on, e.g. from a checkerboard calibration (KITTI: 609.5593). Overrides "
                             "the image-centre / crop-offset principal point. Give with --principal-y.")
    parser.add_argument("--principal-y", type=float, default=None,
                        help="OPTIONAL: principal point y (px). See --principal-x (KITTI: 172.854).")
    parser.add_argument("--joint", action="store_true",
                        help="ALSO fit all frames of one camera setup TOGETHER: shared focal length, "
                             "tilt and height, each frame with its own circle position (the paper's "
                             "model with several frames). Frames are grouped by an optional 'group' "
                             "column in --ellipses; without it, all frames are one group. Uses --points "
                             "if given, else points sampled from each fitted ellipse.")
    parser.add_argument("--joint-roll", action="store_true",
                        help="Like --joint, but also estimates a shared camera roll (an EXTENSION "
                             "beyond the paper's model). Needs at least 2 frames per group.")
    args = parser.parse_args()
    if args.joint_roll:
        args.joint = True
    if (args.principal_x is None) != (args.principal_y is None):
        print("[error] give --principal-x and --principal-y together")
        sys.exit(1)

    ellipses_path = Path(args.ellipses)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ellipses_path.exists():
        print(f"[error] ellipses CSV not found: {ellipses_path}")
        sys.exit(1)

    with open(ellipses_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    points_by_file = {}
    if (args.optimize or args.joint) and args.points:
        pts_path = Path(args.points)
        if not pts_path.exists():
            print(f"[error] points CSV not found: {pts_path}")
            sys.exit(1)
        with open(pts_path, newline="") as fh:
            for prow in csv.DictReader(fh):
                points_by_file.setdefault(prow["filename"], []).append(
                    (float(prow["ix"]), float(prow["iy"])))
    if args.optimize and not args.points:
        print("[warn] --optimize given without --points: points will be sampled from each fitted "
              "ellipse, so the optimised result will be almost identical to the direct one.")

    pairs_by_file = {}
    if args.distances:
        dpath = Path(args.distances)
        if not dpath.exists():
            print(f"[error] distances CSV not found: {dpath}")
            sys.exit(1)
        with open(dpath, newline="") as fh:
            drows = list(csv.DictReader(fh))
        need = {"filename", "u1", "v1", "u2", "v2"}
        if drows and not need.issubset(drows[0].keys()):
            print(f"[error] distances CSV must contain columns {sorted(need)} "
                  f"(plus optional label, true_distance); found {list(drows[0].keys())}")
            sys.exit(1)
        for d in drows:
            pairs_by_file.setdefault(d["filename"], []).append(d)
    distance_rows = []

    required_cols = {"filename", "ellipse_fit", "center_x", "center_y", "major_axis", "minor_axis", "angle"}
    if rows and not required_cols.issubset(rows[0].keys()):
        print(f"[error] CSV must contain columns: {required_cols}")
        print(f"        found columns: {list(rows[0].keys())}")
        sys.exit(1)

    crop_args = [args.original_width, args.original_height, args.crop_x0, args.crop_y0]
    n_given = sum(v is not None for v in crop_args)
    if 0 < n_given < 4:
        print("[error] --original-width, --original-height, --crop-x0 and --crop-y0 must be "
              "given TOGETHER (all four) if you're correcting for an off-centre manual crop. "
              f"You gave {n_given} of 4.")
        sys.exit(1)

    using_crop_correction = n_given == 4

    if using_crop_correction:
        # The true principal point is assumed to be the centre of the ORIGINAL photo.
        # We need the offset, in the coordinate system of the CROPPED image (the one
        # center_x/center_y are actually measured in), from the cropped image's own
        # top-left corner to that true principal point.
        #
        # principal point in ORIGINAL image coords:
        principal_x_in_original = args.original_width / 2.0
        principal_y_in_original = args.original_height / 2.0
        # convert to CROPPED image coords by subtracting the crop's top-left offset:
        half_w = principal_x_in_original - args.crop_x0
        half_h = principal_y_in_original - args.crop_y0
        print(f"[info] Using crop-offset correction. True principal point is at "
              f"({half_w:.1f}, {half_h:.1f}) in your cropped image's coordinates "
              f"(cropped image is {args.image_width:.0f}x{args.image_height:.0f}, "
              f"so this is {'off-centre' if abs(half_w - args.image_width/2) > 1 or abs(half_h - args.image_height/2) > 1 else 'approx centred'}).")
    else:
        # Old behaviour: assume the image you point --image-width/--image-height at
        # IS the original, or was cropped symmetrically, so its own centre = principal point.
        half_w = args.image_width / 2.0
        half_h = args.image_height / 2.0

    if args.principal_x is not None:
        half_w, half_h = args.principal_x, args.principal_y
    print(f"[info] principal point in use: ({half_w:.2f}, {half_h:.2f});  method: {args.method}")
    calibrate = calibrate_from_conic

    results = []
    succeeded = 0
    failed = 0
    joint_frames = {}     # group -> list of (filename, (ix, iy), conic)

    for row in rows:
        if row.get("ellipse_fit") != "ok":
            continue  # skip rows where ellipse fitting itself already failed

        filename = row["filename"]
        cx = float(row["center_x"]) - half_w
        cy = float(row["center_y"]) - half_h
        major = float(row["major_axis"])
        minor = float(row["minor_axis"])
        angle_deg = float(row["angle"])
        theta_rad = np.radians(angle_deg)
        semi_a = major / 2.0
        semi_b = minor / 2.0

        if args.joint:
            raw = points_by_file.get(filename)
            if raw and len(raw) >= 5:
                jx = np.array([q[0] for q in raw]) - half_w
                jy = np.array([q[1] for q in raw]) - half_h
            else:
                jx, jy = points_from_ellipse(float(row["center_x"]), float(row["center_y"]),
                                             major, minor, angle_deg, n=60)
                jx, jy = jx - half_w, jy - half_h
            joint_frames.setdefault(row.get("group", "") or "all", []).append(
                (filename, (jx, jy), ellipse_geom_to_conic(cx, cy, semi_a, semi_b, theta_rad)))

        try:
            H, B, G, F, E = ellipse_geom_to_conic(cx, cy, semi_a, semi_b, theta_rad)
            params = calibrate(H, B, G, F, E, args.radius, known_f=args.focal_length)
            succeeded += 1
            row_out = {
                "filename": filename,
                "status": "ok",
                "reason": "",
                "tilt_deg": round(np.degrees(params["tilt_rad"]), 2),
                "pan_deg": round(np.degrees(params["pan_rad"]), 2),
                "focal_length_px": round(params["focal_length_px"], 2),
                "focal_length_used_px": round(params["focal_length_used_px"], 2),
                "camera_height": round(params["height"], 4),
                "circle_center_a": round(params["circle_center_a"], 4),
                "circle_center_b": round(params["circle_center_b"], 4),
                "forward_from_below_camera": round(params["circle_center_b"] + params["height"] / np.tan(params["tilt_rad"]), 4),
                "method": args.method,
            }
            if args.optimize:
                raw = points_by_file.get(filename)
                if raw and len(raw) >= 5:
                    pix = np.array([p[0] for p in raw]) - half_w
                    piy = np.array([p[1] for p in raw]) - half_h
                    src = "clicked points"
                else:
                    fx_, fy_ = points_from_ellipse(float(row["center_x"]), float(row["center_y"]),
                                                   major, minor, angle_deg)
                    pix, piy = fx_ - half_w, fy_ - half_h
                    src = "sampled from ellipse"
                try:
                    opt = optimise_from_points(pix, piy, args.radius, params,
                                               known_f=args.focal_length)
                    row_out.update({
                        "opt_status": "ok" if opt["converged"] else "not converged",
                        "opt_tilt_deg": round(np.degrees(opt["tilt_rad"]), 2),
                        "opt_pan_deg": round(np.degrees(opt["pan_rad"]), 2),
                        "opt_focal_length_px": round(opt["focal_length_px"], 2),
                        "opt_camera_height": round(opt["height"], 4),
                        "opt_circle_center_a": round(opt["circle_center_a"], 4),
                        "opt_circle_center_b": round(opt["circle_center_b"], 4),
                        "opt_forward_from_below_camera": round(opt["circle_center_b"] + opt["height"] / np.tan(opt["tilt_rad"]), 4),
                        "opt_n_points": opt["n_points"],
                        "opt_points_source": src,
                        "opt_cost": f"{opt['cost']:.3e}",
                        "opt_at_bound": ";".join(opt["at_bound"]),
                    })
                except (ValueError, FloatingPointError) as e:
                    row_out["opt_status"] = f"failed: {e}"

            # ---- distance measurement between marked ground points ----
            for pair in pairs_by_file.get(filename, []):
                p1 = (float(pair["u1"]) - half_w, float(pair["v1"]) - half_h)
                p2 = (float(pair["u2"]) - half_w, float(pair["v2"]) - half_h)
                true_d = pair.get("true_distance", "")
                true_d = float(true_d) if str(true_d).strip() else None
                drow = {"filename": filename, "label": pair.get("label", ""),
                        "u1": pair["u1"], "v1": pair["v1"], "u2": pair["u2"], "v2": pair["v2"],
                        "true_distance": "" if true_d is None else true_d}
                versions = [("direct", params["tilt_rad"], params["focal_length_used_px"],
                             params["height"])]
                if args.optimize and str(row_out.get("opt_status", "")).startswith(("ok", "not")):
                    versions.append(("opt", opt["tilt_rad"], opt["focal_length_px"], opt["height"]))
                for tag, phi_, f_, h_ in versions:
                    try:
                        est = ground_distance(p1, p2, phi_, f_, h_)
                        drow[f"{tag}_distance"] = round(est, 4)
                        if true_d is not None:
                            drow[f"{tag}_error"] = round(est - true_d, 4)
                            drow[f"{tag}_error_pct"] = round(100 * (est - true_d) / true_d, 2)
                    except ValueError as e:
                        drow[f"{tag}_distance"] = f"failed: {e}"
                distance_rows.append(drow)

            results.append(row_out)
        except (ValueError, ZeroDivisionError, FloatingPointError) as e:
            failed += 1
            results.append({
                "filename": filename,
                "status": "failed",
                "reason": str(e),
                "tilt_deg": "", "pan_deg": "", "focal_length_px": "", "focal_length_used_px": "",
                "camera_height": "", "circle_center_a": "", "circle_center_b": "",
                **({"opt_status": "skipped: direct method failed, no starting point"}
                   if args.optimize else {}),
            })

    joint_by_file = {}
    if args.joint:
        label = ("JOINT FIT WITH ROLL (extension)" if args.joint_roll else "JOINT FIT (paper's model)") + \
                " -- paper's equations (4)-(6), (10)-(15)"
        print("=" * 55)
        print(label)
        print("=" * 55)
        grows, frows = [], []
        for group, items in joint_frames.items():
            names = [it[0] for it in items]
            try:
                J = joint_fit([it[1] for it in items], [it[2] for it in items], args.radius,
                              with_roll=args.joint_roll)
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"  group '{group}' ({len(items)} frame(s)): failed -- {e}")
                grows.append({"group": group, "n_frames": len(items), "status": f"failed: {e}"})
                continue
            print(f"  group '{group}': {J['n_frames']} frame(s), "
                  f"{'converged' if J['converged'] else 'NOT converged'}, RMS {J['rms_px']:.2f} px")
            print(f"    focal length {J['focal_length_px']:9.2f} +/- {J['sd_focal_length_px']:.2f} px")
            print(f"    tilt         {np.degrees(J['tilt_rad']):9.3f} +/- {np.degrees(J['sd_tilt_rad']):.3f} deg")
            if args.joint_roll:
                print(f"    roll         {np.degrees(J['roll_rad']):9.3f} +/- {np.degrees(J['sd_roll_rad']):.3f} deg")
            print(f"    height       {J['height']:9.4f} +/- {J['sd_height']:.4f}  (units of --radius)")
            if J["tilt_at_limit"]:
                print("    [warn] tilt reached the paper model's lower limit (0.05 deg): the data suggest a "
                      "level or upward-looking camera,\n           which the paper's equations cannot represent; "
                      "treat this group's result as unreliable.")
            if J["n_frames"] == 1:
                print("    (one frame: identical to the single-frame method, nothing is shared)")
            grows.append({"group": group, "n_frames": J["n_frames"],
                          "status": "ok" if J["converged"] else "not converged",
                          "with_roll": args.joint_roll,
                          "focal_length_px": round(J["focal_length_px"], 2),
                          "sd_focal_length_px": round(J["sd_focal_length_px"], 2),
                          "tilt_deg": round(np.degrees(J["tilt_rad"]), 4),
                          "sd_tilt_deg": round(np.degrees(J["sd_tilt_rad"]), 4),
                          "roll_deg": round(np.degrees(J["roll_rad"]), 4),
                          "sd_roll_deg": round(np.degrees(J["sd_roll_rad"]), 4),
                          "camera_height": round(J["height"], 5),
                          "sd_camera_height": round(J["sd_height"], 5),
                          "rms_px": round(J["rms_px"], 3)})
            for name, (a, b), r in zip(names, J["ab"], J["frame_rms_px"]):
                joint_by_file[name] = J
                fwd = float(b) + J["height"] / np.tan(J["tilt_rad"])
                frows.append({"group": group, "filename": name,
                              "circle_center_b": round(float(b), 4),
                              "circle_center_a": round(float(a), 4),
                              "forward_from_below_camera": round(fwd, 4),
                              "rms_px": round(r, 3)})
        with open(output_dir / "joint_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["group", "n_frames", "status", "with_roll",
                                               "focal_length_px", "sd_focal_length_px", "tilt_deg",
                                               "sd_tilt_deg", "roll_deg", "sd_roll_deg",
                                               "camera_height", "sd_camera_height", "rms_px"],
                               restval="")
            w.writeheader()
            w.writerows(grows)
        with open(output_dir / "joint_frames.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["group", "filename", "circle_center_a", "circle_center_b",
                                               "forward_from_below_camera", "rms_px"])
            w.writeheader()
            w.writerows(frows)
        print(f"  written: {output_dir / 'joint_results.csv'}, {output_dir / 'joint_frames.csv'}")
        print("  (+/- values are 1 standard deviation from the fit's residuals)")

        # marked distances with the joint camera values
        for drow in distance_rows:
            J = joint_by_file.get(drow["filename"])
            if J is None:
                continue
            p1 = (float(drow["u1"]) - half_w, float(drow["v1"]) - half_h)
            p2 = (float(drow["u2"]) - half_w, float(drow["v2"]) - half_h)
            try:
                est = ground_distance_joint(p1, p2, J["tilt_rad"], J["focal_length_px"], J["height"],
                                            roll=J["roll_rad"])
                drow["joint_distance"] = round(est, 4)
                true_d = drow.get("true_distance")
                if isinstance(true_d, float):
                    drow["joint_error"] = round(est - true_d, 4)
                    drow["joint_error_pct"] = round(100 * (est - true_d) / true_d, 2)
            except ValueError as e:
                drow["joint_distance"] = f"failed: {e}"

    csv_path = output_dir / "calibration_results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["filename", "status", "reason", "tilt_deg", "pan_deg",
                      "focal_length_px", "focal_length_used_px", "camera_height",
                      "circle_center_a", "circle_center_b", "forward_from_below_camera", "method"]
        if args.optimize:
            fieldnames += ["opt_status", "opt_tilt_deg", "opt_pan_deg", "opt_focal_length_px",
                           "opt_camera_height", "opt_circle_center_a", "opt_circle_center_b",
                           "opt_forward_from_below_camera",
                           "opt_n_points", "opt_points_source", "opt_cost", "opt_at_bound"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(results)

    print("=" * 55)
    print("CAMERA CALIBRATION SUMMARY")
    print("=" * 55)
    print(f"Ellipses processed:   {succeeded + failed}")
    print(f"Successful:           {succeeded}")
    print(f"Failed (bad domain):  {failed}  (see 'reason' column in CSV for why)")
    if args.focal_length is not None:
        print(f"[info] Using known focal length ({args.focal_length:.2f}px) for height/circle-centre "
              f"calculations. 'focal_length_px' below is still the value RECOVERED from each "
              f"ellipse (unaffected by this input) -- see 'focal_length_used_px' in the CSV for "
              f"what was actually used.")

    if succeeded > 0:
        ok_rows = [r for r in results if r["status"] == "ok"]
        for key, label in [("tilt_deg", "Tilt (deg)"), ("pan_deg", "Pan (deg)"),
                            ("focal_length_px", "Focal length (px)"),
                            ("camera_height", f"Camera height ({args.radius}-unit scale)")]:
            values = [r[key] for r in ok_rows]
            print(f"{label:28s} mean={np.mean(values):10.3f}  std={np.std(values):10.3f}  "
                  f"min={min(values):10.3f}  max={max(values):10.3f}")

        f_values = [r["focal_length_px"] for r in ok_rows]
        f_std_ratio = np.std(f_values) / abs(np.mean(f_values)) if np.mean(f_values) != 0 else float("inf")
        print()
        if f_std_ratio > 0.3:
            print(f"[caution] focal length varies a lot across detections (std/mean = {f_std_ratio:.2f}).")
            print("          If these images all come from the same physical camera, focal length")
            print("          should be roughly CONSTANT across shots -- this much spread suggests")
            print("          poor ellipse fits, an incorrect --radius assumption, or that your")
            print("          images actually come from different cameras/sources. Treat individual")
            print("          per-image results with caution rather than as a reliable calibration.")
        else:
            print(f"Focal length is reasonably consistent across detections (std/mean = {f_std_ratio:.2f}),")
            print("which is a good sign the underlying assumptions (single camera, correct radius) hold.")

    if args.optimize:
        opt_rows = [r for r in results if str(r.get("opt_status", "")).startswith(("ok", "not"))]
        if opt_rows:
            print()
            print("OPTIMISED (option two):")
            for key, label in [("opt_tilt_deg", "Tilt (deg)"), ("opt_pan_deg", "Pan (deg)"),
                                ("opt_focal_length_px", "Focal length (px)"),
                                ("opt_camera_height", f"Camera height ({args.radius}-unit scale)")]:
                values = [r[key] for r in opt_rows]
                print(f"{label:28s} mean={np.mean(values):10.3f}  std={np.std(values):10.3f}  "
                      f"min={min(values):10.3f}  max={max(values):10.3f}")
            pinned = [r["filename"] for r in opt_rows if r.get("opt_at_bound")]
            if pinned:
                print(f"[caution] {len(pinned)} image(s) finished with a parameter pressed against "
                      f"its +/-50% bound -- the optimiser wanted to go further than the paper's "
                      f"method allows, so that result is not a true optimum. See 'opt_at_bound'.")

    if args.distances:
        print()
        print("DISTANCE MEASUREMENT (marked ground points):")
        if args.focal_length is not None and not args.optimize:
            print("  [caution] --focal-length without --optimize: the direct method mixes the known "
                  "focal length with a tilt/height\n            from the same ellipse, which can scale "
                  "all distances by several percent. Re-run with --optimize.")
        dcsv = output_dir / "distance_results.csv"
        dfields = ["filename", "label", "u1", "v1", "u2", "v2", "true_distance",
                   "direct_distance", "direct_error", "direct_error_pct"]
        if args.optimize:
            dfields += ["opt_distance", "opt_error", "opt_error_pct"]
        if args.joint:
            dfields += ["joint_distance", "joint_error", "joint_error_pct"]
        with open(dcsv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=dfields, restval="")
            w.writeheader()
            w.writerows(distance_rows)
        n_pairs = sum(len(v) for v in pairs_by_file.values())
        print(f"  {len(distance_rows)} of {n_pairs} pair(s) measured "
              f"(pairs in images whose calibration failed are skipped)")
        for tag, name in [("direct", "Direct method"), ("opt", "Optimised"), ("joint", "Joint fit")]:
            errs = [r[f"{tag}_error"] for r in distance_rows if isinstance(r.get(f"{tag}_error"), float)]
            pcts = [r[f"{tag}_error_pct"] for r in distance_rows
                    if isinstance(r.get(f"{tag}_error_pct"), float)]
            if errs:
                e = np.array(errs)
                print(f"  {name:14s} n={len(e):3d}  mean abs error={np.mean(np.abs(e)):.4f}  "
                      f"RMSE={np.sqrt(np.mean(e ** 2)):.4f}  bias={np.mean(e):+.4f}  "
                      f"mean abs % error={np.mean(np.abs(pcts)):.2f}%   (units of --radius)")
        if not any(isinstance(r.get("direct_error"), float) for r in distance_rows):
            print("  (no true_distance values given, so only the estimates are reported)")
        print(f"  per-pair results: {dcsv}")

    print("=" * 55)
    print()
    print(f"Full per-image results: {csv_path}")
    print()


if __name__ == "__main__":
    main()
