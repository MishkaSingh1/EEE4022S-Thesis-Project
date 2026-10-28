#!/usr/bin/env python3
"""
Ground truth for camera HEIGHT, TILT and ROLL relative to the ROAD, from KITTI raw Velodyne scans.

For each frame:
  1. Read the Velodyne scan (velodyne_points/data/<frame>.bin: float32 x, y, z, reflectance).
  2. Move the points into the rectified camera's coordinates (x right, y down, z forward, camera at
     the origin) with the KITTI calibration:  p_cam = offset_k * R_rect_00 * T_velo->cam * p_velo
     (same chain as kitti_oxts_pose.py, without the IMU step).
  3. Keep candidate road points: in front of the camera, inside --forward and --side (or --region),
     and below the camera.
  4. RANSAC plane fit, only accepting planes whose normal is within --max-normal-angle of the
     camera's "up" direction (so walls and car sides are never chosen), then a least-squares refit
     on the inliers (smallest-eigenvector of their covariance).
  5. With the plane's unit normal n pointing UP (towards the camera) and n.p + d = 0:
        height   = |d|                       distance from the camera centre to the road
        tilt     = asin(-n_z)                optical axis below the road plane (+ = looking down)
        roll     = asin( n_x)                image x axis above the road plane (+ = right side up)
     Same definitions as kitti_oxts_pose.py, but relative to the ROAD instead of gravity.

OUTPUT: road_plane.csv, one row per frame, plus mean/std over the frames printed at the end.

USAGE
    python kitti_road_plane.py --velodyne <drive>/velodyne_points --calib-dir <date folder> \
        --camera 2 --frames 28 29 30 31 --output road_plane.csv
    # only the road around the manhole (camera coordinates: x right, z forward, metres):
    python kitti_road_plane.py ... --region 3 8 8 18
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def read_calib_file(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            out[key.strip()] = np.array([float(v) for v in val.split()])
        except ValueError:
            pass
    return out


def velo_to_camera(calib_dir, camera):
    """4x4 transform from Velodyne coordinates to rectified camera `camera` coordinates."""
    d = Path(calib_dir)
    v2c = read_calib_file(d / "calib_velo_to_cam.txt")
    c2c = read_calib_file(d / "calib_cam_to_cam.txt")
    T = np.eye(4)
    T[:3, :3] = v2c["R"].reshape(3, 3)
    T[:3, 3] = v2c["T"]
    R_rect = np.eye(4)
    R_rect[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    P = c2c[f"P_rect_0{camera}"].reshape(3, 4)
    offset = np.eye(4)
    offset[:3, 3] = np.linalg.solve(P[:, :3], P[:, 3])
    return offset @ R_rect @ T, P[:, :3]


def read_scan(path):
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % 4:
        sys.exit(f"[error] {path}: size is not a multiple of 4 floats -- not a KITTI Velodyne scan?")
    return pts.reshape(-1, 4)[:, :3].astype(np.float64)


def fit_plane_ransac(P, thresh, max_angle_deg, iters, rng):
    """RANSAC plane among points P (N,3) in camera coords; normal must be near camera-up (0,-1,0)."""
    up = np.array([0.0, -1.0, 0.0])
    cos_max = np.cos(np.radians(max_angle_deg))
    best = None
    n_pts = len(P)
    for _ in range(iters):
        i = rng.choice(n_pts, 3, replace=False)
        n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n @ up < 0:
            n = -n
        if n @ up < cos_max:                      # not road-like: wall, car side, etc.
            continue
        d = -n @ P[i[0]]
        inl = np.abs(P @ n + d) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        raise ValueError("no road-like plane found")
    # least-squares refit on the inliers, then one more inlier pass with the refined plane
    for _ in range(2):
        Q = P[best]
        c = Q.mean(axis=0)
        w, v = np.linalg.eigh(np.cov((Q - c).T))
        n = v[:, 0]
        if n @ up < 0:
            n = -n
        d = -n @ c
        best = np.abs(P @ n + d) < thresh
    res = P[best] @ n + d
    return n, d, best, float(np.sqrt(np.mean(res ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--velodyne", required=True, help="the drive's velodyne_points folder (contains data/)")
    ap.add_argument("--calib-dir", required=True, help="date folder with calib_velo_to_cam.txt and calib_cam_to_cam.txt")
    ap.add_argument("--camera", type=int, default=2, choices=[0, 1, 2, 3], help="rectified camera (image_0X). Default 2")
    ap.add_argument("--frames", type=int, nargs="+", required=True, help="frame numbers, e.g. 28 29 30 31")
    ap.add_argument("--output", required=True, help="CSV to write")
    ap.add_argument("--forward", type=float, nargs=2, default=[4.0, 25.0], metavar=("MIN", "MAX"),
                    help="forward range (m, camera z) of candidate road points. Default 4 25")
    ap.add_argument("--side", type=float, default=6.0,
                    help="max sideways distance (m, |camera x|) of candidate road points. Default 6")
    ap.add_argument("--region", type=float, nargs=4, default=None, metavar=("XMIN", "XMAX", "ZMIN", "ZMAX"),
                    help="OPTIONAL: fit only this patch (camera x right, z forward, metres), e.g. around the "
                         "manhole. Overrides --forward/--side")
    ap.add_argument("--threshold", type=float, default=0.05, help="RANSAC inlier distance (m). Default 0.05")
    ap.add_argument("--max-normal-angle", type=float, default=20.0,
                    help="reject planes whose normal is further than this from camera-up (deg). Default 20")
    ap.add_argument("--iterations", type=int, default=500, help="RANSAC iterations. Default 500")
    args = ap.parse_args()

    vdir = Path(args.velodyne)
    data_dir = vdir / "data" if (vdir / "data").is_dir() else vdir
    T, K = velo_to_camera(args.calib_dir, args.camera)
    print(f"[info] camera {args.camera}: focal length {K[0, 0]:.3f} px, principal point ({K[0, 2]:.3f}, {K[1, 2]:.3f})")
    rng = np.random.default_rng(0)

    rows = []
    for fid in args.frames:
        path = data_dir / f"{fid:010d}.bin"
        if not path.exists():
            print(f"[skip] frame {fid}: {path} not found")
            continue
        pv = read_scan(path)
        pc = (T @ np.column_stack([pv, np.ones(len(pv))]).T).T[:, :3]
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        if args.region:
            x0, x1, z0, z1 = args.region
            keep = (x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)
        else:
            keep = (z >= args.forward[0]) & (z <= args.forward[1]) & (np.abs(x) <= args.side)
        keep &= y > 0.3                               # below the camera (y points down)
        P = pc[keep]
        if len(P) < 50:
            print(f"[skip] frame {fid}: only {len(P)} candidate road points in the chosen area")
            continue
        try:
            n, d, inl, rms = fit_plane_ransac(P, args.threshold, args.max_normal_angle, args.iterations, rng)
        except ValueError as e:
            print(f"[skip] frame {fid}: {e}")
            continue
        height = abs(d)
        tilt = np.degrees(np.arcsin(np.clip(-n[2], -1, 1)))
        roll = np.degrees(np.arcsin(np.clip(n[0], -1, 1)))
        rows.append({"frame": fid, "candidate_points": len(P), "inliers": int(inl.sum()),
                     "inlier_fraction": round(inl.mean(), 3), "plane_rms_m": round(rms, 4),
                     "camera_height_m": round(height, 4), "tilt_deg": round(tilt, 4), "roll_deg": round(roll, 4),
                     "normal_x": round(n[0], 6), "normal_y": round(n[1], 6), "normal_z": round(n[2], 6)})
        print(f"[ok] frame {fid}: height {height:.4f} m  tilt {tilt:+.3f} deg  roll {roll:+.3f} deg   "
              f"({inl.sum()}/{len(P)} road points, plane RMS {100 * rms:.1f} cm)")

    if not rows:
        sys.exit("[error] no frame could be fitted")
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for key, name, unit in [("camera_height_m", "height", " m"), ("tilt_deg", "tilt", " deg"), ("roll_deg", "roll", " deg")]:
        v = np.array([r[key] for r in rows])
        print(f"{name:7s} mean {v.mean():+.4f}{unit}   std {v.std():.4f}{unit}   over {len(v)} frame(s)")
    print(f"written: {args.output}")
    print("tilt/roll/height are relative to the ROAD PLANE fitted in the chosen area (see --region).")


if __name__ == "__main__":
    main()
