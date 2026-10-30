#!/usr/bin/env python3
"""
Synthetic KITTI-like manhole images with EXACT ground-truth ellipses.

Purpose: test the ellipse-fitting scripts. Every image contains one round manhole
cover lying on a flat road, seen by a KITTI-like pinhole camera. The true image ellipse is
computed analytically (no fitting, no drawing), so the fitter's output can be compared with it.

Outputs (in --output):
    images/000000.png ...      grey road images, 3 channels (KITTI size by default 1242x375)
    detections.csv             filename,x1,y1,x2,y2,confidence   (fake detector boxes, same
                               columns the fitting scripts read)
    ground_truth.csv           one row per manhole, row i  <->  detection_index i
    preview/000000.png         (only with --preview) true ellipse in green, box in red

    random    : every image is an independent random manhole        (default)

Usage examples:
    python generate_kitti_manholes.py --output syn_clean --num-images 100 --preview
    python generate_kitti_manholes.py --output syn_hard --num-images 100 --occlude 0.3 \
           --shadow 0.6 --paint --noise 4 --blur 1.0 --jpeg 60
    python generate_kitti_manholes.py --output syn_seq --mode sequence --num-sequences 10 --frames 6
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# A. Geometry: camera, ground plane, exact ellipse of a ground circle
# =============================================================================
def ground_to_image_H(f, px, py, h, tilt_deg, roll_deg=0.0):
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    M = np.array([[1.0, 0.0, 0.0], [0.0, -s, h * c], [0.0, c, h * s]])
    r = math.radians(roll_deg)
    Rz = np.array([[math.cos(r), -math.sin(r), 0.0], [math.sin(r), math.cos(r), 0.0], [0.0, 0.0, 1.0]])
    K = np.array([[f, 0.0, px], [0.0, f, py], [0.0, 0.0, 1.0]])
    return K @ Rz @ M


def project(H, X, Y):
    P = H @ np.vstack([X, Y, np.ones_like(X)])
    return P[0] / P[2], P[1] / P[2], P[2]


def circle_ellipse(H, X0, Y0, R):
    """EXACT image ellipse of the ground circle (centre X0,Y0, radius R).
    Image conic = H^-T C0 H^-1 where C0 is the circle's conic matrix on the ground.
    Returns dict(cx, cy, major, minor, angle_deg) (major/minor are FULL axis lengths; angle is the
    direction of the major axis in image coordinates, x right / y down, in [0,180)) or None if the
    circle is not a proper ellipse in front of the camera."""
    t = np.linspace(0.0, 2 * math.pi, 360, endpoint=False)
    _, _, depth = project(H, X0 + R * np.cos(t), Y0 + R * np.sin(t))
    if np.any(depth <= 1e-6):
        return None
    C0 = np.array([[1.0, 0.0, -X0], [0.0, 1.0, -Y0], [-X0, -Y0, X0 * X0 + Y0 * Y0 - R * R]])
    Hi = np.linalg.inv(H)
    C = Hi.T @ C0 @ Hi
    C = 0.5 * (C + C.T)
    Q, b = C[:2, :2], C[:2, 2]
    if np.linalg.det(Q) <= 0:
        return None
    ctr = -np.linalg.solve(Q, b)                        
    Fc = C[2, 2] + b @ ctr                            
    w, V = np.linalg.eigh(Q)
    semi2 = -Fc / w
    if np.any(semi2 <= 0):
        return None
    semi = np.sqrt(semi2)
    i = int(np.argmax(semi))
    ang = math.degrees(math.atan2(V[1, i], V[0, i])) % 180.0
    return dict(cx=float(ctr[0]), cy=float(ctr[1]), major=float(2 * semi[i]),
                minor=float(2 * semi[1 - i]), angle=ang)


def ellipse_points(e, n=720):
    """Points on the ellipse dict from circle_ellipse (for drawing / visibility)."""
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    th = math.radians(e["angle"])
    a, b = e["major"] / 2, e["minor"] / 2
    x = a * np.cos(t)
    y = b * np.sin(t)
    return e["cx"] + x * math.cos(th) - y * math.sin(th), e["cy"] + x * math.sin(th) + y * math.cos(th)


def ellipse_halfextent(e):
    th, a, b = math.radians(e["angle"]), e["major"] / 2, e["minor"] / 2
    return (math.sqrt((a * math.cos(th)) ** 2 + (b * math.sin(th)) ** 2),
            math.sqrt((a * math.sin(th)) ** 2 + (b * math.cos(th)) ** 2))


# =============================================================================
# B. Anti-aliased polygon coverage (supersampled, sub-pixel exact edges)
# =============================================================================
def coverage(poly_u, poly_v, W, H, ss=8):
    x0 = max(0, int(math.floor(poly_u.min())) - 2)
    y0 = max(0, int(math.floor(poly_v.min())) - 2)
    x1 = min(W, int(math.ceil(poly_u.max())) + 3)
    y1 = min(H, int(math.ceil(poly_v.max())) + 3)
    if x1 <= x0 or y1 <= y0:
        return None, 0, 0
    w, h = x1 - x0, y1 - y0
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    # image coordinate p -> hi-res index: (p - crop_origin + 0.5) * ss - 0.5   (pixel centre convention)
    hu = (poly_u - x0 + 0.5) * ss - 0.5
    hv = (poly_v - y0 + 0.5) * ss - 0.5
    pts = np.round(np.column_stack([hu, hv]) * 16).astype(np.int32)    # 4 fractional bits
    cv2.fillPoly(canvas, [pts], 255, lineType=cv2.LINE_8, shift=4)
    alpha = canvas.reshape(h, ss, w, ss).mean(axis=(1, 3)) / 255.0
    return alpha.astype(np.float32), x0, y0


def paste_max(full, alpha, x0, y0):
    if alpha is None:
        return
    h, w = alpha.shape
    full[y0:y0 + h, x0:x0 + w] = np.maximum(full[y0:y0 + h, x0:x0 + w], alpha)


def circle_polygon(H, X0, Y0, R, n=720):
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    u, v, _ = project(H, X0 + R * np.cos(t), Y0 + R * np.sin(t))
    return u, v


def image_conic(H, X0, Y0, R):
    C0 = np.array([[1.0, 0.0, -X0], [0.0, 1.0, -Y0], [-X0, -Y0, X0 * X0 + Y0 * Y0 - R * R]])
    Hi = np.linalg.inv(H)
    C = Hi.T @ C0 @ Hi
    C = 0.5 * (C + C.T)
    ctr = -np.linalg.solve(C[:2, :2], C[:2, 2])
    v = np.array([ctr[0], ctr[1], 1.0])
    return -C if v @ C @ v > 0 else C


def conic_coverage(C, u, v, W, H, ss=8):
    x0 = max(0, int(math.floor(u.min())) - 2)
    y0 = max(0, int(math.floor(v.min())) - 2)
    x1 = min(W, int(math.ceil(u.max())) + 3)
    y1 = min(H, int(math.ceil(v.max())) + 3)
    if x1 <= x0 or y1 <= y0:
        return None, 0, 0
    off = -0.5 + (np.arange(ss) + 0.5) / ss                     # sample positions inside a pixel
    xs = (np.arange(x0, x1)[:, None] + off[None, :]).ravel()
    ys = (np.arange(y0, y1)[:, None] + off[None, :]).ravel()
    X, Y = np.meshgrid(xs, ys)
    q = (C[0, 0] * X * X + 2 * C[0, 1] * X * Y + C[1, 1] * Y * Y + 2 * C[0, 2] * X + 2 * C[1, 2] * Y + C[2, 2])
    inside = (q < 0).astype(np.float32)
    alpha = inside.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))
    return alpha, x0, y0


# =============================================================================
# C. Road "world": features fixed to the ground (so they stay put when the camera moves)
# =============================================================================
def smooth_noise(rng, H, W, cell):
    small = rng.normal(size=(H // cell + 3, W // cell + 3)).astype(np.float32)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


class World:
    """Everything that belongs to the road and must be identical in every frame of a sequence."""

    def __init__(self, rng, X0, Y0w, R, args, frng=None):
        frng = frng if frng is not None else rng      # effect parameters (paint/shadow) use their own stream
        self.g0 = rng.uniform(*args.asphalt)
        # soft brighter/darker patches (repairs, wear, oil) in ground coordinates
        self.patches = [(rng.uniform(-8, 8), rng.uniform(0, 40), rng.uniform(0.4, 2.5),
                         rng.uniform(0.4, 4.0), rng.uniform(-18, 18)) for _ in range(args.patches)]
        # contrast of the cover relative to the road (sign chooses edge polarity)
        mag = rng.uniform(*args.contrast)
        sign = {"dark": -1, "light": 1}.get(args.cover_tone, rng.choice([-1, 1]))
        self.cover_delta = sign * mag
        self.cover_pattern = args.cover_pattern if args.cover_pattern != "random" \
            else rng.choice(["flat", "ribs", "grid"])
        self.pattern_amp = args.pattern_amp
        self.k_ribs = int(rng.integers(3, 7))
        self.grid_angle = rng.uniform(0, math.pi)
        self.grid_period = rng.uniform(0.04, 0.08)
        self.X0, self.Y0w, self.R = X0, Y0w, R
        self.rim = args.rim
        self.rim_dark = args.rim_contrast
        # one painted stripe next to the cover (optional)
        self.paint = None
        if args.paint:
            side = frng.choice([-1, 1])
            self.paint = dict(xs=X0 + side * (R + frng.uniform(0.04, 0.35) + args.rim * R),
                              half=frng.uniform(0.05, 0.12), amp=frng.uniform(40, 90),
                              dashed=bool(frng.random() < 0.5), period=frng.uniform(3.0, 6.0),
                              phase=frng.uniform(0, 6))
        # soft cast shadow edge (a straight line on the ground) cutting near the cover
        self.shadow = None
        if args.shadow > 0:
            ang = frng.uniform(0, 2 * math.pi)
            n = np.array([math.cos(ang), math.sin(ang)])
            off = frng.uniform(-0.8, 0.8) * R
            self.shadow = dict(n=n, d=n @ np.array([X0, Y0w]) + off, soft=frng.uniform(0.03, 0.12),
                               factor=1.0 - args.shadow)

    def asphalt(self, X, Yw):
        g = np.full(X.shape, self.g0, np.float32)
        for (px, py, sx, sy, amp) in self.patches:
            r2 = ((X - px) / sx) ** 2 + ((Yw - py) / sy) ** 2
            g += amp * np.exp(-r2).astype(np.float32)
        if self.paint:
            p = self.paint
            m = np.clip(0.5 + (p["half"] - np.abs(X - p["xs"])) / 0.02, 0, 1)
            if p["dashed"]:
                ph = (Yw + p["phase"]) % p["period"]
                m = m * np.clip(0.5 + (0.5 * p["period"] - ph) / 0.05, 0, 1)
            g = g * (1 - m) + (self.g0 + p["amp"]) * m
        return g

    def cover_tone(self, X, Yw):
        """Grey level of the cover surface (pattern is defined on the cover, so it stays attached)."""
        dx, dy = X - self.X0, Yw - self.Y0w
        t = np.full(X.shape, self.g0 + self.cover_delta, np.float32)
        if self.cover_pattern == "ribs":
            rho = np.sqrt(dx * dx + dy * dy) / self.R
            t += self.pattern_amp * np.sin(2 * math.pi * self.k_ribs * rho).astype(np.float32)
        elif self.cover_pattern == "grid":
            c, s = math.cos(self.grid_angle), math.sin(self.grid_angle)
            ux, uy = (c * dx + s * dy) / self.grid_period, (-s * dx + c * dy) / self.grid_period
            t += self.pattern_amp * (np.sin(2 * math.pi * ux) * np.sin(2 * math.pi * uy)).astype(np.float32)
        return t

    def shadow_factor(self, X, Yw):
        if not self.shadow:
            return None
        s = self.shadow
        dist = s["n"][0] * X + s["n"][1] * Yw - s["d"]
        k = np.clip(0.5 + dist / s["soft"], 0, 1)               # 0 = lit side, 1 = shadowed side
        return (1 - k * (1 - s["factor"])).astype(np.float32)


# =============================================================================
# D. Occluder: straight-edged blob that hides a chosen fraction of the rim
# =============================================================================
def occluder_polygon(rng, e, frac):
    """Quadrilateral whose straight edge cuts the ellipse boundary so exactly `frac` of the
    boundary samples lie behind it."""
    bu, bv = ellipse_points(e)
    ang = rng.uniform(0, 2 * math.pi)
    n = np.array([math.cos(ang), math.sin(ang)])
    tau = np.array([-n[1], n[0]])
    s = (bu - e["cx"]) * n[0] + (bv - e["cy"]) * n[1]
    t = np.quantile(s, 1.0 - frac)
    hidden = s > t
    lat = np.abs((bu - e["cx"]) * tau[0] + (bv - e["cy"]) * tau[1])
    half = (lat[hidden].max() if hidden.any() else 1.0) + rng.uniform(3, 15)
    depth = max(e["major"], 20.0) * rng.uniform(0.8, 1.6)
    c0 = np.array([e["cx"], e["cy"]])
    corners = [c0 + n * t - tau * half, c0 + n * t + tau * half,
               c0 + n * (t + depth) + tau * (half + rng.uniform(0, 10)),
               c0 + n * (t + depth) - tau * (half + rng.uniform(0, 10))]
    corners = np.array(corners)
    return corners[:, 0], corners[:, 1]


# =============================================================================
# D2. Motion blur: a straight streak along the direction the image moves
# =============================================================================
def line_kernel(length, dx, dy):
    """Normalised convolution kernel for a straight streak of `length` pixels pointing along (dx, dy).
    The streak is sampled finely and every sample is shared bilinearly between the 4 nearest kernel cells,
    so the streak can have any sub-pixel length and any angle. The kernel sums to 1 (no brightness change)."""
    half = int(math.ceil(length / 2.0)) + 2
    size = 2 * half + 1
    k = np.zeros((size, size), np.float32)
    n = max(8, int(length * 16))
    for t in np.linspace(-length / 2.0, length / 2.0, n):
        x, y = half + t * dx, half + t * dy
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        k[y0, x0] += (1 - fx) * (1 - fy)
        k[y0, x0 + 1] += fx * (1 - fy)
        k[y0 + 1, x0] += (1 - fx) * fy
        k[y0 + 1, x0 + 1] += fx * fy
    return k / k.sum()


def image_motion_direction(Hm, X0, Y0_rel):
    """Unit vector (du, dv) of how the manhole centre moves in the image when the camera drives forward
    (found by projecting the centre at two distances 5 cm apart)."""
    u1, v1, _ = project(Hm, np.array([X0]), np.array([Y0_rel]))
    u0, v0, _ = project(Hm, np.array([X0]), np.array([Y0_rel - 0.05]))
    d = np.array([float(u0[0] - u1[0]), float(v0[0] - v1[0])])
    n = np.hypot(*d)
    return (1.0, 0.0) if n < 1e-9 else (d[0] / n, d[1] / n)


# =============================================================================
# E. Render one frame
# =============================================================================
def render_frame(rng, args, world, cam, manholes, y_cam_world, frng=None):
    """manholes: list of dicts with X0, Yw (world Y), R, ellipse (exact), occ_frac.
    Returns gray image uint8 HxW and, per manhole, the visible fraction of its rim."""
    frng = frng if frng is not None else rng      # occluder randomness has its own stream
    W, Hh = args.width, args.height_px
    Hm = ground_to_image_H(**cam)
    Hinv = np.linalg.inv(Hm)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(Hh, dtype=np.float32))
    g = np.einsum("ij,jhw->ihw", Hinv.astype(np.float32), np.stack([uu, vv, np.ones_like(uu)]))
    X, Y = g[0] / g[2], g[1] / g[2]
    ground_ok = (Y > 0.3) & (Y < 150.0)
    Yw = Y + y_cam_world

    # --- road + sky ---
    road = world.asphalt(X, Yw) + args.lowfreq * smooth_noise(rng, Hh, W, 48)
    road += args.grain * cv2.GaussianBlur(rng.normal(size=(Hh, W)).astype(np.float32), (0, 0), 0.7) * 1.8
    sky = np.linspace(185, 150, Hh, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    img = np.where(ground_ok, road, sky)

    visible = []
    # --- rim ring (frame around the cover) and cover, supersampled alpha ---
    for m in manholes:
        a_cov = np.zeros((Hh, W), np.float32)
        a_rim = np.zeros((Hh, W), np.float32)
        u, v = circle_polygon(Hm, m["X0"], m["Yw"] - y_cam_world, m["R"])
        paste_max(a_cov, *conic_coverage(image_conic(Hm, m["X0"], m["Yw"] - y_cam_world, m["R"]), u, v, W, Hh))
        if world.rim > 0:
            ur, vr = circle_polygon(Hm, m["X0"], m["Yw"] - y_cam_world, m["R"] * (1 + world.rim))
            paste_max(a_rim, *conic_coverage(image_conic(Hm, m["X0"], m["Yw"] - y_cam_world,
                                                         m["R"] * (1 + world.rim)), ur, vr, W, Hh))
            img = img * (1 - a_rim) + (img - world.rim_dark) * a_rim
        cover = world.cover_tone(X, Yw) + args.cover_grain * cv2.GaussianBlur(
            rng.normal(size=(Hh, W)).astype(np.float32), (0, 0), 0.7) * 1.8
        img = img * (1 - a_cov) + cover * a_cov
        m["_alpha"] = a_cov

    # --- cast shadow over everything on the ground ---
    sf = world.shadow_factor(X, Yw)
    if sf is not None:
        img = np.where(ground_ok, img * sf, img)

    # --- occluders (image space) ---
    occ_mask = np.zeros((Hh, W), np.float32)
    for m in manholes:
        if m["occ_frac"] > 0:
            ou, ov = occluder_polygon(frng, m["ellipse"], m["occ_frac"])
            a_occ, x0, y0 = coverage(ou, ov, W, Hh)
            if a_occ is not None:
                gray = frng.uniform(*args.occluder_gray)
                h2, w2 = a_occ.shape
                tex = gray + 4.0 * cv2.GaussianBlur(frng.normal(size=(h2, w2)).astype(np.float32), (0, 0), 1.0)
                sl = (slice(y0, y0 + h2), slice(x0, x0 + w2))
                img[sl] = img[sl] * (1 - a_occ) + tex * a_occ
                occ_mask[sl] = np.maximum(occ_mask[sl], a_occ)

    # --- motion blur: one streak along the manhole's image motion (the manhole is small, so one direction
    #     and length for the whole picture is a good local model). Applied before the optical blur; the two
    #     blurs are both convolutions, so their order does not matter.
    if args.motion_blur > 0 and manholes:
        m0 = manholes[0]
        ddx, ddy = image_motion_direction(Hm, m0["X0"], m0["Yw"] - y_cam_world)
        img = cv2.filter2D(img, -1, line_kernel(args.motion_blur, ddx, ddy), borderType=cv2.BORDER_REPLICATE)

    # --- camera: blur (PSF), sensor noise, clipping ---
    if args.blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), args.blur)
    if args.noise > 0:
        img = img + args.noise * rng.standard_normal(img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if args.jpeg > 0:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg)])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    # --- visible fraction of each rim (inside image and not under an occluder) ---
    for m in manholes:
        bu, bv = ellipse_points(m["ellipse"])
        inside = (bu >= 0) & (bu <= W - 1) & (bv >= 0) & (bv <= Hh - 1)
        iu, iv = np.clip(np.round(bu).astype(int), 0, W - 1), np.clip(np.round(bv).astype(int), 0, Hh - 1)
        vis = inside & (occ_mask[iv, iu] < 0.5)
        visible.append(float(vis.mean()))
    return img, visible


# =============================================================================
# F. Sampling manholes and detector boxes
# =============================================================================
def sample_manhole(rng, args, tilt_list, h_list, frames_cam):
    """Pick (X0, Y0 below-camera at frame 0, R) so the ellipse is valid & inside the image in
    EVERY frame (frames_cam = list of (cam dict, y_cam_world)). Returns None if 200 tries fail."""
    for _ in range(200):
        Y0 = math.exp(rng.uniform(math.log(args.dist[0]), math.log(args.dist[1])))
        X0 = rng.uniform(-args.max_lateral, args.max_lateral)
        R = rng.uniform(*args.radius)
        ok, ells = True, []
        for cam, ycw in frames_cam:
            Hm = ground_to_image_H(**cam)
            e = circle_ellipse(Hm, X0, Y0 - ycw, R)
            if e is None or e["minor"] < args.min_minor:
                ok = False
                break
            hx, hy = ellipse_halfextent(e)
            margin = 2 if not args.allow_cut else -10 ** 6
            if (e["cx"] - hx < margin or e["cx"] + hx > args.width - 1 - margin or
                    e["cy"] - hy < margin or e["cy"] + hy > args.height_px - 1 - margin):
                ok = False
                break
            ells.append(e)
        if ok:
            return dict(X0=X0, Yw=Y0, R=R), ells
    return None


def make_box(rng, args, e):
    hx, hy = ellipse_halfextent(e)
    w, h = 2 * hx * args.box_scale_w, 2 * hy * args.box_scale_h
    x1, x2 = e["cx"] - w / 2, e["cx"] + w / 2
    y1, y2 = e["cy"] - h / 2, e["cy"] + h / 2
    j = args.box_jitter
    x1 += rng.uniform(-j, j) * w
    x2 += rng.uniform(-j, j) * w
    y1 += rng.uniform(-j, j) * h
    y2 += rng.uniform(-j, j) * h
    return (max(0.0, x1), max(0.0, y1), min(args.width - 1.0, x2), min(args.height_px - 1.0, y2))


# =============================================================================
# G. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["random", "sequence"], default="random")
    ap.add_argument("--num-images", type=int, default=50, help="random mode: number of images")
    ap.add_argument("--per-image", type=int, default=1, help="random mode: manholes per image")
    ap.add_argument("--num-sequences", type=int, default=5, help="sequence mode")
    ap.add_argument("--frames", type=int, default=6, help="sequence mode: frames per sequence")
    ap.add_argument("--step", type=float, default=0.8, help="sequence mode: camera advance per frame (m)")
    ap.add_argument("--pitch-jitter", type=float, default=0.1, help="sequence mode: per-frame tilt noise (deg)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene-ids", type=int, nargs="+",
                    help="random mode: build exactly these scene ids. A scene id fixes geometry, road, cover, noise "
                         "pattern and box jitter (via --seed), so the same id with different effect options gives a PAIRED image.")
    # camera (KITTI defaults)
    ap.add_argument("--width", type=int, default=1242)
    ap.add_argument("--height-px", type=int, default=375)
    ap.add_argument("--focal", type=float, default=721.5377)
    ap.add_argument("--principal", type=float, nargs=2, default=[609.5593, 172.854])
    ap.add_argument("--cam-height", type=float, default=1.65, help="camera height above road (m)")
    ap.add_argument("--tilt", type=float, nargs=2, default=[0.5, 2.5], help="camera pitch-down range (deg)")
    ap.add_argument("--roll", type=float, default=0.0, help="camera roll (deg)")
    # manhole placement
    ap.add_argument("--dist", type=float, nargs=2, default=[6.0, 12.0],
                    help="forward distance range from the point below the camera (m)")
    ap.add_argument("--max-lateral", type=float, default=3.5, help="max |sideways offset| (m)")
    ap.add_argument("--radius", type=float, nargs=2, default=[0.28, 0.40], help="cover radius range (m)")
    ap.add_argument("--min-minor", type=float, default=6.0, help="reject ellipses thinner than this (px)")
    ap.add_argument("--allow-cut", action="store_true", help="allow the ellipse to be cut by the image border")
    # appearance
    ap.add_argument("--asphalt", type=float, nargs=2, default=[85, 135], help="road grey level range")
    ap.add_argument("--contrast", type=float, nargs=2, default=[20, 45], help="|cover - road| grey levels")
    ap.add_argument("--cover-tone", choices=["dark", "light", "random"], default="random")
    ap.add_argument("--cover-pattern", choices=["flat", "ribs", "grid", "random"], default="random")
    ap.add_argument("--pattern-amp", type=float, default=8.0)
    ap.add_argument("--rim", type=float, default=0.0, help="dark gap ring width as a fraction of R (0 = none)")
    ap.add_argument("--rim-contrast", type=float, default=30.0)
    ap.add_argument("--patches", type=int, default=6, help="soft brighter/darker road patches")
    ap.add_argument("--paint", action="store_true", help="add a painted stripe next to the cover")
    ap.add_argument("--shadow", type=float, default=0.0, help="cast-shadow darkness 0..1 (0 = none)")
    ap.add_argument("--lowfreq", type=float, default=6.0, help="large-scale brightness blotches (grey levels)")
    ap.add_argument("--grain", type=float, default=3.0, help="road texture noise (grey levels)")
    ap.add_argument("--cover-grain", type=float, default=1.5)
    ap.add_argument("--occlude", type=float, default=0.0, help="fraction of the rim hidden by an occluder")
    ap.add_argument("--occlude-prob", type=float, default=1.0, help="probability an occluder is added")
    ap.add_argument("--occluder-gray", type=float, nargs=2, default=[20, 220])
    ap.add_argument("--blur", type=float, default=0.8, help="camera blur sigma (px)")
    ap.add_argument("--motion-blur", type=float, default=0.0,
                    help="motion-blur streak length in px at the manhole, along its image motion direction "
                         "(length ~ f*h/Y^2 * speed * exposure; 0 = none)")
    ap.add_argument("--noise", type=float, default=2.0, help="sensor noise sigma (grey levels)")
    ap.add_argument("--jpeg", type=int, default=0, help="JPEG quality 1..100 (0 = none)")
    # detector box
    ap.add_argument("--box-scale-w", type=float, default=1.0)
    ap.add_argument("--box-scale-h", type=float, default=1.4,
                    help="box height / ellipse height (about 1.4 in the real detections)")
    ap.add_argument("--box-jitter", type=float, default=0.08, help="each box edge moves by up to this fraction")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / "preview").mkdir(exist_ok=True)

    gt_fields = ["filename", "detection_index", "scene_id", "sequence", "frame", "center_x", "center_y",
                 "major_axis", "minor_axis", "angle", "axis_ratio", "rim_visible", "occlusion_requested",
                 "box_x1", "box_y1", "box_x2", "box_y2",
                 "true_tilt_deg", "true_roll_deg", "true_focal", "true_cam_height", "principal_x", "principal_y",
                 "ground_X", "ground_Y_from_below_camera", "paper_a", "paper_b", "true_radius",
                 "cover_delta", "cover_pattern"]
    det_rows, gt_rows = [], []
    counter = 0

    def cam_dict(tilt, h):
        return dict(f=args.focal, px=args.principal[0], py=args.principal[1], h=h, tilt_deg=tilt,
                    roll_deg=args.roll)

    def emit(world, frames, seq_id, frame_ids, scene_id=-1, nrng=None, frng=None, brng=None):
        """frames: list of (cam, y_cam_world, [manhole dicts]) -> writes images + csv rows."""
        nonlocal counter
        for (cam, ycw, mans), fr in zip(frames, frame_ids):
            img, vis = render_frame(nrng or rng, args, world, cam, mans, ycw, frng)
            name = f"{counter:06d}.png"
            counter += 1
            cv2.imwrite(str(out / "images" / name), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            prev = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if args.preview else None
            for m, vf in zip(mans, vis):
                e = m["ellipse"]
                box = make_box(brng or rng, args, e)
                idx = len(det_rows)
                det_rows.append([name, *[round(b, 2) for b in box], 1.0])
                tilt = cam["tilt_deg"]
                Yb = m["Yw"] - ycw
                pb = Yb - cam["h"] / math.tan(math.radians(tilt)) if tilt > 1e-6 else float("nan")
                gt_rows.append(dict(
                    filename=name, detection_index=idx, scene_id=scene_id, sequence=seq_id, frame=fr,
                    center_x=round(e["cx"], 4), center_y=round(e["cy"], 4),
                    major_axis=round(e["major"], 4), minor_axis=round(e["minor"], 4),
                    angle=round(e["angle"], 4), axis_ratio=round(e["minor"] / e["major"], 4),
                    rim_visible=round(vf, 3), occlusion_requested=round(m["occ_frac"], 3),
                    box_x1=round(box[0], 2), box_y1=round(box[1], 2), box_x2=round(box[2], 2), box_y2=round(box[3], 2),
                    true_tilt_deg=round(tilt, 4), true_roll_deg=cam["roll_deg"], true_focal=cam["f"],
                    true_cam_height=cam["h"], principal_x=cam["px"], principal_y=cam["py"],
                    ground_X=round(m["X0"], 4), ground_Y_from_below_camera=round(Yb, 4),
                    paper_a=round(m["X0"], 4), paper_b=round(pb, 4), true_radius=round(m["R"], 4),
                    cover_delta=round(world.cover_delta, 1), cover_pattern=world.cover_pattern))
                if prev is not None:
                    bu, bv = ellipse_points(e, 1440)
                    pts = np.round(np.column_stack([bu, bv]) * 16).astype(np.int32)
                    cv2.polylines(prev, [pts], True, (0, 255, 0), 1, cv2.LINE_AA, shift=4)
                    cv2.rectangle(prev, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
            if prev is not None:
                cv2.imwrite(str(out / "preview" / name), prev)

    def draw_occ(r=None):
        u = (r or rng).random()                         # always consume one draw (keeps streams aligned)
        return args.occlude if (args.occlude > 0 and u < args.occlude_prob) else 0.0

    if args.mode == "random" and args.scene_ids:
        for sid in args.scene_ids:
            # five independent streams per scene, so changing one effect never shifts the others
            srng, wrng, frng, nrng, brng = (np.random.default_rng([args.seed, sid, k]) for k in range(5))
            tilt = srng.uniform(*args.tilt)
            cam = cam_dict(tilt, args.cam_height)
            got = sample_manhole(srng, args, [tilt], [args.cam_height], [(cam, 0.0)])
            if got is None:
                print(f"  scene {sid}: no valid placement, skipped")
                continue
            m, e = got[0], got[1][0]
            m.update(ellipse=e, occ_frac=draw_occ(frng))
            world = World(wrng, m["X0"], m["Yw"], m["R"], args, frng)
            emit(world, [(cam, 0.0, [m])], -1, [0], scene_id=sid, nrng=nrng, frng=frng, brng=brng)
    elif args.mode == "random":
        made = 0
        while made < args.num_images:
            tilt = rng.uniform(*args.tilt)
            cam = cam_dict(tilt, args.cam_height)
            mans, ells = [], []
            for _ in range(args.per_image):
                got = sample_manhole(rng, args, [tilt], [args.cam_height], [(cam, 0.0)])
                if got is None:
                    continue
                m, e = got[0], got[1][0]
                # keep boxes of the same image from overlapping
                if any(abs(m["X0"] - o["X0"]) < 1.0 and abs(m["Yw"] - o["Yw"]) < 1.0 for o in mans):
                    continue
                m.update(ellipse=e, occ_frac=draw_occ())
                mans.append(m)
            if not mans:
                continue
            world = World(rng, mans[0]["X0"], mans[0]["Yw"], mans[0]["R"], args)
            emit(world, [(cam, 0.0, mans)], -1, [0])
            made += 1
            if made % 20 == 0:
                print(f"  {made}/{args.num_images} images")
    else:
        made = 0
        while made < args.num_sequences:
            tilt0 = rng.uniform(*args.tilt)
            tilts = [max(0.05, tilt0 + rng.normal(0, args.pitch_jitter)) for _ in range(args.frames)]
            ycw = [k * args.step for k in range(args.frames)]
            cams = [cam_dict(t, args.cam_height) for t in tilts]
            got = sample_manhole(rng, args, tilts, None, list(zip(cams, ycw)))
            if got is None:
                continue
            m0, ells = got
            world = World(rng, m0["X0"], m0["Yw"], m0["R"], args)
            frames = []
            for k in range(args.frames):
                m = dict(m0)
                m.update(ellipse=ells[k], occ_frac=draw_occ())
                frames.append((cams[k], ycw[k], [m]))
            emit(world, frames, made, list(range(args.frames)))
            made += 1
            print(f"  sequence {made}/{args.num_sequences}")

    with open(out / "detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "x1", "y1", "x2", "y2", "confidence"])
        w.writerows(det_rows)
    with open(out / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=gt_fields)
        w.writeheader()
        w.writerows(gt_rows)
    print(f"wrote {counter} images, {len(gt_rows)} ground-truth ellipses -> {out}")


if __name__ == "__main__":
    main()
