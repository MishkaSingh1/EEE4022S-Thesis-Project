#!/usr/bin/env python3
"""
Make synthetic KITTI-style road images of manhole covers, together with the exact
ground-truth ellipse of each cover.

Why: to test the ellipse-fitting scripts. Every image shows round manhole cover(s) lying on a
flat road, seen by a pinhole camera with KITTI's settings. The true ellipse in the image is
worked out from the geometry (it is not fitted or measured from the picture), so the ellipse
found by the fitter can be compared against it.

Files written to --output:
    images/000000.png ...   grey road images, saved with 3 channels (1242x375 by default, like KITTI)
    detections.csv          filename,x1,y1,x2,y2,confidence - fake detector boxes, in the same
                            format the fitting scripts read
    ground_truth.csv        one row per manhole; row i belongs to detection i
    preview/000000.png      only with --preview: true ellipse in green, detector box in red

Examples:
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


# ---------------------------------------------------------------------------
# 1. Camera geometry
#
# Road coordinates: X to the right, Y forward, both in metres, measured from the point on the
# road directly below the camera. Image coordinates: u to the right, v down, in pixels.
# ---------------------------------------------------------------------------

def ground_to_image_homography(f, px, py, h, tilt_deg, roll_deg=0.0):
    """3x3 matrix H that maps a road point (X, Y, 1) to the image (u*w, v*w, w).

    f: focal length (px), (px, py): principal point (px), h: camera height above the road (m),
    tilt_deg: camera pitch down (deg), roll_deg: camera roll (deg).
    """
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    # road point -> camera coordinates (x right, y down, z forward) for a camera
    # h metres up and pitched down by the tilt
    road_to_camera = np.array([[1.0, 0.0, 0.0],
                               [0.0, -s, h * c],
                               [0.0, c, h * s]])
    r = math.radians(roll_deg)
    roll = np.array([[math.cos(r), -math.sin(r), 0.0],
                     [math.sin(r), math.cos(r), 0.0],
                     [0.0, 0.0, 1.0]])
    K = np.array([[f, 0.0, px],
                  [0.0, f, py],
                  [0.0, 0.0, 1.0]])
    return K @ roll @ road_to_camera


def project(H, X, Y):
    """Project road points (arrays X, Y) into the image. Returns u, v and the depth
    in front of the camera (negative depth = behind the camera)."""
    P = H @ np.vstack([X, Y, np.ones_like(X)])
    return P[0] / P[2], P[1] / P[2], P[2]


def circle_conic_in_image(H, X0, Y0, R):
    """Conic matrix of the road circle (centre X0, Y0, radius R) as seen in the image.

    On the road the circle is the conic C0: the points p = (X, Y, 1) with p^T C0 p = 0.
    Points move into the image with p_img = H p, so the conic becomes C = H^-T C0 H^-1.
    """
    C0 = np.array([[1.0, 0.0, -X0],
                   [0.0, 1.0, -Y0],
                   [-X0, -Y0, X0 * X0 + Y0 * Y0 - R * R]])
    Hinv = np.linalg.inv(H)
    C = Hinv.T @ C0 @ Hinv
    return 0.5 * (C + C.T)              # make it exactly symmetric (removes rounding errors)


def exact_ellipse(H, X0, Y0, R):
    """The exact image ellipse of a road circle.

    Returns a dict with cx, cy (centre), major, minor (FULL axis lengths, px) and angle (direction
    of the major axis in degrees, 0..180, x right / y down). Returns None if the circle does not
    show up as a proper ellipse in front of the camera.
    """
    # the whole circle has to be in front of the camera
    t = np.linspace(0.0, 2 * math.pi, 360, endpoint=False)
    _, _, depth = project(H, X0 + R * np.cos(t), Y0 + R * np.sin(t))
    if np.any(depth <= 1e-6):
        return None

    C = circle_conic_in_image(H, X0, Y0, R)
    Q, b = C[:2, :2], C[:2, 2]          # quadratic part and linear part of the conic
    if np.linalg.det(Q) <= 0:           # not an ellipse (a parabola or hyperbola)
        return None

    centre = -np.linalg.solve(Q, b)
    F = C[2, 2] + b @ centre            # constant term after moving the conic to its centre
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    semi_axes_squared = -F / eigenvalues
    if np.any(semi_axes_squared <= 0):
        return None
    semi_axes = np.sqrt(semi_axes_squared)

    i = int(np.argmax(semi_axes))       # which eigenvector is the major axis
    angle = math.degrees(math.atan2(eigenvectors[1, i], eigenvectors[0, i])) % 180.0
    return dict(cx=float(centre[0]), cy=float(centre[1]),
                major=float(2 * semi_axes[i]), minor=float(2 * semi_axes[1 - i]),
                angle=angle)


def ellipse_points(ellipse, n=720):
    """n points (u, v) around an ellipse dict from exact_ellipse()."""
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    theta = math.radians(ellipse["angle"])
    a, b = ellipse["major"] / 2, ellipse["minor"] / 2
    x = a * np.cos(t)
    y = b * np.sin(t)
    u = ellipse["cx"] + x * math.cos(theta) - y * math.sin(theta)
    v = ellipse["cy"] + x * math.sin(theta) + y * math.cos(theta)
    return u, v


def ellipse_half_size(ellipse):
    """Half width and half height of the box that just fits around the ellipse."""
    theta = math.radians(ellipse["angle"])
    a, b = ellipse["major"] / 2, ellipse["minor"] / 2
    half_width = math.sqrt((a * math.cos(theta)) ** 2 + (b * math.sin(theta)) ** 2)
    half_height = math.sqrt((a * math.sin(theta)) ** 2 + (b * math.cos(theta)) ** 2)
    return half_width, half_height


# ---------------------------------------------------------------------------
# 2. Smooth (anti-aliased) shapes
#
# Each pixel is split into ss x ss sub-pixels. The share of sub-pixels inside the shape is the
# pixel's "alpha" (0 = outside, 1 = fully inside), so edges are accurate to a fraction of a pixel.
# Only a small crop around the shape is computed; the functions return (alpha, x0, y0), where
# (x0, y0) is the top-left corner of the crop in the full image.
# ---------------------------------------------------------------------------

def crop_around(u, v, width, height):
    """Pixel box around the points (u, v) with a small margin, clipped to the image."""
    x0 = max(0, int(math.floor(u.min())) - 2)
    y0 = max(0, int(math.floor(v.min())) - 2)
    x1 = min(width, int(math.ceil(u.max())) + 3)
    y1 = min(height, int(math.ceil(v.max())) + 3)
    return x0, y0, x1, y1


def polygon_alpha(poly_u, poly_v, width, height, ss=8):
    """Alpha of a polygon: draw it on an ss-times larger canvas, then average back down."""
    x0, y0, x1, y1 = crop_around(poly_u, poly_v, width, height)
    if x1 <= x0 or y1 <= y0:
        return None, 0, 0
    w, h = x1 - x0, y1 - y0
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    # image position p -> big-canvas position (p - crop corner + 0.5) * ss - 0.5
    # (keeps pixel centres lined up)
    big_u = (poly_u - x0 + 0.5) * ss - 0.5
    big_v = (poly_v - y0 + 0.5) * ss - 0.5
    # OpenCV takes integer corners; shift=4 means the integers are in 1/16 px
    corners = np.round(np.column_stack([big_u, big_v]) * 16).astype(np.int32)
    cv2.fillPoly(canvas, [corners], 255, lineType=cv2.LINE_8, shift=4)
    alpha = canvas.reshape(h, ss, w, ss).mean(axis=(1, 3)) / 255.0
    return alpha.astype(np.float32), x0, y0


def ellipse_alpha(C, u, v, width, height, ss=8):
    """Alpha of the inside of conic C (negative inside, see inside_negative_conic).
    Every sub-pixel is tested against the conic equation, so the edge is the exact ellipse.
    (u, v) are outline points, only used to find the crop."""
    x0, y0, x1, y1 = crop_around(u, v, width, height)
    if x1 <= x0 or y1 <= y0:
        return None, 0, 0
    offsets = -0.5 + (np.arange(ss) + 0.5) / ss            # sub-pixel positions inside one pixel
    xs = (np.arange(x0, x1)[:, None] + offsets[None, :]).ravel()
    ys = (np.arange(y0, y1)[:, None] + offsets[None, :]).ravel()
    X, Y = np.meshgrid(xs, ys)
    q = (C[0, 0] * X * X + 2 * C[0, 1] * X * Y + C[1, 1] * Y * Y
         + 2 * C[0, 2] * X + 2 * C[1, 2] * Y + C[2, 2])
    inside = (q < 0).astype(np.float32)
    alpha = inside.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))
    return alpha, x0, y0


def paste_max(full, alpha, x0, y0):
    """Put a cropped alpha into the full-size mask (keeping the larger value)."""
    if alpha is None:
        return
    h, w = alpha.shape
    full[y0:y0 + h, x0:x0 + w] = np.maximum(full[y0:y0 + h, x0:x0 + w], alpha)


def circle_outline(H, X0, Y0, R, n=720):
    """n image points around a road circle."""
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    u, v, _ = project(H, X0 + R * np.cos(t), Y0 + R * np.sin(t))
    return u, v


def inside_negative_conic(H, X0, Y0, R):
    """Image conic of the road circle, with its sign flipped if needed so that
    points INSIDE the ellipse give a negative value."""
    C = circle_conic_in_image(H, X0, Y0, R)
    centre = -np.linalg.solve(C[:2, :2], C[:2, 2])
    p = np.array([centre[0], centre[1], 1.0])
    return -C if p @ C @ p > 0 else C


def circle_alpha(H, X0, Y0, R, width, height):
    """Alpha of a road circle drawn into the image."""
    u, v = circle_outline(H, X0, Y0, R)
    return ellipse_alpha(inside_negative_conic(H, X0, Y0, R), u, v, width, height)


# ---------------------------------------------------------------------------
# 3. The road
#
# Everything here is defined in road coordinates, so it stays in the same place on the road
# when the camera moves (needed for sequence mode).
# ---------------------------------------------------------------------------

def smooth_noise(rng, height, width, cell):
    """Large soft blotches: coarse random grid scaled up to full size."""
    small = rng.normal(size=(height // cell + 3, width // cell + 3)).astype(np.float32)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


class World:
    """Everything that belongs to the road and has to stay the same in every frame of a sequence.

    rng draws the road and the cover; effect_rng draws the paint stripe and the shadow.
    Normally they are the same generator; for --scene-ids they are separate.
    """

    def __init__(self, rng, X0, Y0w, R, args, effect_rng):
        self.road_grey = rng.uniform(*args.asphalt)

        # soft lighter/darker patches on the road (repairs, wear, oil),
        # each one stored as (centre X, centre Y, size X, size Y, brightness change)
        self.patches = [(rng.uniform(-8, 8), rng.uniform(0, 40), rng.uniform(0.4, 2.5),
                         rng.uniform(0.4, 4.0), rng.uniform(-18, 18))
                        for _ in range(args.patches)]

        # how much lighter (+) or darker (-) the cover is than the road.
        # The random sign is always drawn, even when --cover-tone fixes it (see note at the top).
        contrast = rng.uniform(*args.contrast)
        random_sign = rng.choice([-1, 1])
        sign = {"dark": -1, "light": 1}.get(args.cover_tone, random_sign)
        self.cover_delta = sign * contrast

        if args.cover_pattern == "random":
            self.cover_pattern = rng.choice(["flat", "ribs", "grid"])
        else:
            self.cover_pattern = args.cover_pattern
        self.pattern_amp = args.pattern_amp
        # pattern settings are always drawn, even when the pattern does not use them
        self.num_ribs = int(rng.integers(3, 7))
        self.grid_angle = rng.uniform(0, math.pi)
        self.grid_period = rng.uniform(0.04, 0.08)

        self.X0, self.Y0w, self.R = X0, Y0w, R
        self.rim = args.rim
        self.rim_dark = args.rim_contrast

        # optional painted stripe along the road, next to the cover
        self.paint = None
        if args.paint:
            side = effect_rng.choice([-1, 1])
            gap = effect_rng.uniform(0.04, 0.35)
            self.paint = dict(centre_x=X0 + side * (R + gap + args.rim * R),
                              half_width=effect_rng.uniform(0.05, 0.12),
                              brightness=effect_rng.uniform(40, 90),
                              dashed=bool(effect_rng.random() < 0.5),
                              period=effect_rng.uniform(3.0, 6.0),
                              phase=effect_rng.uniform(0, 6))

        # optional shadow: a straight soft edge on the road that passes near the cover
        self.shadow = None
        if args.shadow > 0:
            angle = effect_rng.uniform(0, 2 * math.pi)
            normal = np.array([math.cos(angle), math.sin(angle)])
            offset = effect_rng.uniform(-0.8, 0.8) * R
            self.shadow = dict(normal=normal,
                               d=normal @ np.array([X0, Y0w]) + offset,
                               softness=effect_rng.uniform(0.03, 0.12),
                               factor=1.0 - args.shadow)

    def asphalt(self, X, Yw):
        """Road grey level at road points (X, Yw)."""
        grey = np.full(X.shape, self.road_grey, np.float32)
        for (px, py, sx, sy, amp) in self.patches:
            r2 = ((X - px) / sx) ** 2 + ((Yw - py) / sy) ** 2
            grey += amp * np.exp(-r2).astype(np.float32)

        if self.paint:
            p = self.paint
            # 1 on the stripe, 0 off it, with a 2 cm soft edge
            stripe = np.clip(0.5 + (p["half_width"] - np.abs(X - p["centre_x"])) / 0.02, 0, 1)
            if p["dashed"]:
                # painted for the first half of every period, 5 cm soft ends
                pos = (Yw + p["phase"]) % p["period"]
                stripe = stripe * np.clip(0.5 + (0.5 * p["period"] - pos) / 0.05, 0, 1)
            grey = grey * (1 - stripe) + (self.road_grey + p["brightness"]) * stripe
        return grey

    def cover_tone(self, X, Yw):
        """Grey level of the cover surface. The pattern is defined on the cover,
        so it moves with the cover."""
        dx, dy = X - self.X0, Yw - self.Y0w
        tone = np.full(X.shape, self.road_grey + self.cover_delta, np.float32)
        if self.cover_pattern == "ribs":
            # rings around the centre
            rho = np.sqrt(dx * dx + dy * dy) / self.R
            tone += self.pattern_amp * np.sin(2 * math.pi * self.num_ribs * rho).astype(np.float32)
        elif self.cover_pattern == "grid":
            # rotated checker pattern
            c, s = math.cos(self.grid_angle), math.sin(self.grid_angle)
            gx = (c * dx + s * dy) / self.grid_period
            gy = (-s * dx + c * dy) / self.grid_period
            tone += self.pattern_amp * (np.sin(2 * math.pi * gx) * np.sin(2 * math.pi * gy)).astype(np.float32)
        return tone

    def shadow_factor(self, X, Yw):
        """Brightness multiplier of the shadow (1 = lit), or None when there is no shadow."""
        if not self.shadow:
            return None
        s = self.shadow
        dist = s["normal"][0] * X + s["normal"][1] * Yw - s["d"]
        k = np.clip(0.5 + dist / s["softness"], 0, 1)          # 0 = lit side, 1 = shadow side
        return (1 - k * (1 - s["factor"])).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Occluder: a block with one straight edge that hides part of the rim
# ---------------------------------------------------------------------------

def occluder_polygon(rng, ellipse, fraction):
    """4-corner polygon (image coordinates) whose straight edge crosses the ellipse so that
    about 'fraction' of the rim points are behind it."""
    rim_u, rim_v = ellipse_points(ellipse)
    angle = rng.uniform(0, 2 * math.pi)
    n = np.array([math.cos(angle), math.sin(angle)])       # direction the occluder comes from
    tau = np.array([-n[1], n[0]])                           # direction along its straight edge

    # position of each rim point along n; the edge is placed so `fraction` of them are past it
    along = (rim_u - ellipse["cx"]) * n[0] + (rim_v - ellipse["cy"]) * n[1]
    edge = np.quantile(along, 1.0 - fraction)
    hidden = along > edge

    # make the block wide enough to cover all hidden rim points, plus a few pixels
    sideways = np.abs((rim_u - ellipse["cx"]) * tau[0] + (rim_v - ellipse["cy"]) * tau[1])
    half = (sideways[hidden].max() if hidden.any() else 1.0) + rng.uniform(3, 15)
    depth = max(ellipse["major"], 20.0) * rng.uniform(0.8, 1.6)

    c0 = np.array([ellipse["cx"], ellipse["cy"]])
    corners = np.array([c0 + n * edge - tau * half,
                        c0 + n * edge + tau * half,
                        c0 + n * (edge + depth) + tau * (half + rng.uniform(0, 10)),
                        c0 + n * (edge + depth) - tau * (half + rng.uniform(0, 10))])
    return corners[:, 0], corners[:, 1]


# ---------------------------------------------------------------------------
# 5. Motion blur: a straight streak in the direction the manhole moves in the image
# ---------------------------------------------------------------------------

def line_kernel(length, dx, dy):
    """Blur kernel for a straight streak `length` px long in direction (dx, dy).
    The streak is sampled finely and each sample is split between the 4 nearest kernel cells,
    so any length (also fractions of a pixel) and any angle work.
    The kernel sums to 1, so the brightness does not change."""
    half = int(math.ceil(length / 2.0)) + 2
    size = 2 * half + 1
    kernel = np.zeros((size, size), np.float32)
    n = max(8, int(length * 16))
    for t in np.linspace(-length / 2.0, length / 2.0, n):
        x, y = half + t * dx, half + t * dy
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        kernel[y0, x0] += (1 - fx) * (1 - fy)
        kernel[y0, x0 + 1] += fx * (1 - fy)
        kernel[y0 + 1, x0] += (1 - fx) * fy
        kernel[y0 + 1, x0 + 1] += fx * fy
    return kernel / kernel.sum()


def image_motion_direction(H, X0, Y0_rel):
    """Unit vector (du, dv) of how the manhole centre moves in the image when the camera drives
    forward. Found by projecting the centre at two distances 5 cm apart."""
    u1, v1, _ = project(H, np.array([X0]), np.array([Y0_rel]))
    u0, v0, _ = project(H, np.array([X0]), np.array([Y0_rel - 0.05]))
    d = np.array([float(u0[0] - u1[0]), float(v0[0] - v1[0])])
    length = np.hypot(*d)
    if length < 1e-9:
        return 1.0, 0.0
    return d[0] / length, d[1] / length


# ---------------------------------------------------------------------------
# 6. Drawing one frame
#
# Each manhole is a dict with X0, Yw (Y on the road in world coordinates), R (radius),
# ellipse (exact, from exact_ellipse) and occ_frac (fraction of the rim to hide).
# y_cam_world is how far the camera has driven forward (0 except in sequence mode).
# ---------------------------------------------------------------------------

def road_coordinates_of_pixels(H, width, height):
    """Road point (X, Y) seen by every pixel, relative to the camera."""
    Hinv = np.linalg.inv(H)
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    g = np.einsum("ij,jhw->ihw", Hinv.astype(np.float32), np.stack([uu, vv, np.ones_like(uu)]))
    return g[0] / g[2], g[1] / g[2]


def draw_road_and_sky(rng, args, world, X, Yw, on_road):
    height, width = X.shape
    road = world.asphalt(X, Yw) + args.lowfreq * smooth_noise(rng, height, width, 48)
    road += args.grain * cv2.GaussianBlur(rng.normal(size=(height, width)).astype(np.float32),
                                          (0, 0), 0.7) * 1.8
    sky = np.linspace(185, 150, height, dtype=np.float32)[:, None] * np.ones((1, width), np.float32)
    return np.where(on_road, road, sky)


def draw_manholes(rng, args, world, H, X, Yw, manholes, y_cam_world, img):
    height, width = X.shape
    for m in manholes:
        Y_rel = m["Yw"] - y_cam_world              # distance ahead of the camera

        # optional dark ring around the cover (the gap between cover and frame)
        if world.rim > 0:
            rim_alpha = np.zeros((height, width), np.float32)
            paste_max(rim_alpha, *circle_alpha(H, m["X0"], Y_rel, m["R"] * (1 + world.rim), width, height))
            img = img * (1 - rim_alpha) + (img - world.rim_dark) * rim_alpha

        # the cover itself, with its own fine grain
        cover_alpha = np.zeros((height, width), np.float32)
        paste_max(cover_alpha, *circle_alpha(H, m["X0"], Y_rel, m["R"], width, height))
        cover = world.cover_tone(X, Yw) + args.cover_grain * cv2.GaussianBlur(
            rng.normal(size=(height, width)).astype(np.float32), (0, 0), 0.7) * 1.8
        img = img * (1 - cover_alpha) + cover * cover_alpha
    return img


def draw_occluders(rng, args, manholes, img):
    """Draw the occluders (in image coordinates). Returns the image and a mask of where they are."""
    height, width = img.shape
    occluded = np.zeros((height, width), np.float32)
    for m in manholes:
        if m["occ_frac"] <= 0:
            continue
        ou, ov = occluder_polygon(rng, m["ellipse"], m["occ_frac"])
        alpha, x0, y0 = polygon_alpha(ou, ov, width, height)
        if alpha is None:
            continue
        grey = rng.uniform(*args.occluder_gray)
        h, w = alpha.shape
        texture = grey + 4.0 * cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), 1.0)
        crop = (slice(y0, y0 + h), slice(x0, x0 + w))
        img[crop] = img[crop] * (1 - alpha) + texture * alpha
        occluded[crop] = np.maximum(occluded[crop], alpha)
    return img, occluded


def apply_camera_effects(rng, args, img):
    """Lens blur, sensor noise, rounding to 8 bit and optional JPEG compression."""
    if args.blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), args.blur)
    if args.noise > 0:
        img = img + args.noise * rng.standard_normal(img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if args.jpeg > 0:
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg)])
        img = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return img


def rim_visible_fraction(ellipse, occluded, width, height):
    """Share of rim points that are inside the image and not behind an occluder."""
    u, v = ellipse_points(ellipse)
    in_image = (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    iu = np.clip(np.round(u).astype(int), 0, width - 1)
    iv = np.clip(np.round(v).astype(int), 0, height - 1)
    visible = in_image & (occluded[iv, iu] < 0.5)
    return float(visible.mean())


def render_frame(noise_rng, effect_rng, args, world, cam, manholes, y_cam_world):
    """Draw one grey image. Returns the uint8 image and, for each manhole,
    the fraction of its rim that is visible.

    noise_rng draws the textures and noise; effect_rng draws the occluders.
    """
    width, height = args.width, args.height_px
    H = ground_to_image_homography(**cam)
    X, Y = road_coordinates_of_pixels(H, width, height)
    on_road = (Y > 0.3) & (Y < 150.0)          # pixels that see the road (the rest is sky)
    Yw = Y + y_cam_world                       # road Y in world coordinates

    img = draw_road_and_sky(noise_rng, args, world, X, Yw, on_road)
    img = draw_manholes(noise_rng, args, world, H, X, Yw, manholes, y_cam_world, img)

    # the shadow darkens everything on the road, covers included
    shadow = world.shadow_factor(X, Yw)
    if shadow is not None:
        img = np.where(on_road, img * shadow, img)

    img, occluded = draw_occluders(effect_rng, args, manholes, img)

    # Motion blur: one streak for the whole image, in the direction the first manhole moves.
    # The manhole is small, so one direction and length is a good model around it.
    # Both blurs are convolutions, so it does not matter that this comes before the lens blur.
    if args.motion_blur > 0 and manholes:
        m0 = manholes[0]
        dx, dy = image_motion_direction(H, m0["X0"], m0["Yw"] - y_cam_world)
        img = cv2.filter2D(img, -1, line_kernel(args.motion_blur, dx, dy), borderType=cv2.BORDER_REPLICATE)

    img = apply_camera_effects(noise_rng, args, img)

    visible = [rim_visible_fraction(m["ellipse"], occluded, width, height) for m in manholes]
    return img, visible


# ---------------------------------------------------------------------------
# 7. Choosing manholes and fake detector boxes
# ---------------------------------------------------------------------------

def sample_manhole(rng, args, frames):
    """Pick a manhole position and radius whose ellipse is valid and inside the image in EVERY
    frame. frames is a list of (camera settings, how far the camera has driven).

    Returns (manhole dict with X0, Yw, R; list of exact ellipses, one per frame),
    or None if 200 tries fail.
    """
    margin = 2 if not args.allow_cut else -10 ** 6      # --allow-cut: no border check
    for _ in range(200):
        # distance is drawn log-uniform, so near and far manholes are equally common per octave
        Y0 = math.exp(rng.uniform(math.log(args.dist[0]), math.log(args.dist[1])))
        X0 = rng.uniform(-args.max_lateral, args.max_lateral)
        R = rng.uniform(*args.radius)

        ellipses = []
        for cam, y_cam_world in frames:
            H = ground_to_image_homography(**cam)
            e = exact_ellipse(H, X0, Y0 - y_cam_world, R)
            if e is None or e["minor"] < args.min_minor:
                break
            half_w, half_h = ellipse_half_size(e)
            if (e["cx"] - half_w < margin or e["cx"] + half_w > args.width - 1 - margin or
                    e["cy"] - half_h < margin or e["cy"] + half_h > args.height_px - 1 - margin):
                break
            ellipses.append(e)
        else:
            # the loop did not break, so every frame is fine
            return dict(X0=X0, Yw=Y0, R=R), ellipses
    return None


def make_box(rng, args, ellipse):
    """Fake detector box: the box around the ellipse, scaled and with each edge moved randomly."""
    half_w, half_h = ellipse_half_size(ellipse)
    w, h = 2 * half_w * args.box_scale_w, 2 * half_h * args.box_scale_h
    x1, x2 = ellipse["cx"] - w / 2, ellipse["cx"] + w / 2
    y1, y2 = ellipse["cy"] - h / 2, ellipse["cy"] + h / 2
    j = args.box_jitter
    x1 += rng.uniform(-j, j) * w
    x2 += rng.uniform(-j, j) * w
    y1 += rng.uniform(-j, j) * h
    y2 += rng.uniform(-j, j) * h
    return (max(0.0, x1), max(0.0, y1),
            min(args.width - 1.0, x2), min(args.height_px - 1.0, y2))


def pick_occlusion(rng, args):
    """Fraction of the rim to hide for one manhole (0 = no occluder).
    One random number is always drawn, even without occlusion (see note at the top)."""
    u = rng.random()
    if args.occlude > 0 and u < args.occlude_prob:
        return args.occlude
    return 0.0


def camera_settings(args, tilt):
    """Camera dict, in the form ground_to_image_homography(**cam) takes."""
    return dict(f=args.focal, px=args.principal[0], py=args.principal[1],
                h=args.cam_height, tilt_deg=tilt, roll_deg=args.roll)


# ---------------------------------------------------------------------------
# 8. Saving images and CSV files
# ---------------------------------------------------------------------------

GROUND_TRUTH_COLUMNS = [
    "filename", "detection_index", "scene_id", "sequence", "frame",
    "center_x", "center_y", "major_axis", "minor_axis", "angle", "axis_ratio",
    "rim_visible", "occlusion_requested",
    "box_x1", "box_y1", "box_x2", "box_y2",
    "true_tilt_deg", "true_roll_deg", "true_focal", "true_cam_height", "principal_x", "principal_y",
    "ground_X", "ground_Y_from_below_camera", "paper_a", "paper_b", "true_radius",
    "cover_delta", "cover_pattern",
]


class DatasetWriter:
    """Draws frames, saves the images and collects the CSV rows."""

    def __init__(self, args):
        self.args = args
        self.out = Path(args.output)
        (self.out / "images").mkdir(parents=True, exist_ok=True)
        if args.preview:
            (self.out / "preview").mkdir(exist_ok=True)
        self.detection_rows = []
        self.ground_truth_rows = []
        self.image_count = 0

    def add_frames(self, world, frames, sequence_id, frame_ids, scene_id,
                   noise_rng, effect_rng, box_rng):
        """frames: list of (camera settings, how far the camera has driven, list of manholes)."""
        args = self.args
        for (cam, y_cam_world, manholes), frame_id in zip(frames, frame_ids):
            img, visible = render_frame(noise_rng, effect_rng, args, world, cam, manholes, y_cam_world)
            name = f"{self.image_count:06d}.png"
            self.image_count += 1
            cv2.imwrite(str(self.out / "images" / name), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if args.preview else None

            for m, visible_fraction in zip(manholes, visible):
                e = m["ellipse"]
                box = make_box(box_rng, args, e)
                index = len(self.detection_rows)
                self.detection_rows.append([name, *[round(b, 2) for b in box], 1.0])

                tilt = cam["tilt_deg"]
                Y_below = m["Yw"] - y_cam_world        # forward distance from the point below the camera
                # paper_b: forward distance measured from where the optical axis hits the road
                # (h / tan(tilt) ahead of the camera); not defined for a level camera
                if tilt > 1e-6:
                    paper_b = Y_below - cam["h"] / math.tan(math.radians(tilt))
                else:
                    paper_b = float("nan")

                self.ground_truth_rows.append(dict(
                    filename=name, detection_index=index, scene_id=scene_id,
                    sequence=sequence_id, frame=frame_id,
                    center_x=round(e["cx"], 4), center_y=round(e["cy"], 4),
                    major_axis=round(e["major"], 4), minor_axis=round(e["minor"], 4),
                    angle=round(e["angle"], 4), axis_ratio=round(e["minor"] / e["major"], 4),
                    rim_visible=round(visible_fraction, 3), occlusion_requested=round(m["occ_frac"], 3),
                    box_x1=round(box[0], 2), box_y1=round(box[1], 2),
                    box_x2=round(box[2], 2), box_y2=round(box[3], 2),
                    true_tilt_deg=round(tilt, 4), true_roll_deg=cam["roll_deg"], true_focal=cam["f"],
                    true_cam_height=cam["h"], principal_x=cam["px"], principal_y=cam["py"],
                    ground_X=round(m["X0"], 4), ground_Y_from_below_camera=round(Y_below, 4),
                    paper_a=round(m["X0"], 4), paper_b=round(paper_b, 4), true_radius=round(m["R"], 4),
                    cover_delta=round(world.cover_delta, 1), cover_pattern=world.cover_pattern))

                if preview is not None:
                    u, v = ellipse_points(e, 1440)
                    points = np.round(np.column_stack([u, v]) * 16).astype(np.int32)   # 1/16 px units
                    cv2.polylines(preview, [points], True, (0, 255, 0), 1, cv2.LINE_AA, shift=4)
                    cv2.rectangle(preview, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                  (0, 0, 255), 1)

            if preview is not None:
                cv2.imwrite(str(self.out / "preview" / name), preview)

    def save_csv_files(self):
        with open(self.out / "detections.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "x1", "y1", "x2", "y2", "confidence"])
            w.writerows(self.detection_rows)
        with open(self.out / "ground_truth.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=GROUND_TRUTH_COLUMNS)
            w.writeheader()
            w.writerows(self.ground_truth_rows)


# ---------------------------------------------------------------------------
# 9. The three ways of making a dataset
# ---------------------------------------------------------------------------

def make_paired_scenes(args, writer):
    """--scene-ids: each scene gets five random streams of its own, seeded from (seed, scene id).
    Changing an effect option (occlusion, shadow, ...) only changes that effect's stream,
    so the rest of the scene stays exactly the same."""
    for scene_id in args.scene_ids:
        scene_rng, world_rng, effect_rng, noise_rng, box_rng = (
            np.random.default_rng([args.seed, scene_id, k]) for k in range(5))

        cam = camera_settings(args, scene_rng.uniform(*args.tilt))
        found = sample_manhole(scene_rng, args, [(cam, 0.0)])
        if found is None:
            print(f"  scene {scene_id}: no valid placement, skipped")
            continue
        manhole, ellipses = found
        manhole.update(ellipse=ellipses[0], occ_frac=pick_occlusion(effect_rng, args))

        world = World(world_rng, manhole["X0"], manhole["Yw"], manhole["R"], args, effect_rng)
        writer.add_frames(world, [(cam, 0.0, [manhole])], -1, [0], scene_id,
                          noise_rng, effect_rng, box_rng)


def make_random_images(args, rng, writer):
    """Default mode: independent random images, all from one random stream."""
    made = 0
    while made < args.num_images:
        cam = camera_settings(args, rng.uniform(*args.tilt))
        manholes = []
        for _ in range(args.per_image):
            found = sample_manhole(rng, args, [(cam, 0.0)])
            if found is None:
                continue
            manhole, ellipses = found
            # skip manholes within 1 m of one already in this image, so they don't overlap
            if any(abs(manhole["X0"] - o["X0"]) < 1.0 and abs(manhole["Yw"] - o["Yw"]) < 1.0
                   for o in manholes):
                continue
            manhole.update(ellipse=ellipses[0], occ_frac=pick_occlusion(rng, args))
            manholes.append(manhole)
        if not manholes:
            continue

        # the road is built around the first manhole of the image
        first = manholes[0]
        world = World(rng, first["X0"], first["Yw"], first["R"], args, rng)
        writer.add_frames(world, [(cam, 0.0, manholes)], -1, [0], -1, rng, rng, rng)
        made += 1
        if made % 20 == 0:
            print(f"  {made}/{args.num_images} images")


def make_sequences(args, rng, writer):
    """Sequence mode: the camera moves forward --step metres per frame over one manhole,
    with a small random pitch change in every frame."""
    made = 0
    while made < args.num_sequences:
        start_tilt = rng.uniform(*args.tilt)
        tilts = [max(0.05, start_tilt + rng.normal(0, args.pitch_jitter)) for _ in range(args.frames)]
        driven = [k * args.step for k in range(args.frames)]
        cams = [camera_settings(args, t) for t in tilts]

        found = sample_manhole(rng, args, list(zip(cams, driven)))
        if found is None:
            continue
        manhole, ellipses = found

        world = World(rng, manhole["X0"], manhole["Yw"], manhole["R"], args, rng)
        frames = []
        for k in range(args.frames):
            m = dict(manhole)                  # same manhole, but its own ellipse and occlusion per frame
            m.update(ellipse=ellipses[k], occ_frac=pick_occlusion(rng, args))
            frames.append((cams[k], driven[k], [m]))
        writer.add_frames(world, frames, made, list(range(args.frames)), -1, rng, rng, rng)
        made += 1
        print(f"  sequence {made}/{args.num_sequences}")


# ---------------------------------------------------------------------------
# 10. Command line
# ---------------------------------------------------------------------------

def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = ap.add_argument_group("output and mode")
    g.add_argument("--output", required=True, help="output folder")
    g.add_argument("--mode", choices=["random", "sequence"], default="random")
    g.add_argument("--num-images", type=int, default=50, help="random mode: number of images")
    g.add_argument("--per-image", type=int, default=1, help="random mode: manholes per image")
    g.add_argument("--num-sequences", type=int, default=5, help="sequence mode: number of sequences")
    g.add_argument("--frames", type=int, default=6, help="sequence mode: frames per sequence")
    g.add_argument("--step", type=float, default=0.8, help="sequence mode: camera moves this far per frame (m)")
    g.add_argument("--pitch-jitter", type=float, default=0.1, help="sequence mode: random tilt change per frame (deg)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--scene-ids", type=int, nargs="+",
                   help="random mode: make exactly these scene ids. A scene id (with --seed) fixes the geometry, "
                        "road, cover, noise pattern and box jitter, so the same id with different effect options "
                        "gives a PAIRED image.")

    g = ap.add_argument_group("camera (KITTI defaults)")
    g.add_argument("--width", type=int, default=1242, help="image width (px)")
    g.add_argument("--height-px", type=int, default=375, help="image height (px)")
    g.add_argument("--focal", type=float, default=721.5377, help="focal length (px)")
    g.add_argument("--principal", type=float, nargs=2, default=[609.5593, 172.854], help="principal point (px)")
    g.add_argument("--cam-height", type=float, default=1.65, help="camera height above the road (m)")
    g.add_argument("--tilt", type=float, nargs=2, default=[0.5, 2.5], help="camera pitch-down range (deg)")
    g.add_argument("--roll", type=float, default=0.0, help="camera roll (deg)")

    g = ap.add_argument_group("manhole placement")
    g.add_argument("--dist", type=float, nargs=2, default=[6.0, 12.0],
                   help="forward distance range from the point below the camera (m)")
    g.add_argument("--max-lateral", type=float, default=3.5, help="largest sideways offset either way (m)")
    g.add_argument("--radius", type=float, nargs=2, default=[0.28, 0.40], help="cover radius range (m)")
    g.add_argument("--min-minor", type=float, default=6.0, help="reject ellipses thinner than this (px)")
    g.add_argument("--allow-cut", action="store_true", help="allow the image border to cut the ellipse")

    g = ap.add_argument_group("appearance")
    g.add_argument("--asphalt", type=float, nargs=2, default=[85, 135], help="road grey level range")
    g.add_argument("--contrast", type=float, nargs=2, default=[20, 45],
                   help="difference between cover and road (grey levels)")
    g.add_argument("--cover-tone", choices=["dark", "light", "random"], default="random")
    g.add_argument("--cover-pattern", choices=["flat", "ribs", "grid", "random"], default="random")
    g.add_argument("--pattern-amp", type=float, default=8.0, help="strength of the cover pattern (grey levels)")
    g.add_argument("--rim", type=float, default=0.0, help="dark ring width as a fraction of R (0 = none)")
    g.add_argument("--rim-contrast", type=float, default=30.0, help="how much darker the ring is (grey levels)")
    g.add_argument("--patches", type=int, default=6, help="number of soft lighter/darker road patches")
    g.add_argument("--paint", action="store_true", help="add a painted stripe next to the cover")
    g.add_argument("--shadow", type=float, default=0.0, help="shadow darkness 0..1 (0 = none)")
    g.add_argument("--lowfreq", type=float, default=6.0, help="large soft brightness blotches (grey levels)")
    g.add_argument("--grain", type=float, default=3.0, help="road texture noise (grey levels)")
    g.add_argument("--cover-grain", type=float, default=1.5, help="cover texture noise (grey levels)")
    g.add_argument("--occlude", type=float, default=0.0, help="fraction of the rim hidden by an occluder")
    g.add_argument("--occlude-prob", type=float, default=1.0, help="chance that a manhole gets an occluder")
    g.add_argument("--occluder-gray", type=float, nargs=2, default=[20, 220], help="occluder grey level range")

    g = ap.add_argument_group("camera effects")
    g.add_argument("--blur", type=float, default=0.8, help="lens blur sigma (px)")
    g.add_argument("--motion-blur", type=float, default=0.0,
                   help="motion-blur streak length at the manhole (px), along its image motion "
                        "(length ~ f*h/Y^2 * speed * exposure; 0 = none)")
    g.add_argument("--noise", type=float, default=2.0, help="sensor noise sigma (grey levels)")
    g.add_argument("--jpeg", type=int, default=0, help="JPEG quality 1..100 (0 = none)")

    g = ap.add_argument_group("fake detector box")
    g.add_argument("--box-scale-w", type=float, default=1.0, help="box width / ellipse width")
    g.add_argument("--box-scale-h", type=float, default=1.4,
                   help="box height / ellipse height (about 1.4 in the real detections)")
    g.add_argument("--box-jitter", type=float, default=0.08,
                   help="each box edge moves by up to this fraction of the box size")
    g.add_argument("--preview", action="store_true", help="also save preview images with the true ellipse")
    return ap


def main():
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    writer = DatasetWriter(args)

    if args.mode == "random" and args.scene_ids:
        make_paired_scenes(args, writer)
    elif args.mode == "random":
        make_random_images(args, rng, writer)
    else:
        make_sequences(args, rng, writer)

    writer.save_csv_files()
    print(f"wrote {writer.image_count} images, {len(writer.ground_truth_rows)} ground-truth ellipses -> {writer.out}")


if __name__ == "__main__":
    main()
