#!/usr/bin/env python3
"""
Manually fit an ellipse to a circular landmark (e.g. manhole cover, roundabout
circle) by clicking points along its edge in the ORIGINAL, uncropped image.

This mirrors how the paper itself validated its method (Dinh & Tang, 2014,
Section 3.1): "15-20 points are manually and randomly selected from the
ellipse in the image."

HOW TO USE
----------
    python manual_ellipse_fit.py --image path/to/photo.jpg --output ellipse_results.csv

    or, to work through every image in a folder in one sitting:

    python manual_ellipse_fit.py --folder path/to/images --output ellipse_results.csv


OUTPUT
------
Appends one row to the output CSV, in the SAME format fit_ellipse_manholes.py
produces, so it's a drop-in input for camera_calibration_from_ellipse.py:
    filename, ellipse_fit, center_x, center_y, major_axis, minor_axis, angle
The individual clicked points are also appended to <output name>_points.csv
(filename, point_index, ix, iy), which camera_calibration_from_ellipse.py reads with
--points.
(center_x/center_y/major_axis/minor_axis/angle are in ORIGINAL image pixel
coordinates -- so when you run camera_calibration_from_ellipse.py on this
output, just pass --image-width/--image-height as your original image's
dimensions
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW_NAME = "Manual ellipse fit (click=point, wheel=zoom, right-drag=pan, u=undo, f=fit, a=accept, n=skip, q=quit)"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def already_processed_filenames(output_csv_path):
    """Return the set of filenames already present in the output CSV, so a
    --folder run can be resumed later without re-doing (or duplicating) images
    you already fit and accepted."""
    names = set()
    if output_csv_path.exists():
        with open(output_csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("filename"):
                    names.add(row["filename"])
    return names

def display_to_image(dx, dy, view):
    return dx / view["zoom"] + view["x0"], dy / view["zoom"] + view["y0"]


def image_to_display(ix, iy, view):
    return (ix - view["x0"]) * view["zoom"], (iy - view["y0"]) * view["zoom"]


def clamp_view(view, img_w, img_h, disp_w, disp_h):
    """Keep the view inside the image; centre the image if it is smaller than the window."""
    for key, size, disp in (("x0", img_w, disp_w), ("y0", img_h, disp_h)):
        visible = disp / view["zoom"]
        if visible >= size:
            view[key] = (size - visible) / 2.0
        else:
            view[key] = min(max(view[key], 0.0), size - visible)


def zoom_at(view, factor, dx, dy, min_zoom, max_zoom):
    """Zoom by 'factor' while keeping the image point under display pixel (dx, dy) fixed."""
    ix, iy = display_to_image(dx, dy, view)
    view["zoom"] = float(np.clip(view["zoom"] * factor, min_zoom, max_zoom))
    view["x0"] = ix - dx / view["zoom"]
    view["y0"] = iy - dy / view["zoom"]


def render_view(img, view, disp_w, disp_h):

    z = view["zoom"]
    M = np.float32([[z, 0, -z * view["x0"]], [0, z, -z * view["y0"]]])
    interp = cv2.INTER_NEAREST if z >= 2.0 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (disp_w, disp_h), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


def process_image(image_path, args):
    img_full = cv2.imread(str(image_path))
    if img_full is None:
        print(f"[error] could not read image (unsupported format or corrupt file): {image_path}")
        return "skipped"

    full_h, full_w = img_full.shape[:2]

    fit_scale = min(1.0, args.max_display_width / full_w, args.max_display_height / full_h)
    disp_w, disp_h = int(full_w * fit_scale), int(full_h * fit_scale)
    min_zoom, max_zoom = fit_scale, args.max_zoom

    
    view = {"zoom": fit_scale, "x0": 0.0, "y0": 0.0}
    clamp_view(view, full_w, full_h, disp_w, disp_h)

    print(f"[info] Original image: {full_w}x{full_h}px, window {disp_w}x{disp_h}px.")
    print("[info] Zoom: mouse wheel or +/- (at the cursor), 0 = whole image. "
          "Pan: right- or middle-drag, or i/j/k/l.")

    # points_full: list of (x, y) in FULL-RESOLUTION image coordinates
    points_full = []
    fitted_ellipse = None  # ((cx, cy), (axis0, axis1), angle) in full-res coords, once fit
    mouse = {"x": disp_w // 2, "y": disp_h // 2, "drag": None}

    def redraw():
        canvas = render_view(img_full, view, disp_w, disp_h)
        z = view["zoom"]
        for (fx, fy) in points_full:
            dx, dy = image_to_display(fx, fy, view)
            cv2.circle(canvas, (int(round(dx)), int(round(dy))), 4, (0, 0, 255), -1)  # red = clicked
        if fitted_ellipse is not None:
            (fcx, fcy), (faxis0, faxis1), fangle = fitted_ellipse
            dcx, dcy = image_to_display(fcx, fcy, view)
            cv2.ellipse(canvas, ((dcx, dcy), (faxis0 * z, faxis1 * z), fangle),
                        (0, 255, 255), 2)  # yellow fitted ellipse
        # small crosshair at the cursor, to aim precisely
        mx, my = mouse["x"], mouse["y"]
        cv2.line(canvas, (mx - 10, my), (mx + 10, my), (0, 255, 0), 1)
        cv2.line(canvas, (mx, my - 10), (mx, my + 10), (0, 255, 0), 1)
        ix, iy = display_to_image(mx, my, view)
        cv2.putText(canvas, f"Points: {len(points_full)}  |  Fit: {'YES' if fitted_ellipse else 'no'}"
                            f"  |  zoom {z / fit_scale:.1f}x  |  cursor ({ix:.1f}, {iy:.1f})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, canvas)

    def on_mouse(event, x, y, flags, param):
        mouse["x"], mouse["y"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            # convert the click on the (zoomed) display back to full-resolution image coordinates
            points_full.append(display_to_image(x, y, view))
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            mouse["drag"] = (x, y, view["x0"], view["y0"])
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            mouse["drag"] = None
        elif event == cv2.EVENT_MOUSEMOVE and mouse["drag"] is not None:
            sx, sy, ox, oy = mouse["drag"]
            view["x0"] = ox - (x - sx) / view["zoom"]
            view["y0"] = oy - (y - sy) / view["zoom"]
            clamp_view(view, full_w, full_h, disp_w, disp_h)
        elif event == cv2.EVENT_MOUSEWHEEL:
            # scroll direction: OpenCV packs the wheel amount into the upper 16 bits of flags
            delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else (flags >> 16)
            factor = 1.25 if delta > 0 else 1 / 1.25
            zoom_at(view, factor, x, y, min_zoom, max_zoom)
            clamp_view(view, full_w, full_h, disp_w, disp_h)
        redraw()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('+'), ord('=')):  # zoom in at the cursor
            zoom_at(view, 1.5, mouse["x"], mouse["y"], min_zoom, max_zoom)
            clamp_view(view, full_w, full_h, disp_w, disp_h)
            redraw()

        elif key == ord('-'):  # zoom out at the cursor
            zoom_at(view, 1 / 1.5, mouse["x"], mouse["y"], min_zoom, max_zoom)
            clamp_view(view, full_w, full_h, disp_w, disp_h)
            redraw()

        elif key == ord('0'):  # back to the whole image
            view.update(zoom=fit_scale, x0=0.0, y0=0.0)
            clamp_view(view, full_w, full_h, disp_w, disp_h)
            redraw()

        elif key in (ord('i'), ord('j'), ord('k'), ord('l')):  # pan by a quarter window
            step_x = 0.25 * disp_w / view["zoom"]
            step_y = 0.25 * disp_h / view["zoom"]
            view["x0"] += {ord('j'): -step_x, ord('l'): step_x}.get(key, 0.0)
            view["y0"] += {ord('i'): -step_y, ord('k'): step_y}.get(key, 0.0)
            clamp_view(view, full_w, full_h, disp_w, disp_h)
            redraw()

        elif key == ord('u'):  # undo last point
            if points_full:
                points_full.pop()
                fitted_ellipse = None
                redraw()

        elif key == ord('r'):  # reset everything
            points_full.clear()
            fitted_ellipse = None
            redraw()

        elif key == ord('f'):  # fit ellipse to clicked points
            if len(points_full) < 5:
                print(f"[warn] need at least 5 points to fit an ellipse (cv2.fitEllipse requirement); "
                      f"you have {len(points_full)}.")
            else:
                pts_array = np.array(points_full, dtype=np.float32)
                fitted_ellipse = cv2.fitEllipseDirect(pts_array)
                (fcx, fcy), (axis0, axis1), angle = fitted_ellipse
                print(f"[info] Fitted ellipse: center=({fcx:.1f}, {fcy:.1f}), "
                      f"axes=({axis0:.1f}, {axis1:.1f}), angle={angle:.1f} deg "
                      f"(in FULL-RESOLUTION image coordinates)")
                redraw()

        elif key == ord('a'):  # accept fit, save to CSV
            if fitted_ellipse is None:
                print("[warn] no fit yet -- press 'f' to fit an ellipse before accepting.")
                continue

            (fcx, fcy), (axis0, axis1), angle = fitted_ellipse
            if axis0 >= axis1:
                major, minor = axis0, axis1
            else:
                major, minor = axis1, axis0
                angle = (angle + 90) % 180

            output_path = Path(args.output)
            file_exists = output_path.exists()
            with open(output_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "filename", "ellipse_fit", "center_x", "center_y",
                    "major_axis", "minor_axis", "angle"
                ])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "filename": image_path.name,
                    "ellipse_fit": "ok",
                    "center_x": round(fcx, 2),
                    "center_y": round(fcy, 2),
                    "major_axis": round(major, 2),
                    "minor_axis": round(minor, 2),
                    "angle": round(angle, 2),
                })

            # The individual clicked points, for camera_calibration_from_ellipse.py --points
            # saved next to the main CSV as <name>_points.csv.
            points_path = output_path.parent / f"{output_path.stem}_points.csv"
            points_exist = points_path.exists()
            with open(points_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "point_index", "ix", "iy"])
                if not points_exist:
                    writer.writeheader()
                for idx, (px, py) in enumerate(points_full):
                    writer.writerow({"filename": image_path.name, "point_index": idx,
                                     "ix": round(px, 2), "iy": round(py, 2)})

            print(f"[saved] Row appended to {output_path}  (+ {len(points_full)} points to {points_path.name})")
            print(f"        center=({fcx:.2f}, {fcy:.2f})  major={major:.2f}  minor={minor:.2f}  angle={angle:.2f}")
            print(f"        Remember: run camera_calibration_from_ellipse.py with "
                  f"--image-width {full_w} --image-height {full_h} "
                  f"(NO crop-offset args needed -- these are full-image coordinates).")
            cv2.destroyAllWindows()
            return "accepted"

        elif key in (ord('n'), ord('s')):  # skip this image, move to next (folder mode)
            cv2.destroyAllWindows()
            print("[info] Skipped -- nothing saved for this image.")
            return "skipped"

        elif key == ord('q') or key == 27:  # 'q' or Esc -- quit without saving
            cv2.destroyAllWindows()
            return "quit"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", help="Path to a single ORIGINAL (uncropped) image")
    image_group.add_argument("--folder", help="Path to a folder of ORIGINAL (uncropped) images -- "
                                               "you'll be shown each one in turn (sorted by filename).")
    parser.add_argument("--output", required=True,
                         help="Path to output CSV. Created if it doesn't exist; a new row is "
                              "APPENDED if it does (so you can run this on multiple images).")
    parser.add_argument("--max-display-width", type=int, default=1400,
                         help="If the image is wider than this, it's downscaled for DISPLAY only "
                              "-- clicked points are automatically mapped back to full-resolution "
                              "original-image coordinates before saving. Default: 1400.")
    parser.add_argument("--max-display-height", type=int, default=800,
                         help="Same as --max-display-width, but for height -- prevents tall/"
                              "portrait images from being too tall for the screen even if their "
                              "width already fits. Default: 800.")
    parser.add_argument("--max-zoom", type=float, default=16.0,
                         help="Largest zoom, in display pixels per image pixel. Default 16 "
                              "(each image pixel drawn as a 16x16 block).")
    parser.add_argument("--skip-existing", action="store_true",
                         help="In --folder mode, skip images whose filename is already a row in "
                              "--output (useful for resuming a batch across multiple sessions).")
    args = parser.parse_args()

    if args.image:
        image_paths = [Path(args.image)]
        if not image_paths[0].exists():
            print(f"[error] image not found: {image_paths[0]}")
            sys.exit(1)
    else:
        folder_path = Path(args.folder)
        if not folder_path.is_dir():
            print(f"[error] folder not found: {folder_path}")
            sys.exit(1)
        image_paths = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            print(f"[error] no images found in {folder_path} "
                  f"(looked for extensions: {', '.join(sorted(IMAGE_EXTENSIONS))})")
            sys.exit(1)

        if args.skip_existing:
            done = already_processed_filenames(Path(args.output))
            before = len(image_paths)
            image_paths = [p for p in image_paths if p.name not in done]
            skipped_count = before - len(image_paths)
            if skipped_count:
                print(f"[info] Skipping {skipped_count} image(s) already in {args.output}.")

        print(f"[info] Found {len(image_paths)} image(s) in {folder_path}.")

    accepted_count = 0
    skipped_count = 0
    for i, image_path in enumerate(image_paths, start=1):
        if len(image_paths) > 1:
            print(f"\n[info] Image {i}/{len(image_paths)}: {image_path.name}")
        result = process_image(image_path, args)
        if result == "accepted":
            accepted_count += 1
        elif result == "skipped":
            skipped_count += 1
        elif result == "quit":
            print("[info] Quit requested -- stopping batch.")
            break

    if len(image_paths) > 1:
        print(f"\n[info] Done. Accepted {accepted_count}, skipped {skipped_count} "
              f"out of {len(image_paths)} image(s).")
    elif accepted_count == 0:
        print("[info] Exited without saving anything.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 55)
        print("[CRASH] The script hit an unexpected error:")
        print("=" * 55)
        traceback.print_exc()
        print("=" * 55)
        input("\nPress Enter to close...")  # keeps the window open so you can read the error
