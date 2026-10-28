"""
batch_test_yolo.py

Runs a trained Ultralytics YOLO model (YOLOv8/YOLO11/YOLO26, etc.) over a
folder of test images, saves annotated copies, and writes a summary CSV.

USAGE
-----
    python batch_test_yolo.py --weights path/to/best.pt --input path/to/test_images --output path/to/results
"""

import argparse
import csv
from pathlib import Path
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained .pt file")
    parser.add_argument("--input", required=True, help="Folder of test images")
    parser.add_argument("--output", required=True, help="Folder to save results")
    parser.add_argument("--confidence", type=float, default=0.6)
    args = parser.parse_args()

    model = YOLO(args.weights)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    annotated_dir = output_dir / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]

    rows = []
    images_with_detections = 0
    total_detections = 0
    all_confidences = []

    for img_path in image_paths:
        results = model.predict(source=str(img_path), conf=args.confidence, verbose=False)
        result = results[0]

        result.save(filename=str(annotated_dir / img_path.name))

        boxes = result.boxes
        if len(boxes) > 0:
            images_with_detections += 1

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            rows.append({
                "filename": img_path.name,
                "class": cls_name,
                "confidence": round(conf, 4),
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
            })
            total_detections += 1
            all_confidences.append(conf)

    csv_path = output_dir / "detections.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nImages processed:        {len(image_paths)}")
    print(f"Images with detections:  {images_with_detections} ({images_with_detections/len(image_paths)*100:.1f}%)")
    print(f"Total detections:        {total_detections}")
    if all_confidences:
        print(f"Avg confidence:           {sum(all_confidences)/len(all_confidences):.3f}")
        print(f"Min / Max confidence:     {min(all_confidences):.3f} / {max(all_confidences):.3f}")
    print(f"\nAnnotated images saved to: {annotated_dir}")
    print(f"Detection CSV saved to:    {csv_path}")


if __name__ == "__main__":
    main()