"""
Vehicle detection with a pretrained YOLO nano model.

What this file does:
- Opens a video file or a webcam
- Runs YOLOv8n on each frame
- Keeps only 4 classes: car, motorcycle, bus, truck
- Draws boxes, shows the counts, and can return counts for one frame
  through detect_frame(), which the backend can reuse later

This file needs ultralytics and opencv-python (see requirements.txt).
The rest of the system (algorithm, API, dashboard, simulation) runs
without it, so a missing video never blocks the whole project.

Run it (PowerShell, from the smart-traffic-system folder):
    python backend/detection.py --source 0
    python backend/detection.py --source data/videos/test.mp4
Press q to quit.
"""

import argparse
from collections import Counter

# COCO class IDs used by the pretrained model
VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def detect_frame(frame, model, confidence=0.4):
    """Run detection on one frame. Returns (counts_dict, annotated_frame)."""
    results = model(frame, imgsz=640, conf=confidence, verbose=False)[0]
    counts = Counter()
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_IDS:
            counts[VEHICLE_IDS[cls]] += 1
    return dict(counts), results.plot()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/videos/test.mp4",
                        help="Video file path, or 0 for the webcam")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    # Imported here so this file's helpers can be read/tested without
    # the heavy libraries installed.
    import cv2
    from ultralytics import YOLO

    source = 0 if args.source == "0" else args.source
    model = YOLO("yolov8n.pt")  # downloads once, about 6 MB

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Video source could not be opened:", args.source)
        print("Check the file path, or try --source 0 for the webcam.")
        return

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        counts, annotated = detect_frame(frame, model, args.conf)
        text = " | ".join(f"{k}: {v}" for k, v in counts.items()) or "No vehicles"
        cv2.putText(annotated, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Traffic Detection", annotated)

        if frame_no % 30 == 0:
            print(f"Frame {frame_no}: {counts}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
