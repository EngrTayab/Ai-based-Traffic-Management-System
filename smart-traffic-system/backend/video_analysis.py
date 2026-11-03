"""
Video analysis: traffic video in, vehicle counts out.

How it works (simple and explainable):
- A normal traffic video shows ONE road (approach). So the system
  takes one video per approach: North, East, South, West.
- For each video we do NOT run YOLO on every frame - that would be
  far too slow on a normal laptop. We sample frames evenly across
  the video (default: at most 20 frames).
- On each sampled frame, YOLO counts the vehicles that are visible
  at that moment (car, motorcycle, bus, truck).
- The result for an approach is the AVERAGE number of vehicles
  visible per frame. That average is our traffic density estimate:
  "on average, how many vehicles are waiting/present on this road".

Honest limitations of this version:
- This is a density estimate, not a count of unique vehicles.
  The same vehicle appears in many frames, and that is intended -
  we want "how crowded is this road", not "how many passed in total".
- Queue length is approximated by the total average vehicles per
  frame. Separating stopped (queued) vehicles from moving ones
  needs tracking, which is a later improvement.
- Detection quality depends on video quality, angle and lighting.
"""

import os
from collections import Counter

from traffic_algorithm import VEHICLE_CLASSES

# COCO class IDs used by the pretrained YOLO model
VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def average_counts(frame_counts):
    """Average a list of per-frame count dicts.

    frame_counts example: [{"car": 4, "bus": 1}, {"car": 6}, ...]
    Returns counts rounded to the nearest whole vehicle per class,
    plus the raw averages (useful for the report).
    """
    if not frame_counts:
        return ({name: 0 for name in VEHICLE_CLASSES},
                {name: 0.0 for name in VEHICLE_CLASSES})
    totals = Counter()
    for counts in frame_counts:
        for name in VEHICLE_CLASSES:
            totals[name] += counts.get(name, 0)
    n = len(frame_counts)
    averages = {name: round(totals[name] / n, 2) for name in VEHICLE_CLASSES}
    rounded = {name: int(round(averages[name])) for name in VEHICLE_CLASSES}
    return rounded, averages


def total_vehicles(counts):
    return sum(counts.get(name, 0) for name in VEHICLE_CLASSES)


def count_frame(frame, model, confidence=0.4):
    """Count vehicles in one frame with an already-loaded YOLO model."""
    results = model(frame, imgsz=640, conf=confidence, verbose=False)[0]
    counts = Counter()
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_IDS:
            counts[VEHICLE_IDS[cls]] += 1
    return dict(counts)


def get_model():
    """Load YOLOv8n once. Raises ImportError with a clear message
    if the detection libraries are not installed yet."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Detection libraries are not installed. Install them with: "
            "pip install -r backend/requirements.txt"
        )
    if not hasattr(get_model, "_model"):
        get_model._model = YOLO("yolov8n.pt")  # downloads once, ~6 MB
    return get_model._model


def analyze_video_file(path, max_samples=20, confidence=0.4):
    """Analyze one video file. Returns a dict with counts + averages.

    Raises ImportError if opencv / ultralytics are missing, and
    ValueError if the file cannot be read as a video.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "Detection libraries are not installed. Install them with: "
            "pip install -r backend/requirements.txt"
        )

    model = get_model()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Video could not be opened: {os.path.basename(path)}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {os.path.basename(path)}")

    # Evenly spaced frame indexes across the whole video
    samples = max(1, min(max_samples, total_frames))
    indexes = sorted({int(i * total_frames / samples) for i in range(samples)})

    frame_counts = []
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_counts.append(count_frame(frame, model, confidence))
    cap.release()

    if not frame_counts:
        raise ValueError(f"No readable frames in video: {os.path.basename(path)}")

    rounded, averages = average_counts(frame_counts)
    return {
        "counts": rounded,                 # used by the signal algorithm
        "average_counts": averages,        # shown in the result, for honesty
        "frames_sampled": len(frame_counts),
        "video_fps": round(fps, 2),
        "video_total_frames": total_frames,
    }
