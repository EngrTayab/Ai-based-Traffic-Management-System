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

from traffic_algorithm import APPROACHES, VEHICLE_CLASSES

# COCO class IDs used by the pretrained YOLO model
VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class DetectionUnavailableError(RuntimeError):
    """Detection cannot run on this machine yet (libraries missing
    or installed but unable to load). The API turns this into a
    clear 503 message instead of a confusing 500 error."""


INSTALL_HINT = (
    "Detection libraries are not installed. Install them with: "
    "pip install -r backend/requirements.txt"
)

TORCH_DLL_HINT = (
    "PyTorch is installed but Windows could not load its DLL files "
    "(WinError 126). This usually means the Microsoft Visual C++ "
    "Redistributable (x64) is missing. Install it from "
    "https://aka.ms/vs/17/release/vc_redist.x64.exe , restart "
    "PowerShell, and test with: python -c \"import torch\" . "
    "Original error: "
)


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
    """Load YOLOv8n once. Raises DetectionUnavailableError with a
    clear message if the libraries are missing, or if PyTorch is
    installed but its Windows DLLs cannot load."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        # Example: WinError 126 while loading torch\lib\shm.dll
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))
    if not hasattr(get_model, "_model"):
        try:
            get_model._model = YOLO("yolov8n.pt")  # downloads once, ~6 MB
        except OSError as exc:
            raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))
    return get_model._model


def analyze_video_file(path, max_samples=20, confidence=0.4):
    """Analyze one video file. Returns a dict with counts + averages.

    Raises ImportError if opencv / ultralytics are missing, and
    ValueError if the file cannot be read as a video.
    """
    try:
        import cv2
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))

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


# ---------------------------------------------------------------------
# Single wide video with regions of interest (ROI) - v0.5
#
# If ONE camera/video shows the whole intersection, all four
# approaches are visible in the same frame. The user draws one box
# (region) per approach on the video. YOLO still runs only ONCE per
# frame on the full picture; each vehicle is then assigned to the
# approach whose box contains the CENTRE of the vehicle's box.
# Cropping each region and running YOLO 4 times would be 4x slower
# on a normal laptop for the same answer.
#
# Regions are fractions of the frame (0..1), e.g.
#   {"North": {"x": 0.35, "y": 0.0, "w": 0.3, "h": 0.45}, ...}
# so they work on any video resolution.
# ---------------------------------------------------------------------


def detect_frame(frame, model, confidence=0.4, imgsz=640):
    """All vehicle detections in one frame, with box centres.

    Returns a list of {"name", "cx", "cy"} where cx/cy are the
    centre of the vehicle's box as fractions of the frame size.
    """
    height, width = frame.shape[:2]
    results = model(frame, imgsz=imgsz, conf=confidence, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_IDS:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append({
            "name": VEHICLE_IDS[cls],
            "cx": ((x1 + x2) / 2) / width,
            "cy": ((y1 + y2) / 2) / height,
        })
    return detections


def normalize_rois(rois):
    """Validate/clean region boxes coming from the dashboard.

    Keeps only known approaches with numeric x, y, w, h in range.
    Returns {approach: {"x","y","w","h"}} with values clamped so
    the box stays inside the frame. Raises ValueError if nothing
    usable remains.
    """
    cleaned = {}
    for approach in APPROACHES:
        box = rois.get(approach) if isinstance(rois, dict) else None
        if not isinstance(box, dict):
            continue
        try:
            x = float(box["x"]); y = float(box["y"])
            w = float(box["w"]); h = float(box["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(w, 1.0 - x)
        h = min(h, 1.0 - y)
        if w <= 0 or h <= 0:
            continue
        cleaned[approach] = {"x": x, "y": y, "w": w, "h": h}
    if not cleaned:
        raise ValueError("No valid regions were provided. Draw at "
                         "least one box on the video first.")
    return cleaned


def assign_to_roi(cx, cy, rois):
    """Which approach's box contains this point? None if outside all.

    Edges: left/top inclusive, right/bottom exclusive, so a vehicle
    exactly on a shared border is counted once, not twice. If boxes
    overlap, the first approach in N/E/S/W order wins.
    """
    for approach in APPROACHES:
        box = rois.get(approach)
        if not box:
            continue
        if (box["x"] <= cx < box["x"] + box["w"]
                and box["y"] <= cy < box["y"] + box["h"]):
            return approach
    return None


def split_counts_by_roi(detections, rois):
    """Split one frame's detections into per-approach count dicts."""
    per_approach = {a: Counter() for a in rois}
    for det in detections:
        approach = assign_to_roi(det["cx"], det["cy"], rois)
        if approach is not None:
            per_approach[approach][det["name"]] += 1
    return {a: dict(c) for a, c in per_approach.items()}


def analyze_intersection_video(path, rois, max_samples=20,
                               confidence=0.25, imgsz=1280):
    """Analyze ONE wide video that shows several approaches at once.

    Returns {"frames_sampled", "video_fps", "video_total_frames",
    "approaches": {approach: {"counts", "average_counts"}}}.

    Why different settings from analyze_video_file? A wide shot of a
    whole intersection makes every vehicle SMALL in the frame, and
    small vehicles are easy to miss. Measured on an aerial test
    clip, the roadside settings (imgsz 640, conf 0.4) found only
    ~1 vehicle per frame, while imgsz 1280 + conf 0.25 found ~3x
    more. It is slower per frame, which is acceptable here because
    only 20 frames are sampled. A straight-down (nadir) drone view
    stays hard even then - vehicles are roof-shaped rectangles that
    barely resemble the ground-level cars the model was trained on.
    """
    try:
        import cv2
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))

    rois = normalize_rois(rois)
    model = get_model()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Video could not be opened: {os.path.basename(path)}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {os.path.basename(path)}")

    samples = max(1, min(max_samples, total_frames))
    indexes = sorted({int(i * total_frames / samples) for i in range(samples)})

    frame_counts = {a: [] for a in rois}
    used_frames = 0
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        used_frames += 1
        split = split_counts_by_roi(
            detect_frame(frame, model, confidence, imgsz), rois)
        for approach in rois:
            frame_counts[approach].append(split[approach])
    cap.release()

    if not used_frames:
        raise ValueError(f"No readable frames in video: {os.path.basename(path)}")

    approaches = {}
    for approach in rois:
        rounded, averages = average_counts(frame_counts[approach])
        approaches[approach] = {"counts": rounded, "average_counts": averages}
    return {
        "frames_sampled": used_frames,
        "video_fps": round(fps, 2),
        "video_total_frames": total_frames,
        "approaches": approaches,
    }
