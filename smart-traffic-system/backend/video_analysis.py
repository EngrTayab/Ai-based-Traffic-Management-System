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

import base64
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


HOSTED_HINT = (
    "Video detection is not available on this hosted version. "
    "The full version with video counting runs on the project "
    "laptop and is demonstrated live at the expo."
)


def _install_hint():
    # Render (the free hosting used for the public link) sets the
    # RENDER environment variable automatically. There the detection
    # libraries are deliberately absent - its free memory cannot hold
    # them - so a visitor needs the hosted explanation, not a pip
    # command meant for the developer's own laptop.
    if os.environ.get("RENDER"):
        return HOSTED_HINT
    return (
        "Detection libraries are not installed. Install them with: "
        "pip install -r backend/requirements.txt"
    )


INSTALL_HINT = _install_hint()

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


def detect_frame_boxes(frame, model, confidence=0.4, imgsz=640):
    """All vehicle detections in one frame, with PIXEL box corners.

    Returns [{"name", "x1", "y1", "x2", "y2"}]. Counting uses this
    too (the counts are just the names tallied), so the counts and
    the dashboard proof image always come from the SAME single
    YOLO run per frame - no double work on a small laptop.
    """
    results = model(frame, imgsz=imgsz, conf=confidence, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_IDS:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in box.xyxy[0])
        boxes.append({"name": VEHICLE_IDS[cls],
                      "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


# Box colours for the proof image (BGR - OpenCV's order). The ROI
# colours match the ones the dashboard uses when drawing regions.
_PROOF_CLASS_COLORS = {"car": (0, 200, 0), "motorcycle": (0, 215, 255),
                       "bus": (255, 0, 200), "truck": (255, 200, 0)}
_PROOF_ROI_COLORS = {"North": (113, 113, 248), "East": (36, 191, 251),
                     "South": (153, 211, 52), "West": (250, 165, 96)}


def _encode_proof_image(cv2, frame, boxes, rois=None, max_width=800):
    """Draw detection boxes (and any ROI boxes) on a copy of the
    frame and return it as a base64 JPEG string for the dashboard:
    the user-visible "what the AI saw" evidence (v0.9).

    It must NEVER break the analysis itself: any failure here
    returns None and the counts still go through.
    """
    try:
        img = frame.copy()
        height, width = img.shape[:2]
        if rois:
            for approach, box in rois.items():
                color = _PROOF_ROI_COLORS.get(approach, (255, 255, 255))
                x1 = int(box["x"] * width)
                y1 = int(box["y"] * height)
                x2 = int((box["x"] + box["w"]) * width)
                y2 = int((box["y"] + box["h"]) * height)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, approach, (x1 + 4, max(16, y1 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for det in boxes:
            color = _PROOF_CLASS_COLORS.get(det["name"], (0, 255, 0))
            cv2.rectangle(img, (det["x1"], det["y1"]),
                          (det["x2"], det["y2"]), color, 2)
            cv2.putText(img, det["name"],
                        (det["x1"], max(14, det["y1"] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if width > max_width:
            scale = max_width / width
            img = cv2.resize(img, (max_width, int(height * scale)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def analyze_image_bytes(data):
    """One camera frame (JPEG/PNG bytes) in, counts + proof out.

    Used by the live camera mode (v0.10): the dashboard sends one
    frame every few seconds and the SAME single YOLO pass produces
    both the counts and the proof boxes - see detect_frame_boxes.
    Returns {"counts", "proof_image"}.
    Raises DetectionUnavailableError if detection cannot run, and
    ValueError if the bytes are not a readable image.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The camera frame could not be read as an image.")
    # Live frames usually show a road - or a screen playing one -
    # from a distance, so vehicles are SMALL in the frame. Use the
    # sensitive settings proven on the wide ROI videos (v0.5.1):
    # a larger detection size and a lower confidence than the
    # roadside defaults. One frame every few seconds makes the
    # extra computing time affordable. (v0.10.1 fix after a first
    # live test saw no vehicles through a webcam.)
    # v0.17: the live paths use the stronger live model (YOLOv8s)
    # at conf 0.25 - the nano model barely recognises the toy cars
    # on the expo model (measured: die-cast coupe 0.27 with v8n,
    # 0.76 with v8s). Video uploads keep the fast nano model.
    boxes = detect_frame_boxes(frame, _get_live_model(), 0.25, 1280)
    return {
        "counts": dict(Counter(d["name"] for d in boxes)),
        "proof_image": _encode_proof_image(cv2, frame, boxes),
    }


def analyze_image_rois_bytes(data, rois):
    """One camera frame showing the WHOLE intersection, split into
    approaches by the drawn regions (the expo model mode, v0.14).

    The physical model use case: toy cars stand still in each lane
    of a table-top intersection, one camera watches the whole model,
    and this counts the cars per lane - they do not need to move,
    because the system counts vehicles PRESENT in the frame (the
    same density estimate as video analysis). ONE detection pass
    with the stronger live model at the sensitive settings
    (1280 / 0.25, v0.17) is split by each box's centre, exactly
    as in the single-video ROI analysis. When the fine-tuned
    toy-car model is installed (toy_model.pt next to this file,
    v0.18), it runs instead of the pretrained live model - it was
    trained on the actual toy cars used on the model, which the
    pretrained models cannot recognise from above.

    Returns {"approaches": {approach: {"counts": {...}}},
    "proof_image", "proof_frame_vehicles", "proof_frame_counted"}.
    Approaches without a drawn region are absent from the result
    (they are treated as empty by the caller). Raises
    DetectionUnavailableError / ValueError like analyze_image_bytes.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))

    rois = normalize_rois(rois)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The camera frame could not be read as an image.")
    toy_model = _get_toy_model()
    if toy_model is not None:
        # v0.27: conf 0.25 with the v5 weights (was 0.35 since v0.19,
        # chosen for v4). Tayyab's live frames 2026-11-13 showed
        # cars flickering in and out at 0.35 - the overlay showed
        # boxes but the settled count (2nd-highest of 3 frames)
        # stayed 0 on North/South/West. Measured on his 4 real
        # frames: 0.25 catches +1 car on the far/dark frame and +2
        # on the current frame, with NO new boxes on empty scenes
        # and NO new watermark boxes on real frames (the 6 North
        # letter-boxes on frame_new are there at 0.35 too). The
        # settle rule still filters single-frame phantoms.
        boxes = detect_frame_boxes_by_name(frame, toy_model, 0.25, 1280)
    else:
        boxes = detect_frame_boxes(frame, _get_live_model(), 0.25, 1280)
    height, width = frame.shape[:2]
    per_approach = {approach: [] for approach in rois}
    for d in boxes:
        cx = (d["x1"] + d["x2"]) / 2 / width
        cy = (d["y1"] + d["y2"]) / 2 / height
        owner = assign_to_roi(cx, cy, rois)
        if owner is not None:
            per_approach[owner].append(d["name"])
    return {
        "approaches": {
            approach: {"counts": dict(Counter(names))}
            for approach, names in per_approach.items()
        },
        "proof_image": _encode_proof_image(cv2, frame, boxes, rois),
        "proof_frame_vehicles": len(boxes),
        "proof_frame_counted": sum(len(v) for v in per_approach.values()),
    }


def _get_tracking_model():
    """A SEPARATE YOLO instance for tracking.

    Tracking keeps state (the tracker remembers vehicles between
    frames), so it must never share the plain detection model -
    and each tracking run starts from a clean tracker.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))
    try:
        return YOLO("yolov8n.pt")
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))


def _get_live_model():
    """A SEPARATE, stronger YOLO instance for the LIVE camera paths
    (v0.17, from Tayyab's toy-car tests on the expo model).

    The live paths send one frame every few seconds, so they can
    afford YOLOv8s where the video paths (20 sampled frames per
    video, several videos) stay on the fast YOLOv8n. Measured
    reason: the nano model barely recognises toy cars - a die-cast
    coupe scored 0.27 with v8n (below the live threshold) but 0.76
    with v8s, and a die-cast muscle car was labelled "truck" by
    v8n but "car" by v8s. Loaded once and reused (detection is
    stateless, unlike the tracker). Downloads once, ~22 MB.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))
    if not hasattr(_get_live_model, "_model"):
        try:
            _get_live_model._model = YOLO("yolov8s.pt")
        except OSError as exc:
            raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))
    return _get_live_model._model


def detect_frame_boxes_by_name(frame, model, confidence=0.3, imgsz=1280):
    """Detections from a CUSTOM-trained model (the toy-car model).

    detect_frame_boxes() filters by COCO class IDs, which only exist
    in the pretrained models. A fine-tuned model has its OWN class
    list (the toy model has a single class, {0: "car"}), so this
    variant reads each box's class NAME from the model's own names
    and keeps the vehicle classes. Same return shape as
    detect_frame_boxes(), so counting/proof code is shared.
    """
    results = model(frame, imgsz=imgsz, conf=confidence, verbose=False)[0]
    names = results.names
    boxes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        name = names.get(cls) if isinstance(names, dict) else names[cls]
        if name not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in box.xyxy[0])
        boxes.append({"name": name,
                      "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


_TOY_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "toy_model.pt")


def _get_toy_model():
    """The fine-tuned toy-car model (v0.18), or None if not installed.

    The trained file lives NEXT TO this file in the backend folder
    as toy_model.pt; it was fine-tuned on Tayyab's own toy cars
    (synthetic copy-paste scenes built from photos of the 12 cars).
    When the file is absent, the live-model path falls back to the
    stronger pretrained model exactly as in v0.17, so this code is
    safe to copy in before the trained file arrives. A file that is
    present but broken raises DetectionUnavailableError instead of
    falling back silently - a silent fallback would detect none of
    the toy cars and give no explanation. Loaded once and reused.
    """
    if not os.path.exists(_TOY_MODEL_PATH):
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))
    if not hasattr(_get_toy_model, "_model"):
        try:
            _get_toy_model._model = YOLO(_TOY_MODEL_PATH)
        except Exception as exc:
            raise DetectionUnavailableError(
                "The trained toy-car model (toy_model.pt) could not "
                "be loaded: " + str(exc))
    return _get_toy_model._model


TRACKING_HINT = (
    "Tracking could not run. It needs one small extra package "
    "called 'lap' (the tracker uses it to match vehicles between "
    "frames). Connect to the internet once and try again - the "
    "library usually installs it by itself - or install it "
    "manually with: pip install lap"
)


def track_video_file(path, frame_step=2, confidence=0.3,
                     max_tracked_frames=1200):
    """Count the UNIQUE vehicles in one video, using tracking.

    The normal analysis answers "how many vehicles are visible at
    a typical moment" (a density). This answers the other question
    (Tayyab's, 2026-11-06): "how many vehicles CAME AND PASSED in
    total". YOLO + ByteTrack gives every vehicle an ID and follows
    it across frames; each ID is counted once, with the class it
    was first seen as.

    Every frame_step-th frame is tracked (tracking every frame of
    a long video is slow on a normal laptop). To bound the time,
    at most max_tracked_frames are processed; if the video is
    longer, "partial" is True and the counts cover only the first
    part - the caller must say so.

    Honest limits: an ID can switch when a vehicle is hidden
    behind another for a while, and parked vehicles that stay in
    view are counted too (they did "appear" in the video). The
    number is a close measure of flow, not a perfect census.
    """
    try:
        import cv2
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))

    model = _get_tracking_model()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Video could not be opened: {os.path.basename(path)}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {os.path.basename(path)}")

    seen = {}  # track id -> vehicle class (first seen)
    tracked = 0
    idx = 0
    try:
        while tracked < max_tracked_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_step == 0:
                result = model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    imgsz=640, conf=confidence, verbose=False)[0]
                tracked += 1
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    for tid, cls in zip(boxes.id.tolist(),
                                        boxes.cls.tolist()):
                        tid = int(tid)
                        cls = int(cls)
                        if tid >= 0 and cls in VEHICLE_IDS and tid not in seen:
                            seen[tid] = VEHICLE_IDS[cls]
            idx += 1
    except DetectionUnavailableError:
        raise
    except Exception as exc:
        # Most likely the tracker's 'lap' package is missing and
        # its automatic install could not run (no internet).
        raise DetectionUnavailableError(TRACKING_HINT + f" ({exc})")
    finally:
        cap.release()

    if tracked == 0:
        raise ValueError(f"No readable frames in video: {os.path.basename(path)}")

    unique_counts = {name: 0 for name in VEHICLE_CLASSES}
    for name in seen.values():
        unique_counts[name] += 1
    partial = idx < total_frames  # stopped before the video ended
    return {
        "unique_counts": unique_counts,
        "unique_total": len(seen),
        "frames_tracked": tracked,
        "video_total_frames": total_frames,
        "partial": partial,
    }


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
    best = None  # (vehicles found, frame, boxes): the proof frame
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        boxes = detect_frame_boxes(frame, model, confidence)
        frame_counts.append(dict(Counter(d["name"] for d in boxes)))
        if best is None or len(boxes) > best[0]:
            best = (len(boxes), frame.copy(), boxes)
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
        # v0.9: the sampled frame with the most vehicles found,
        # with the AI's boxes drawn on it (None if drawing failed).
        # proof_frame_vehicles says how many vehicles THAT frame
        # shows - it is the busiest moment, so it is higher than
        # the averages above; the dashboard explains this under
        # the picture (v0.12.1, after Tayyab spotted the mismatch).
        "proof_image": (_encode_proof_image(cv2, best[1], best[2])
                        if best else None),
        "proof_frame_vehicles": best[0] if best else 0,
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


def _count_in_rois(frame, boxes, rois):
    """How many of these pixel boxes have their centre inside a
    drawn region (and so were counted for an approach)?"""
    height, width = frame.shape[:2]
    counted = 0
    for d in boxes:
        cx = (d["x1"] + d["x2"]) / 2 / width
        cy = (d["y1"] + d["y2"]) / 2 / height
        if assign_to_roi(cx, cy, rois) is not None:
            counted += 1
    return counted


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
    best = None  # (vehicles found, frame, boxes): the proof frame
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        used_frames += 1
        boxes = detect_frame_boxes(frame, model, confidence, imgsz)
        height, width = frame.shape[:2]
        detections = [{
            "name": d["name"],
            "cx": (d["x1"] + d["x2"]) / 2 / width,
            "cy": (d["y1"] + d["y2"]) / 2 / height,
        } for d in boxes]
        split = split_counts_by_roi(detections, rois)
        for approach in rois:
            frame_counts[approach].append(split[approach])
        if best is None or len(boxes) > best[0]:
            best = (len(boxes), frame.copy(), boxes)
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
        # v0.9: busiest sampled frame with the drawn regions and
        # the AI's detection boxes on it (None if drawing failed).
        # proof_frame_counted: of the vehicles in that frame, how
        # many fell inside a drawn region and were counted - the
        # picture shows every detection, but only these enter the
        # numbers (v0.12.1).
        "proof_image": (_encode_proof_image(cv2, best[1], best[2], rois)
                        if best else None),
        "proof_frame_vehicles": best[0] if best else 0,
        "proof_frame_counted": (_count_in_rois(best[1], best[2], rois)
                                if best else 0),
    }


# ---------------------------------------------------------------------
# Auto-fit regions (v0.29)
#
# Tayyab's complaint 2026-11-13: the hand-drawn model regions spill
# outside the roads, and sometimes TWO regions land on ONE road.
# This detector finds the roads in a camera frame and fits one
# region inside each road, so the dashboard button "Auto-fit
# regions to roads" can snap the boxes into place. It uses NO
# YOLO model - only the fact that the model roads are BLACK chart
# paper on a light table (max channel < 85), while the red cloth,
# the thermocol and the table are all brighter.
#
# How it works: (1) a dark-pixel mask, cleaned up (letterbox bars
# removed, dashes closed); (2) the intersection centre, found by a
# grid search scoring the harmonic mean of the four cardinal ray
# lengths; (3) each arm traced outward from an on-road anchor, with
# a run-containment rule and a corridor that stop the trace
# wandering onto neighbouring dark clutter; (4) an axis-aligned
# rectangle inscribed in each traced arm, placed on the fitted
# centreline, skipping the intersection square; (5) a snap step
# that nudges each rectangle onto the road (only for a real gain).
# An arm whose trace is untrustworthy (axis > 35 degrees off its
# cardinal direction, or too short) is reported as MISSING instead
# of returning a bad box - the dashboard then keeps the user's
# hand-drawn box for that road. Measured on his 4 real frames:
# 2-3 good regions per frame, no slivers, no overlaps.
# ---------------------------------------------------------------------

_AUTO_ROAD_MAX_CHANNEL = 85
_AUTO_WORK_WIDTH = 640
_AUTO_DIRS = {"North": (0, -1), "East": (1, 0),
              "South": (0, 1), "West": (-1, 0)}


def _auto_road_mask(frame):
    """Dark pixels = road. The black chart roads read max(B,G,R)
    35-55; the red cloth 121-160, thermocol 202-224, the table
    75-116 - so 85 separates the roads from everything else on
    his table. Letterbox bars (whole rows/columns dark) are
    stripped, and a closing fills the white dashes/zebra so each
    road is one solid blob."""
    import cv2
    import numpy as np
    dark = np.max(frame, axis=2) < _AUTO_ROAD_MAX_CHANNEL
    h, w = dark.shape
    row_frac = dark.mean(axis=1)
    col_frac = dark.mean(axis=0)
    dark[row_frac > 0.92, :] = False
    dark[:, col_frac > 0.92] = False
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(dark.astype(np.uint8),
                              cv2.MORPH_CLOSE, kernel)
    kernel2 = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2)
    return opened.astype(bool)


def _auto_box_sum(integral, x1, y1, x2, y2):
    h, w = integral.shape[0] - 1, integral.shape[1] - 1
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(integral[y2, x2] - integral[y1, x2]
                 - integral[y2, x1] + integral[y1, x1])


def _auto_ray_length(integral, cx, cy, dx, dy, step=6, half_win=6,
                     max_gap_steps=20):
    """How far the road runs from (cx, cy) in direction (dx, dy):
    walk out in 6px steps; a step counts as road when >40% of the
    +/-6px window across the ray is road. Gaps up to 20 steps
    (sheet gaps, cars) are tolerated."""
    h, w = integral.shape[0] - 1, integral.shape[1] - 1
    px, py = -dy, dx
    farthest, gap = 0, 0
    dist = 0
    while True:
        dist += step
        x, y = cx + dx * dist, cy + dy * dist
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        xa = sorted([int(round(x + px * -half_win)),
                     int(round(x + px * half_win)) + 1])
        ya = sorted([int(round(y + py * -half_win)),
                     int(round(y + py * half_win)) + 1])
        x1, x2 = xa
        y1, y2 = ya
        area = max(1, (min(w, x2) - max(0, x1))
                   * (min(h, y2) - max(0, y1)))
        if _auto_box_sum(integral, x1, y1, x2, y2) / area > 0.4:
            farthest, gap = dist, 0
        else:
            gap += 1
            if gap > max_gap_steps:
                break
    return farthest


def _auto_centre_score(integral, cx, cy):
    """Harmonic mean of the four cardinal ray lengths: all four
    arms must be long (a point on one road scores badly through
    its short sideways rays), but one ray capped by clutter (the
    red cloth over the West road) no longer vetoes the true
    centre the way a plain minimum did."""
    rays = [_auto_ray_length(integral, cx, cy, dx, dy)
            for dx, dy in _AUTO_DIRS.values()]
    rays = [max(r, 10.0) for r in rays]
    return len(rays) / sum(1.0 / r for r in rays), min(rays)


def _auto_find_centre(mask):
    """The intersection centre: the point in the middle of the
    frame with the longest roads in all four directions. Returns
    None when no point has a decent road in every direction (the
    model is not in frame)."""
    import cv2
    import numpy as np
    h, w = mask.shape
    integral = cv2.integral(mask.astype(np.uint8))
    best, best_score, best_min = None, -1.0, -1.0
    xs = np.arange(0.28, 0.721, 0.02) * w
    ys = np.arange(0.28, 0.721, 0.02) * h
    for cy in ys:
        for cx in xs:
            score, mn = _auto_centre_score(integral, cx, cy)
            if score > best_score:
                best_score, best_min, best = score, mn, (cx, cy)
    # Fine local search around the coarse winner.
    if best is not None:
        for cy in np.arange(best[1] - 0.03 * h, best[1] + 0.031 * h,
                            0.01 * h):
            for cx in np.arange(best[0] - 0.03 * w,
                                best[0] + 0.031 * w, 0.01 * w):
                score, mn = _auto_centre_score(integral, cx, cy)
                if score > best_score:
                    best_score, best_min, best = score, mn, (cx, cy)
    if best is None or best_min < 0.10 * min(h, w):
        return None
    return np.array(best, dtype=float)


def _auto_runs_in_scan(values, merge_gap):
    runs, start = [], None
    last_true = -10 ** 9
    for i, v in enumerate(values):
        if v:
            if start is None:
                start = i
            last_true = i
        elif start is not None and i - last_true > merge_gap:
            runs.append((start, last_true))
            start = None
    if start is not None:
        runs.append((start, last_true))
    return runs


def _auto_scan_cross_section(mask, point, perp, scan_half, merge_gap):
    """Road runs on the perpendicular line through `point`.

    Returns [(centre_offset, width)] for each merged run, offsets
    measured from the point along `perp`.
    """
    import numpy as np
    h, w = mask.shape
    offs = np.arange(-scan_half, scan_half + 1)
    sx = np.clip((point[0] + perp[0] * offs).astype(int), 0, w - 1)
    sy = np.clip((point[1] + perp[1] * offs).astype(int), 0, h - 1)
    runs = _auto_runs_in_scan(mask[sy, sx], merge_gap)
    return [((r[0] + r[1]) / 2 - scan_half, r[1] - r[0] + 1)
            for r in runs]


def _auto_trace_arm(mask, centre, direction):
    """Trace one arm: anchor on the road, then follow it outward.

    The trace cannot wander: every accepted run must CONTAIN the
    predicted centreline point, and the run's centre must stay
    inside a corridor around the cardinal ray through the anchor
    (+/-12% of the frame width - wide enough for a tilted road,
    narrow enough to stop the trace drifting onto the dark
    clothes/table beside the road, which the earlier versions
    did). Short gaps (sheet gaps, cars) are bridged by dead
    reckoning for up to 15 steps. Returns
    (samples, axis, road_half, axis_point) or
    (None, None, None, None).
    """
    import numpy as np
    h, w = mask.shape
    direction = np.array(direction, dtype=float)
    step = 4
    scan_half = int(0.16 * w)
    merge_gap = int(0.04 * w)
    corridor_half = 0.12 * w
    # --- Phase 1: anchor. Walk outward until a WIDE road run
    # (a real road cross-section, not a wire or a car edge) is
    # found on three consecutive steps. ---
    anchor = None
    perp = np.array([-direction[1], direction[0]])
    consecutive = 0
    dist = 0
    max_dist = 0.38 * min(h, w)
    while dist < max_dist:
        dist += step
        point = centre + direction * dist
        if (point[0] < 2 or point[0] >= w - 2
                or point[1] < 2 or point[1] >= h - 2):
            break
        runs = _auto_scan_cross_section(mask, point, perp, scan_half,
                                        merge_gap)
        wide = [r for r in runs if r[1] >= 0.05 * w
                and abs(r[0]) <= 0.08 * w]
        if wide:
            consecutive += 1
            if consecutive >= 3:
                best = min(wide, key=lambda r: abs(r[0]))
                anchor = point + perp * best[0]
                break
        else:
            consecutive = 0
    if anchor is None:
        return None, None, None, None
    # --- Phase 2: trace outward from the anchor. ---
    samples = []
    point = anchor.copy()
    direction = direction.copy()
    miss = 0
    cos_limit = np.cos(np.radians(25))
    cardinal = np.array(direction, dtype=float)
    while True:
        point = point + direction * step
        if (point[0] < 2 or point[0] >= w - 2
                or point[1] < 2 or point[1] >= h - 2):
            break
        perp = np.array([-direction[1], direction[0]])
        runs = _auto_scan_cross_section(mask, point, perp,
                                        int(0.10 * w), merge_gap)
        near = [r for r in runs
                if abs(r[0]) <= r[1] / 2 + 3]
        if near:
            # Corridor: the run's centre must stay near the
            # cardinal ray through the anchor.
            kept = []
            for r in near:
                cand = point + perp * r[0]
                lateral = abs(float(np.dot(cand - anchor,
                                           np.array([-cardinal[1],
                                                     cardinal[0]]))))
                if lateral <= corridor_half:
                    kept.append(r)
            near = kept
        if not near:
            miss += 1
            if miss > 15:
                break
            continue
        miss = 0
        best = min(near, key=lambda r: abs(r[0]))
        centre_pt = point + perp * best[0]
        samples.append((centre_pt, best[1]))
        point = centre_pt
        if len(samples) >= 6:
            recent = samples[-1][0] - samples[-6][0]
            n = np.linalg.norm(recent)
            if n > 1e-6:
                new_dir = recent / n
                if float(np.dot(new_dir, cardinal)) >= cos_limit:
                    direction = new_dir
    if len(samples) < 8:
        return None, None, None, None
    # Two-pass cleaning: the road's width is roughly constant, so
    # a sample far wider than the trace's low widths is a merged
    # neighbouring road or clutter, and a far narrower one is a
    # car splitting the run. Keep the longest contiguous run of
    # good samples (the anchor may sit in the crossing zone, so
    # the good run need not start at the first sample).
    widths_all = np.array([s[1] for s in samples], dtype=float)
    ref = float(np.percentile(widths_all, 20))
    good = (widths_all >= 0.6 * ref) & (widths_all <= 1.8 * ref)
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i, g in enumerate(good):
        if g:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    if best_len < 8:
        return None, None, None, None
    samples = samples[best_start:best_start + best_len]
    pts = np.array([s[0] for s in samples], dtype=float)
    widths = np.array([s[1] for s in samples], dtype=float)
    # Fit on the middle half: the first samples can be biased by
    # the anchor/crossing zone, the last ones by drift.
    lo, hi = len(samples) // 4, max(len(samples) // 4 + 1,
                                    3 * len(samples) // 4)
    fit_pts = pts[lo:hi]
    # +6px: the dark-pixel threshold clips a few px at the road's
    # edges, so the median run width slightly underestimates the
    # true road width.
    road_half = float(np.median(widths[lo:hi])) / 2.0 + 6.0
    if road_half < 0.015 * w or road_half > 0.20 * w:
        return None, None, None, None
    centroid = fit_pts.mean(axis=0)
    cov = np.cov((fit_pts - centroid).T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))].astype(float)
    if np.dot(axis, centroid - centre) < 0:
        axis = -axis
    # The axis LINE passes through the fitted centroid, not the
    # grid centre: return it so the region is placed on the
    # road's true centreline.
    return samples, axis, road_half, centroid


def _auto_refine_centre(traced):
    """Least-squares intersection of the fitted arm axes.

    Only clean arms vote: an axis more than 30 degrees from its
    cardinal direction means the trace went wrong, and its line
    would drag the centre away.
    """
    import numpy as np
    arms = []
    for name, (samples, axis, _, axis_point) in traced.items():
        cardinal = np.array(_AUTO_DIRS[name], dtype=float)
        if abs(float(np.dot(axis, cardinal))) >= np.cos(np.radians(30)):
            arms.append((samples, axis, None, axis_point))
    if len(arms) < 2:
        return None
    a_mat = np.zeros((2, 2))
    b_vec = np.zeros(2)
    for _, axis, _, axis_point in arms:
        proj = np.eye(2) - np.outer(axis, axis)
        a_mat += proj
        b_vec += proj @ axis_point
    try:
        return np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        return None


def _auto_snap_to_road(mask, ii, rect):
    """Nudge a pixel rect (x1, y1, x2, y2) by up to 36px in x/y
    to maximise the fraction of road pixels inside it. Only moves
    when the best shift beats the original by >2% (a tiny gain is
    not worth moving the region away from its traced position).
    The integral image `ii` is built from a DILATED road mask, so
    parked cars (bright holes in the mask) do not push the region
    off the cars."""
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = rect
    rw, rh = x2 - x1, y2 - y1
    orig = _auto_box_sum(ii, x1, y1, x2, y2) / float(rw * rh)
    best, best_frac = rect, orig
    for dy in (-36, -24, -12, 0, 12, 24, 36):
        for dx in (-36, -24, -12, 0, 12, 24, 36):
            nx1 = min(max(x1 + dx, 0), w - rw)
            ny1 = min(max(y1 + dy, 0), h - rh)
            nx2, ny2 = nx1 + rw, ny1 + rh
            road = _auto_box_sum(ii, nx1, ny1, nx2, ny2)
            frac = road / float(rw * rh)
            if frac > best_frac + 1e-9:
                best_frac, best = frac, (nx1, ny1, nx2, ny2)
    # Only move for a real gain: a tiny coverage win is not worth
    # moving the region away from where the trace put it.
    if best_frac < orig + 0.02:
        return rect
    return best


def _auto_arm_region(samples, axis, road_half, axis_point, centre,
                     shape, crossing_half=None):
    """Axis-aligned rectangle inside one traced arm.

    The arm is a rotated rectangle (axis, half-length, road_half).
    An axis-aligned rectangle with half-sizes (hw, hh) reaches
    hw*|ax| + hh*|ay| along the arm and hw*|ay| + hh*|ax| across
    it (worst corner), so the (hw, hh) that exactly spans the
    desired length (the traced span minus a margin) and width
    (the road half-width minus a small safety margin) is the
    solution of that 2x2 system - it reduces to (width, length)
    for a vertical arm and (length, width) for a horizontal one.
    Returns {x,y,w,h} fractions, or None if unusable.
    """
    import numpy as np
    h, w = shape
    pts = np.array([s[0] for s in samples], dtype=float)
    # Distances along the axis, measured from the centre's own
    # projection (so the near skip still excludes the intersection
    # square), but the region's centre sits ON THE FITTED LINE -
    # the road's true centreline - not on the ray from the grid
    # centre, which can be offset from the road. The near skip is
    # the CROSSING road's half-width (the intersection's extent
    # along this arm), falling back to the arm's own width.
    axis_point = np.array(axis_point, dtype=float)
    centre = np.array(centre, dtype=float)
    proj_c = float((centre - axis_point) @ axis)
    proj = (pts - axis_point) @ axis
    far = float(proj.max())
    skip_half = crossing_half if crossing_half else road_half
    near = proj_c + 1.1 * skip_half
    if far - near < 0.08 * min(h, w):
        return None
    mid = (near + far) / 2.0
    half_len = (far - near) / 2.0 - 0.01 * min(h, w)
    if half_len <= 0:
        return None
    mid_pt = axis_point + axis * mid
    # The rectangle's corners must stay inside the arm: its
    # along extent within half_len, its across extent within the
    # road half-width (minus a small safety margin). Solving both
    # constraints tight gives the largest fitting rectangle.
    want_len = half_len
    want_wid = max(road_half - 0.006 * min(h, w), 0.5 * road_half)
    a, b = abs(float(axis[0])), abs(float(axis[1]))
    det = a * a - b * b
    if abs(det) < 0.2:
        # Arm near 45 degrees: fall back to a conservative square.
        hw = hh = 0.5 * min(want_len, want_wid)
    else:
        hw = (a * want_len - b * want_wid) / det
        hh = (-b * want_len + a * want_wid) / det
    if hw <= 0.015 * w or hh <= 0.04 * h:
        return None
    cx, cy = mid_pt
    x1 = max(0.0, (cx - hw) / w)
    y1 = max(0.0, (cy - hh) / h)
    x2 = min(1.0, (cx + hw) / w)
    y2 = min(1.0, (cy + hh) / h)
    if x2 - x1 < 0.03 or y2 - y1 < 0.08:
        return None
    return {"x": round(float(x1), 3), "y": round(float(y1), 3),
            "w": round(float(x2 - x1), 3),
            "h": round(float(y2 - y1), 3)}


def _auto_measure_width(mask, centre, direction):
    """Road half-width along a ray, measured by perpendicular
    scans (v0.30: used for an estimate when no arm was traced at
    all, so there is no traced width to borrow). Samples three
    points along the ray and takes the median run width. Returns
    the half-width in px, or None when no road run is found."""
    import numpy as np
    h, w = mask.shape
    direction = np.array(direction, dtype=float)
    centre = np.array(centre, dtype=float)
    if direction[0] > 0:
        edge = w - centre[0]
    elif direction[0] < 0:
        edge = centre[0]
    elif direction[1] > 0:
        edge = h - centre[1]
    else:
        edge = centre[1]
    perp = np.array([-direction[1], direction[0]])
    widths = []
    for frac in (0.35, 0.5, 0.65):
        point = centre + direction * (edge * frac)
        runs = _auto_scan_cross_section(mask, point, perp,
                                        int(0.16 * w), int(0.04 * w))
        near = [r for r in runs
                if abs(r[0]) <= r[1] / 2 + 3 and r[1] >= 0.03 * w]
        if near:
            widths.append(min(near, key=lambda r: abs(r[0]))[1])
    if not widths:
        return None
    return float(np.median(widths)) / 2.0 + 6.0


def _auto_estimate_region(name, centre, road_half, snap_integral,
                          shape):
    """An axis-aligned region for an arm the trace could not
    follow (v0.30): along the arm's cardinal ray from the
    centre, past the intersection square, to near the frame
    edge, as wide as the measured road. The box must sit mostly
    on road (checked against the dilated mask, so parked cars do
    not count against it) - otherwise it returns None and the
    arm stays honestly missing instead of a box over bare table.
    """
    import numpy as np
    h, w = shape
    direction = np.array(_AUTO_DIRS[name], dtype=float)
    centre = np.array(centre, dtype=float)
    margin = 0.02 * min(h, w)
    if direction[0] > 0:
        edge = w - centre[0]
    elif direction[0] < 0:
        edge = centre[0]
    elif direction[1] > 0:
        edge = h - centre[1]
    else:
        edge = centre[1]
    near = 1.1 * road_half
    far = edge - margin
    if far - near < 0.08 * min(h, w):
        return None
    mid = (near + far) / 2.0
    half_len = (far - near) / 2.0 - 0.01 * min(h, w)
    if half_len <= 0:
        return None
    half_wid = max(road_half - 0.006 * min(h, w), 0.5 * road_half)
    mid_pt = centre + direction * mid
    if direction[0] == 0:  # North/South: tall box
        x1, x2 = mid_pt[0] - half_wid, mid_pt[0] + half_wid
        y1, y2 = mid_pt[1] - half_len, mid_pt[1] + half_len
    else:  # East/West: wide box
        x1, x2 = mid_pt[0] - half_len, mid_pt[0] + half_len
        y1, y2 = mid_pt[1] - half_wid, mid_pt[1] + half_wid
    x1 = max(0.0, x1 / w)
    y1 = max(0.0, y1 / h)
    x2 = min(1.0, x2 / w)
    y2 = min(1.0, y2 / h)
    if x2 - x1 < 0.03 or y2 - y1 < 0.08:
        return None
    px = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    area = max(1, (px[2] - px[0]) * (px[3] - px[1]))
    frac = _auto_box_sum(snap_integral, px[0], px[1],
                         px[2], px[3]) / area
    if frac < 0.35:
        return None
    return {"x": round(float(x1), 3), "y": round(float(y1), 3),
            "w": round(float(x2 - x1), 3),
            "h": round(float(y2 - y1), 3)}


def detect_road_regions(frame):
    """Find the four road regions in a model frame (v0.29).

    `frame` is a BGR image (numpy array) of the table-top model.
    Returns (regions, found, missing, estimated): regions maps
    approach name to {x, y, w, h} fractions for every road with a
    box; found lists the arms traced on the picture itself;
    estimated lists the arms placed from the crossing's own
    geometry instead (v0.30: centre + measured road width, gated
    on the box sitting mostly on road); missing lists the arms
    with no box at all. A road that cannot be traced reliably
    AND cannot be placed on road is reported missing (never a
    guessed box over bare table).
    Raises DetectionUnavailableError when cv2/numpy are absent.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))
    small = frame
    if frame.shape[1] > _AUTO_WORK_WIDTH:
        scale = _AUTO_WORK_WIDTH / frame.shape[1]
        small = cv2.resize(frame, (_AUTO_WORK_WIDTH,
                                   int(frame.shape[0] * scale)))
    h, w = small.shape[:2]
    mask = _auto_road_mask(small)
    # Dilated mask for the snap step (see _auto_snap_to_road):
    # fills the car-sized holes so regions are not pushed off
    # the cars. Tracing uses the undilated mask.
    dil_kernel = np.ones((31, 31), np.uint8)
    snap_mask = cv2.dilate(mask.astype(np.uint8), dil_kernel)
    snap_integral = cv2.integral(snap_mask)
    centre = _auto_find_centre(mask)
    regions, found, missing = {}, [], []
    if centre is None:
        return regions, found, ["North", "East", "South", "West"], []
    traced = {}
    for name, direction in _AUTO_DIRS.items():
        out = _auto_trace_arm(mask, centre, direction)
        if out[0] is not None:
            traced[name] = out
    refined = _auto_refine_centre(traced)
    if refined is not None:
        # Sanity: the refined centre must stay near the grid one.
        if np.linalg.norm(refined - centre) < 0.10 * min(h, w):
            centre = refined
    perp = {"North": ("East", "West"), "South": ("East", "West"),
            "East": ("North", "South"), "West": ("North", "South")}
    for name in ("North", "East", "South", "West"):
        entry = traced.get(name)
        region = None
        if entry:
            # The fitted axis must point along its own road: more
            # than 35 degrees off means the trace followed the
            # wrong feature (a sliver region is worse than none).
            # The known slivers are 48+ degrees off; a genuine
            # arm (his frame4 East) sits at 31 degrees.
            cardinal = np.array(_AUTO_DIRS[name], dtype=float)
            if abs(float(np.dot(entry[1], cardinal))) >= np.cos(
                    np.radians(35)):
                cross = [traced[p][2] for p in perp[name]
                         if p in traced]
                cross_half = (sum(cross) / len(cross)) if cross else None
                region = _auto_arm_region(entry[0], entry[1], entry[2],
                                          entry[3], centre, (h, w),
                                          cross_half)
                if region is not None:
                    px = (int(region["x"] * w), int(region["y"] * h),
                          int((region["x"] + region["w"]) * w),
                          int((region["y"] + region["h"]) * h))
                    sx1, sy1, sx2, sy2 = _auto_snap_to_road(
                        mask, snap_integral, px)
                    region = {"x": round(sx1 / w, 3),
                              "y": round(sy1 / h, 3),
                              "w": round((sx2 - sx1) / w, 3),
                              "h": round((sy2 - sy1) / h, 3)}
        if region:
            regions[name] = region
            found.append(name)
        else:
            missing.append(name)
    # v0.30: arms the trace could not follow still get a box,
    # placed from the crossing's own geometry (the centre + the
    # measured road width), so one press fits all four regions.
    # Estimated boxes are reported separately - the dashboard
    # says which roads were found on the picture and which were
    # placed by estimate, and the user can drag an estimate that
    # is off. The estimate is snapped onto the road like a traced
    # region, and the road-fraction gate inside the estimator
    # keeps boxes off bare table.
    estimated = []
    if missing:
        traced_halves = [entry[2] for entry in traced.values()]
        for name in list(missing):
            if traced_halves:
                road_half = sum(traced_halves) / len(traced_halves)
            else:
                road_half = _auto_measure_width(
                    mask, centre, _AUTO_DIRS[name])
            if road_half is None:
                continue
            region = _auto_estimate_region(name, centre, road_half,
                                           snap_integral, (h, w))
            if region is not None:
                px = (int(region["x"] * w), int(region["y"] * h),
                      int((region["x"] + region["w"]) * w),
                      int((region["y"] + region["h"]) * h))
                sx1, sy1, sx2, sy2 = _auto_snap_to_road(
                    mask, snap_integral, px)
                region = {"x": round(sx1 / w, 3),
                          "y": round(sy1 / h, 3),
                          "w": round((sx2 - sx1) / w, 3),
                          "h": round((sy2 - sy1) / h, 3)}
                regions[name] = region
                estimated.append(name)
                missing.remove(name)
    return regions, found, missing, estimated


def auto_regions_from_bytes(data):
    """One camera frame (JPEG/PNG bytes) in, fitted road regions
    + proof image out (v0.29). Used by the dashboard's "Auto-fit
    regions to roads" button in model mode.

    Returns {"regions", "found", "missing", "estimated",
    "proof_image"}: the proof image is the frame with the NEW
    regions drawn on it, so the user can see exactly what was
    found before the boxes move. Raises
    DetectionUnavailableError / ValueError like analyze_image_bytes.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise DetectionUnavailableError(INSTALL_HINT)
    except OSError as exc:
        raise DetectionUnavailableError(TORCH_DLL_HINT + str(exc))
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The camera frame could not be read as an image.")
    regions, found, missing, estimated = detect_road_regions(frame)
    return {
        "regions": regions,
        "found": found,
        "missing": missing,
        "estimated": estimated,
        "proof_image": _encode_proof_image(cv2, frame, [], regions),
    }
