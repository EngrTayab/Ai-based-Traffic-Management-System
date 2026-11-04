"""
FastAPI backend for the AI-Based Smart Traffic Light Management System.

Run it (PowerShell, from the smart-traffic-system folder):
    cd backend
    uvicorn main:app --reload --port 8000

Then open:
    http://127.0.0.1:8000          -> dashboard
    http://127.0.0.1:8000/docs     -> interactive API documentation

No login is used: this is a local expo demo (decided 2026-10-28).
Endpoints follow the SRS sample API (Section 5.4), simplified where
the SRS itself marked items optional.
"""

import json
import os
import shutil
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import storage
from simulator import SCENARIOS, run_comparison
from traffic_algorithm import (
    APPROACHES,
    apply_gap_out,
    build_cycle,
    check_no_conflicting_greens,
    compute_green_times,
    estimate_remaining_vehicles,
    load_config,
    scale_counts,
)
from video_analysis import (
    ALLOWED_EXTENSIONS,
    DetectionUnavailableError,
    analyze_intersection_video,
    analyze_video_file,
    total_vehicles,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
CONFIG = load_config(os.path.join(BASE_DIR, "config.json"))
START_TIME = time.time()

app = FastAPI(title="Smart Traffic Light Management System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo only
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo traffic used when no video is connected yet. This is clearly
# labelled "demo" in the response - it is NOT a real measurement.
DEMO_TRAFFIC = {
    "North": {"counts": {"car": 14, "motorcycle": 6, "bus": 1, "truck": 1}, "queue": 10},
    "East": {"counts": {"car": 4, "motorcycle": 2, "bus": 0, "truck": 0}, "queue": 3},
    "South": {"counts": {"car": 9, "motorcycle": 4, "bus": 1, "truck": 0}, "queue": 6},
    "West": {"counts": {"car": 2, "motorcycle": 1, "bus": 0, "truck": 0}, "queue": 1},
}

# The dashboard can switch this between "adaptive" and "fixed".
# "anchor" is the moment the current traffic was measured. The live
# cycle clock and the queue-drain estimate both run from the anchor,
# so uploading new videos restarts the cycle with fresh counts.
STATE = {"mode": "adaptive", "traffic": DEMO_TRAFFIC, "source": "demo",
         "anchor": START_TIME}

DRAIN_NOTE = (
    "Live display estimate: while an approach is green, its counts "
    "drain at about 1 vehicle per "
    f"{CONFIG.get('discharge_seconds_per_vehicle', 2.0):g} seconds "
    "(modelled queue discharge - it is not a new detection)."
)


class TrafficInput(BaseModel):
    # {"North": {"counts": {"car": 5, ...}, "queue": 3}, ...}
    traffic: dict
    mode: str = "adaptive"


@app.get("/api/status")
def get_status():
    """Current counts, levels, green times and the live signal phase.

    The counts shown are the measured counts MINUS an estimated
    discharge: while an approach is green, its vehicles are modelled
    as leaving (see traffic_algorithm.estimate_remaining_vehicles).
    Timings and levels always come from the original measurement -
    only the displayed counts/queue drain.
    """
    details = compute_green_times(STATE["traffic"], CONFIG, mode=STATE["mode"])
    gap_out_saved = apply_gap_out(details, CONFIG, mode=STATE["mode"])
    phases = build_cycle(details, CONFIG)
    total_cycle = sum(p["duration"] for p in phases)

    now = time.time()
    elapsed_total = now - STATE.get("anchor", START_TIME)
    elapsed = int(elapsed_total) % total_cycle

    current = None
    cursor = 0
    for phase in phases:
        if cursor <= elapsed < cursor + phase["duration"]:
            current = {
                "state": phase["state"],
                "green_approach": phase.get("green"),
                "approach": phase.get("approach", phase.get("green")),
                "seconds_left": cursor + phase["duration"] - elapsed,
            }
            break
        cursor += phase["duration"]

    # Where does each approach's green window sit inside the cycle?
    green_windows = {}
    cursor = 0
    for phase in phases:
        if phase["state"] == "green":
            green_windows[phase["green"]] = (cursor, phase["duration"])
        cursor += phase["duration"]

    discharge = CONFIG.get("discharge_seconds_per_vehicle", 2.0)
    all_clear = True
    for approach in APPROACHES:
        d = details[approach]
        initial_total = sum(d["counts"].values())
        start, duration = green_windows.get(approach, (0, 0))
        remaining = estimate_remaining_vehicles(
            initial_total, start, duration, total_cycle,
            elapsed_total, discharge,
        )
        d["vehicles_initial"] = initial_total
        d["vehicles_remaining_est"] = remaining
        d["road_clear"] = remaining == 0
        d["counts"] = scale_counts(d["counts"], remaining)
        d["queue"] = remaining
        if remaining > 0:
            all_clear = False

    # v0.5 behaviour Tayyab asked for:
    # - Once EVERY road has cleared, the system stops. The cycle
    #   does not loop forever giving empty roads green again; it
    #   waits until new traffic is measured (new video upload).
    # - A few seconds before the next green, that approach is
    #   flagged as "ready" so the dashboard can show its yellow
    #   early, as a get-ready warning.
    next_green = None
    seconds_to_next = None
    ready_approach = None
    if all_clear:
        current = {"state": "stopped", "green_approach": None,
                   "approach": None, "seconds_left": 0}
    else:
        deltas = [((start - elapsed) % total_cycle, a)
                  for a, (start, _dur) in green_windows.items()]
        deltas = [(d, a) for d, a in deltas if d > 0]
        if deltas:
            seconds_to_next, next_green = min(deltas)
            ready_seconds = CONFIG.get("ready_yellow_seconds", 3)
            if seconds_to_next <= ready_seconds:
                ready_approach = next_green

    return {
        "source": STATE["source"],
        "mode": STATE["mode"],
        "approaches": details,
        "current_phase": current,
        "cycle_seconds": total_cycle,
        "safety_check_passed": check_no_conflicting_greens(phases),
        "all_roads_clear": all_clear,
        "stopped": all_clear,
        "next_green_approach": next_green,
        "seconds_to_next_green": seconds_to_next,
        "ready_approach": ready_approach,
        "gap_out_saved_seconds": gap_out_saved,
        "note": DRAIN_NOTE,
    }


@app.post("/api/evaluate")
def evaluate(payload: TrafficInput):
    """Send counts for all 4 approaches, get levels + green times back."""
    if payload.mode not in ("adaptive", "fixed"):
        raise HTTPException(status_code=400, detail="mode must be 'adaptive' or 'fixed'")
    missing = [a for a in APPROACHES if a not in payload.traffic]
    if missing:
        raise HTTPException(status_code=400,
                            detail=f"Missing approaches: {', '.join(missing)}")
    STATE["traffic"] = payload.traffic
    STATE["mode"] = payload.mode
    STATE["source"] = "manual"
    STATE["anchor"] = time.time()
    details = compute_green_times(payload.traffic, CONFIG, mode=payload.mode)
    apply_gap_out(details, CONFIG, mode=payload.mode)
    return {"mode": payload.mode, "approaches": details,
            "cycle": build_cycle(details, CONFIG)}


class ModeInput(BaseModel):
    mode: str


@app.post("/api/mode")
def set_mode(payload: ModeInput):
    """Change only the mode (adaptive/fixed). Traffic numbers stay as they are."""
    if payload.mode not in ("adaptive", "fixed"):
        raise HTTPException(status_code=400, detail="mode must be 'adaptive' or 'fixed'")
    STATE["mode"] = payload.mode
    return {"mode": STATE["mode"], "source": STATE["source"]}


@app.post("/api/analyze-videos")
def analyze_videos(
    north: Optional[UploadFile] = File(None),
    east: Optional[UploadFile] = File(None),
    south: Optional[UploadFile] = File(None),
    west: Optional[UploadFile] = File(None),
):
    """Upload one traffic video per approach and get signal times back.

    One normal video shows one road, so each approach has its own
    upload. At least one video is required; approaches without a
    video are treated as having no measured traffic, and are listed
    in 'missing_approaches' in the response.
    """
    uploads = {"North": north, "East": east, "South": south, "West": west}
    present = {a: f for a, f in uploads.items() if f is not None and f.filename}
    if not present:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one approach video (north, east, south or west).",
        )

    tmp_dir = tempfile.mkdtemp(prefix="traffic_videos_")
    traffic = {}
    analysis = {}
    try:
        for approach, upload in present.items():
            ext = os.path.splitext(upload.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"{approach}: unsupported file type '{ext}'. "
                           f"Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                )
            path = os.path.join(tmp_dir, f"{approach.lower()}{ext}")
            with open(path, "wb") as out:
                shutil.copyfileobj(upload.file, out)
            try:
                result = analyze_video_file(path)
            except (ImportError, DetectionUnavailableError) as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{approach}: {exc}")
            analysis[approach] = result
            # Queue is approximated by the average vehicles visible
            # per frame - see video_analysis.py for why.
            traffic[approach] = {
                "counts": result["counts"],
                "queue": total_vehicles(result["counts"]),
            }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    missing = [a for a in APPROACHES if a not in traffic]
    for approach in missing:
        traffic[approach] = {"counts": {}, "queue": 0}

    STATE["traffic"] = traffic
    STATE["source"] = "video"
    STATE["anchor"] = time.time()
    details = compute_green_times(traffic, CONFIG, mode=STATE["mode"])
    apply_gap_out(details, CONFIG, mode=STATE["mode"])
    recommendation = sorted(
        (
            {"approach": a, "level": d["level"], "green_time": d["green_time"]}
            for a, d in details.items()
        ),
        key=lambda r: r["green_time"],
        reverse=True,
    )
    return {
        "source": "video",
        "mode": STATE["mode"],
        "approaches": details,
        "cycle": build_cycle(details, CONFIG),
        "recommendation": recommendation,
        "analysis": analysis,
        "missing_approaches": missing,
        "note": "Counts are the average vehicles visible per sampled frame "
                "in each approach video (density estimate, not unique "
                "vehicle totals). Queue is approximated by that average. "
                "Approaches without a video were treated as empty.",
    }


@app.post("/api/analyze-intersection")
def analyze_intersection(
    video: UploadFile = File(...),
    rois: str = Form(...),
):
    """Upload ONE wide video that shows the whole intersection.

    The dashboard sends the regions the user drew on the video as a
    JSON string: {"North": {"x": 0.35, "y": 0, "w": 0.3, "h": 0.45},
    ...} with all values as fractions of the frame (0..1). YOLO runs
    once per sampled frame; each vehicle is counted for the region
    its centre falls in. Approaches without a region are treated as
    empty, exactly like the four-video endpoint.
    """
    try:
        roi_map = json.loads(rois)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400,
                            detail="Regions (rois) must be valid JSON.")
    if not isinstance(roi_map, dict) or not roi_map:
        raise HTTPException(status_code=400,
                            detail="Draw at least one region on the video.")

    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. "
                   f"Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    tmp_dir = tempfile.mkdtemp(prefix="traffic_roi_")
    try:
        path = os.path.join(tmp_dir, f"intersection{ext}")
        with open(path, "wb") as out:
            shutil.copyfileobj(video.file, out)
        try:
            result = analyze_intersection_video(path, roi_map)
        except (ImportError, DetectionUnavailableError) as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    traffic = {}
    analysis = {}
    for approach, res in result["approaches"].items():
        analysis[approach] = {
            "counts": res["counts"],
            "average_counts": res["average_counts"],
            "frames_sampled": result["frames_sampled"],
        }
        traffic[approach] = {
            "counts": res["counts"],
            "queue": total_vehicles(res["counts"]),
        }
    missing = [a for a in APPROACHES if a not in traffic]
    for approach in missing:
        traffic[approach] = {"counts": {}, "queue": 0}

    STATE["traffic"] = traffic
    STATE["source"] = "video-roi"
    STATE["anchor"] = time.time()
    details = compute_green_times(traffic, CONFIG, mode=STATE["mode"])
    apply_gap_out(details, CONFIG, mode=STATE["mode"])
    recommendation = sorted(
        (
            {"approach": a, "level": d["level"], "green_time": d["green_time"]}
            for a, d in details.items()
        ),
        key=lambda r: r["green_time"],
        reverse=True,
    )
    return {
        "source": "video-roi",
        "mode": STATE["mode"],
        "approaches": details,
        "cycle": build_cycle(details, CONFIG),
        "recommendation": recommendation,
        "analysis": analysis,
        "missing_approaches": missing,
        "note": "One video was analyzed for all approaches at once: "
                "YOLO ran once per sampled frame and each vehicle was "
                "counted for the drawn region its centre fell in. "
                "Counts are average vehicles visible per sampled frame "
                "(density estimate, not unique vehicle totals). "
                "Approaches without a drawn region were treated as empty.",
    }


@app.get("/api/config")
def get_config():
    return CONFIG


@app.get("/api/scenarios")
def get_scenarios():
    return {"scenarios": SCENARIOS}


@app.post("/api/experiments/run")
def run_experiment(scenario: str = "medium", duration_seconds: int = 600):
    """Run fixed vs adaptive on the same simulated scenario and store it."""
    try:
        result = run_comparison(scenario, CONFIG, duration_seconds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result["id"] = storage.save_experiment(result)
    return result


@app.get("/api/experiments")
def get_experiments():
    return {"experiments": storage.list_experiments()}


@app.get("/")
def dashboard():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Dashboard file not found. See frontend/index.html"}


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
