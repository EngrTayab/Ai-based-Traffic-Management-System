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

import os
import shutil
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import storage
from simulator import SCENARIOS, run_comparison
from traffic_algorithm import (
    APPROACHES,
    build_cycle,
    check_no_conflicting_greens,
    compute_green_times,
    load_config,
)
from video_analysis import ALLOWED_EXTENSIONS, analyze_video_file, total_vehicles

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
STATE = {"mode": "adaptive", "traffic": DEMO_TRAFFIC, "source": "demo"}


class TrafficInput(BaseModel):
    # {"North": {"counts": {"car": 5, ...}, "queue": 3}, ...}
    traffic: dict
    mode: str = "adaptive"


@app.get("/api/status")
def get_status():
    """Current counts, levels, green times and the live signal phase."""
    details = compute_green_times(STATE["traffic"], CONFIG, mode=STATE["mode"])
    phases = build_cycle(details, CONFIG)
    total_cycle = sum(p["duration"] for p in phases)
    elapsed = int(time.time() - START_TIME) % total_cycle

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

    return {
        "source": STATE["source"],
        "mode": STATE["mode"],
        "approaches": details,
        "current_phase": current,
        "cycle_seconds": total_cycle,
        "safety_check_passed": check_no_conflicting_greens(phases),
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
    details = compute_green_times(payload.traffic, CONFIG, mode=payload.mode)
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
            except ImportError as exc:
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
    details = compute_green_times(traffic, CONFIG, mode=STATE["mode"])
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
