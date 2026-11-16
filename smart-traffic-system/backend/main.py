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
import math
import os
import shutil
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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
    VEHICLE_CLASSES,
    DetectionUnavailableError,
    analyze_image_bytes,
    analyze_image_rois_bytes,
    auto_regions_from_bytes,
    analyze_intersection_video,
    analyze_video_file,
    average_counts,
    total_vehicles,
    track_video_file,
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


@app.on_event("startup")
def _warm_up_toy_model():
    """v0.22 - load the toy-car model ONCE when the server starts.

    Before v0.22 the model loaded lazily on the first camera frame,
    so the first live frame took a very long time on Tayyab's laptop
    (torch + weights + first inference, all at once) and the camera
    looked stuck. Warming up here moves that one-time cost to server
    start, where the console message explains the wait, and the first
    camera frame then costs only one inference. The warm-up runs one
    tiny blank frame - it detects nothing, it only builds the model.
    It must never break startup: on Render (no torch installed) it
    is skipped quietly and the light core still serves.
    """
    try:
        from video_analysis import _get_toy_model, detect_frame_boxes_by_name
        import numpy as np
        print("Warming up the toy-car model (first start takes a few seconds)...",
              flush=True)
        model = _get_toy_model()
        if model is None:
            print("Toy model file not found - live-model mode will use "
                  "the pretrained model.", flush=True)
            return
        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        detect_frame_boxes_by_name(blank, model, confidence=0.35,
                                   imgsz=320)
        print("Toy model ready.", flush=True)
    except Exception as exc:  # noqa: BLE001 - startup must not fail
        print("Toy-model warm-up skipped: " + str(exc), flush=True)

# Demo traffic: sample numbers used ONLY by the hosted demo link
# (Render), where no camera or video is available - the supervisor
# still sees a working cycle there. It is clearly labelled "demo"
# in the response - it is NOT a real measurement. On the laptop
# (v0.30) the system starts with NO measurement at all: the
# signals stay stopped until the camera or a video really measures
# traffic (Tayyab 2026-11-13: no guessing - live runs only when
# it is really live and the camera sees something).
DEMO_TRAFFIC = {
    "North": {"counts": {"car": 14, "motorcycle": 6, "bus": 1, "truck": 1}, "queue": 10},
    "East": {"counts": {"car": 4, "motorcycle": 2, "bus": 0, "truck": 0}, "queue": 3},
    "South": {"counts": {"car": 9, "motorcycle": 4, "bus": 1, "truck": 0}, "queue": 6},
    "West": {"counts": {"car": 2, "motorcycle": 1, "bus": 0, "truck": 0}, "queue": 1},
}

EMPTY_TRAFFIC = {
    approach: {"counts": {}, "queue": 0} for approach in APPROACHES
}


def _initial_state():
    """(traffic, source) at server start. The hosted demo keeps
    the sample numbers; everywhere else the system starts empty
    and stopped, waiting for a real measurement. The traffic is
    copied, so later updates never mutate the module constants."""
    base = DEMO_TRAFFIC if os.environ.get("RENDER") else EMPTY_TRAFFIC
    traffic = {a: {"counts": dict(d["counts"]), "queue": d["queue"]}
               for a, d in base.items()}
    return traffic, ("demo" if os.environ.get("RENDER") else "waiting")


# The dashboard can switch this between "adaptive" and "fixed".
# "anchor" is the moment the current traffic was measured. The live
# cycle clock and the queue-drain estimate both run from the anchor,
# so uploading new videos restarts the cycle with fresh counts.
_INITIAL_TRAFFIC, _INITIAL_SOURCE = _initial_state()
STATE = {"mode": "adaptive", "traffic": _INITIAL_TRAFFIC,
         "source": _INITIAL_SOURCE,
         "anchor": START_TIME, "override": None, "live": {}, "proofs": {},
         "tracking": {}, "pre_green": None, "manual_light": None}

DRAIN_NOTE = (
    "Live display estimate: while an approach is green, its counts "
    "drain at about 1 vehicle per "
    f"{CONFIG.get('discharge_seconds_per_vehicle', 2.0):g} seconds "
    "(modelled queue discharge - it is not a new detection)."
)

# v0.28: on the physical model the toy cars STAND STILL - they
# never drive away - so the drain above does not apply there.
# The live-model counts are the vehicles standing on each road.
MODEL_NOTE = (
    "Live model: the toy cars stand still on the model, so the "
    "counts do not drain during a green - they show the vehicles "
    "standing on each road. The cycle keeps rotating until the "
    "camera mode is switched off."
)


# ------------------------------------------------------------------
# Manual overrides (v0.8): emergency-vehicle priority and pedestrian
# crossing, triggered from the dashboard.
#
# In this prototype the trigger is a button. The pretrained YOLO
# model has no ambulance class, so automatic emergency-vehicle
# detection is future work; what is demonstrated and tested here is
# the priority LOGIC: end the current green safely (yellow, then
# all-red), give the emergency approach its green (or stop everyone
# for pedestrians), then resume the normal cycle where it paused.
# ------------------------------------------------------------------

def _override_total_seconds(ov):
    return ov["yellow_s"] + ov["allred_s"] + ov["active_s"]


def _current_green_approach():
    """Which approach is green in the NORMAL cycle right now (or None).

    Needed when an override starts, to know whose green must be
    ended safely before the override takes over.
    """
    details = compute_green_times(STATE["traffic"], CONFIG, mode=STATE["mode"])
    apply_gap_out(details, CONFIG, mode=STATE["mode"])
    phases = build_cycle(details, CONFIG, mode=STATE["mode"])
    total_cycle = sum(p["duration"] for p in phases)
    if not total_cycle:
        return None
    elapsed = int(time.time() - STATE.get("anchor", START_TIME)) % total_cycle
    cursor = 0
    for phase in phases:
        if cursor <= elapsed < cursor + phase["duration"]:
            return phase.get("green") if phase["state"] == "green" else None
        cursor += phase["duration"]
    return None


def _public_override(ov, now):
    total = _override_total_seconds(ov)
    t = now - ov["started_at"]
    stage = "clearing" if t < ov["yellow_s"] + ov["allred_s"] else "active"
    return {
        "kind": ov["kind"],
        "approach": ov["approach"],
        "seconds_left": max(0, math.ceil(total - t)),
        "stage": stage,
    }


def _start_override(kind, approach):
    """Begin an override and return its public description."""
    from_approach = _current_green_approach()
    yellow_s = 0
    if from_approach and from_approach != approach:
        yellow_s = int(CONFIG.get("yellow", 3))
    if kind == "emergency":
        active_s = int(CONFIG.get("emergency_green_seconds", 20))
    else:
        active_s = int(CONFIG.get("pedestrian_seconds", 12))
    STATE["manual_light"] = None  # v0.25: an override ends a manual test
    STATE["override"] = {
        "kind": kind,
        "approach": approach,
        "from_approach": from_approach,
        "started_at": time.time(),
        "yellow_s": yellow_s,
        "allred_s": int(CONFIG.get("all_red", 2)),
        "active_s": active_s,
    }
    return _public_override(STATE["override"], time.time())


def _resolve_override(now):
    """(current_phase, public_override) while an override runs.

    Returns (None, None) when no override is active. An expired
    override clears itself here and pushes the cycle anchor forward
    by its full length, so the normal cycle resumes exactly where
    it paused - the override never steals green time from anyone.
    """
    ov = STATE.get("override")
    if not ov:
        return None, None
    total = _override_total_seconds(ov)
    t = now - ov["started_at"]
    if t >= total:
        STATE["anchor"] = STATE.get("anchor", START_TIME) + total
        STATE["override"] = None
        return None, None
    if t < ov["yellow_s"]:
        phase = {"state": "yellow", "green_approach": None,
                 "approach": ov["from_approach"],
                 "seconds_left": math.ceil(ov["yellow_s"] - t)}
    elif t < ov["yellow_s"] + ov["allred_s"]:
        phase = {"state": "all_red", "green_approach": None,
                 "approach": None,
                 "seconds_left": math.ceil(ov["yellow_s"] + ov["allred_s"] - t)}
    elif ov["kind"] == "emergency":
        phase = {"state": "green", "green_approach": ov["approach"],
                 "approach": ov["approach"],
                 "seconds_left": math.ceil(total - t)}
    else:
        phase = {"state": "pedestrian", "green_approach": None,
                 "approach": None,
                 "seconds_left": math.ceil(total - t)}
    return phase, _public_override(ov, now)


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
    only the displayed counts/queue drain. In live-model mode
    (v0.28) there is no discharge: the toy cars stand still on
    the model, so the counts shown are the measured counts.
    """
    details = compute_green_times(STATE["traffic"], CONFIG, mode=STATE["mode"])
    gap_out_saved = apply_gap_out(details, CONFIG, mode=STATE["mode"])
    phases = build_cycle(details, CONFIG, mode=STATE["mode"])
    total_cycle = sum(p["duration"] for p in phases)

    now = time.time()
    ov_phase, ov_public = _resolve_override(now)
    if ov_public is not None:
        # While an override runs, the normal cycle clock is paused:
        # the drain estimate keeps using the moment it started.
        elapsed_total = STATE["override"]["started_at"] - STATE.get("anchor", START_TIME)
    else:
        elapsed_total = now - STATE.get("anchor", START_TIME)
    # v0.22: a re-anchored cycle starts with a 3-second yellow for
    # the incoming busiest approach (STATE["pre_green"]). Until that
    # yellow ends the cycle clock has not started, so hold the
    # elapsed time at zero instead of letting a future anchor wrap
    # the modulo into a wrong phase.
    pending = STATE.get("pre_green")
    pending_active = bool(pending) and now < pending.get("until", 0)
    if pending and not pending_active:
        STATE["pre_green"] = None
    if pending_active:
        elapsed_total = 0
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
    # v0.28 - NO DRAIN on the physical model. The toy cars STAND
    # STILL: they never drive away during a green, so modelling
    # them as leaving was wrong for the model. The drain emptied
    # every side before the cycle ended (with Tayyab's counts the
    # dashboard showed "stopped" and the LEDs went all-red for
    # the last ~13s of every 60s cycle), and once drained the
    # counts stayed 0. In live-model mode the remaining count IS
    # the measured count, so the cycle only stops when the model
    # is truly empty. Video uploads keep the drain - real
    # traffic really does leave.
    live_model = STATE.get("source") == "live-model"
    all_clear = True
    for approach in APPROACHES:
        d = details[approach]
        initial_total = sum(d["counts"].values())
        start, duration = green_windows.get(approach, (0, 0))
        if live_model:
            remaining = initial_total
        else:
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
            # Never show the get-ready yellow while the leaving
            # approach is still in its own yellow phase: for about
            # a second both approaches would show yellow together.
            # The ready yellow belongs to the all-red gap.
            if seconds_to_next <= ready_seconds and current["state"] != "yellow":
                ready_approach = next_green

    if pending_active:
        # v0.22 PRE-GREEN YELLOW: the incoming approach shows yellow
        # until its green starts. The bridge and the dashboard both
        # render ready_approach as yellow, so the physical LEDs do
        # too - every green, including one after a re-anchor, is
        # now preceded by a yellow.
        seconds_left = max(1, int(pending["until"] - now + 0.999))
        current = {"state": "yellow", "green_approach": None,
                   "approach": pending["approach"],
                   "seconds_left": seconds_left}
        ready_approach = pending["approach"]
        next_green = pending["approach"]
        seconds_to_next = seconds_left

    if ov_public is not None:
        # An override replaces whatever the normal cycle is doing.
        # It applies even when all roads are clear: an emergency
        # vehicle can arrive at an empty intersection too.
        current = ov_phase
        next_green = None
        seconds_to_next = None
        ready_approach = None

    # v0.25 - Manual light test: the clicked light replaces the
    # displayed signals (dashboard + bridge read manual_light and
    # render only that light). The cycle clock keeps running
    # underneath, so cancelling returns to the live cycle.
    manual = STATE.get("manual_light")
    if manual is not None:
        current = {"state": "manual", "green_approach": None,
                   "approach": (manual["approach"]
                                if manual["approach"] != "All" else None),
                   "seconds_left": 0}
        next_green = None
        seconds_to_next = None
        ready_approach = None

    return {
        "source": STATE["source"],
        "mode": STATE["mode"],
        "approaches": details,
        "current_phase": current,
        "cycle_seconds": total_cycle,
        "safety_check_passed": check_no_conflicting_greens(phases),
        "all_roads_clear": all_clear,
        "stopped": all_clear and ov_public is None and manual is None,
        "next_green_approach": next_green,
        "seconds_to_next_green": seconds_to_next,
        "ready_approach": ready_approach,
        "override": ov_public,
        "manual_light": manual,
        "gap_out_saved_seconds": gap_out_saved,
        "note": MODEL_NOTE if live_model else DRAIN_NOTE,
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
    STATE["override"] = None
    # v0.25.2: new numbers do NOT end a manual test - the cycle
    # returns only when Tayyab presses "Back to automatic" (or
    # starts an override). The fresh counts wait underneath.
    STATE["live"] = {}
    STATE["tracking"] = {}
    details = compute_green_times(payload.traffic, CONFIG, mode=payload.mode)
    apply_gap_out(details, CONFIG, mode=payload.mode)
    return {"mode": payload.mode, "approaches": details,
            "cycle": build_cycle(details, CONFIG, mode=payload.mode)}


class ModeInput(BaseModel):
    mode: str


@app.post("/api/mode")
def set_mode(payload: ModeInput):
    """Change only the mode (adaptive/fixed). Traffic numbers stay as they are."""
    if payload.mode not in ("adaptive", "fixed"):
        raise HTTPException(status_code=400, detail="mode must be 'adaptive' or 'fixed'")
    STATE["mode"] = payload.mode
    # v0.25.2: a mode change does NOT end a manual test - the
    # cycle returns only when Tayyab presses "Back to automatic"
    # (or starts an override). The mode applies underneath and
    # takes effect when the manual test is cancelled.
    return {"mode": STATE["mode"], "source": STATE["source"]}


class EmergencyInput(BaseModel):
    approach: str


@app.post("/api/emergency")
def trigger_emergency(payload: EmergencyInput):
    """Give one approach an immediate priority green.

    Prototype trigger: the dashboard button stands in for an
    emergency vehicle (automatic detection is future work - the
    pretrained model has no ambulance class).
    """
    if payload.approach not in APPROACHES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown approach '{payload.approach}'.")
    return {"override": _start_override("emergency", payload.approach)}


@app.post("/api/pedestrian")
def trigger_pedestrian():
    """Start a pedestrian crossing: every approach red for a few
    seconds so people can cross safely."""
    return {"override": _start_override("pedestrian", None)}


@app.post("/api/override/cancel")
def cancel_override():
    """End a running override early; the normal cycle resumes
    where it paused."""
    ov = STATE.get("override")
    if ov:
        STATE["anchor"] = (STATE.get("anchor", START_TIME)
                           + (time.time() - ov["started_at"]))
        STATE["override"] = None
    return {"override": None}


class ManualLightInput(BaseModel):
    approach: str  # "North" | "East" | "South" | "West" | "All"
    color: str     # "red" | "yellow" | "green"


@app.post("/api/manual-light")
def set_manual_light(payload: ManualLightInput):
    """v0.25 - Manual light test (Tayyab's request): turn on ONE
    light from the dashboard and hold it until cancelled - the
    clicked light turns on, every other light turns red (v0.25.1:
    like a real signal, so switching from North green to East
    green turns North red). The normal cycle keeps running
    underneath; cancelling returns to it. approach "All" lights
    the same colour on all four approaches at once (handy for
    checking every yellow, say). A manual test ends any running
    override, and starting an override ends a manual test -
    last action wins. Nothing else ends it: no timeout, and
    v0.25.2 - mode changes and new measurements leave it alone.
    Only "Back to automatic" (or an override) brings the cycle
    back.
    """
    if payload.approach != "All" and payload.approach not in APPROACHES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown approach '{payload.approach}'.")
    if payload.color not in ("red", "yellow", "green"):
        raise HTTPException(
            status_code=400,
            detail="color must be 'red', 'yellow' or 'green'.")
    STATE["override"] = None
    STATE["manual_light"] = {"approach": payload.approach,
                             "color": payload.color}
    return {"manual_light": STATE["manual_light"]}


@app.post("/api/manual-light/cancel")
def cancel_manual_light():
    """End a manual light test; the normal cycle resumes."""
    STATE["manual_light"] = None
    return {"manual_light": None}


# ------------------------------------------------------------------
# Live camera mode (v0.10)
#
# The dashboard sends one camera frame every few seconds for ONE
# approach. A single frame is noisy (a vehicle half out of view, a
# momentary miss), so the counts of the last few frames are
# averaged - the same idea as the video analysis - and the
# averaged counts are applied to the signals ONLY when they
# actually change. That keeps the live cycle stable instead of
# restarting it on every frame.
# ------------------------------------------------------------------

LIVE_HISTORY_FRAMES = 5

# v0.20 - the physical model reacts FAST. The toy cars stand still,
# but Tayyab moves them between sides by hand during the demo and
# wants the new busiest side decided within ~3 seconds. The model
# mode therefore settles over only the last 3 frames (a count must
# still appear in at least two of them, so one glitch frame cannot
# invent a car), and the dashboard sends a frame every second.
# At 1 frame/second the settled count follows a move in about 2-3
# seconds, and an emptied road is forgotten just as fast. The
# one-road live camera keeps the 5-frame average - real traffic
# there actually moves, so stability matters more than speed.
LIVE_MODEL_HISTORY_FRAMES = 3


def _apply_live_counts(approach, counts_now):
    """Feed one live frame's counts for an approach.

    Returns (averaged_counts, applied). applied is True when the
    averaged counts changed and were pushed into the signal state
    (a fresh measurement - the cycle re-plans from now). While an
    emergency/pedestrian override is running, frames are still
    counted but never applied, so the camera cannot fight the
    override.
    """
    live = STATE.setdefault("live", {})
    entry = live.setdefault(approach, {"history": [], "last_applied": None})
    entry["history"] = (entry["history"] + [counts_now])[-LIVE_HISTORY_FRAMES:]
    rounded, _averages = average_counts(entry["history"])
    if STATE.get("override"):
        return rounded, False
    if entry["last_applied"] == rounded:
        return rounded, False
    entry["last_applied"] = rounded
    traffic = dict(STATE["traffic"])
    traffic[approach] = {"counts": rounded, "queue": total_vehicles(rounded)}
    STATE["traffic"] = traffic
    STATE["source"] = "live-camera"
    STATE["anchor"] = time.time()
    return rounded, True


@app.post("/api/live-frame")
async def live_frame(approach: str = Form(...), file: UploadFile = File(...)):
    """One live camera frame for one approach: count its vehicles
    and auto-apply the averaged counts when they change."""
    if approach not in APPROACHES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown approach '{approach}'.")
    data = await file.read()
    try:
        result = analyze_image_bytes(data)
    except (ImportError, DetectionUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if result.get("proof_image"):
        STATE.setdefault("proofs", {})[approach] = result["proof_image"]
    averaged, applied = _apply_live_counts(approach, result["counts"])
    return {
        "approach": approach,
        "counts_now": result["counts"],
        "counts_avg": averaged,
        "applied": applied,
        "proof_image": result["proof_image"],
    }


@app.post("/api/live-stop")
def live_stop(approach: str):
    """Forget one approach's live history (its camera stopped), so
    a later restart applies its first counts immediately."""
    live = STATE.setdefault("live", {})
    live.pop(approach, None)
    return {"live": list(live.keys())}


# ------------------------------------------------------------------
# Live model mode (v0.14)
#
# The expo model: toy cars stand still in the lanes of a table-top
# intersection while ONE camera watches the whole model. Frames are
# split by the same four drawn regions as the single-video ROI
# analysis, so all four approaches are counted from one frame and
# applied TOGETHER when the averaged counts change. The vehicles
# never need to move - the system counts vehicles present, which
# is exactly what a standing model shows.
# ------------------------------------------------------------------


def _settled_model_counts(frame_counts):
    """v0.19/v0.20 - the settled count for the physical model.

    On the model the toy cars STAND STILL, so a car the camera
    recognises in most frames is really there even when one soft
    frame misses it. A plain average washes such a car out (seen
    in 2 of 5 frames -> 0.4 -> rounds to 0), which collapsed
    Tayyab's live counts to almost nothing on 2026-11-11. The
    settled count takes, per vehicle class, the SECOND-highest
    count among the last few frames: a count must appear in at
    least two frames to be believed (one glitch frame or a doubled
    box cannot invent a car), but a real car no longer has to be
    seen in EVERY frame. v0.20 shrinks the window to the last 3
    frames (see LIVE_MODEL_HISTORY_FRAMES) so a car moved to
    another side is believed in ~2-3 seconds. The one-road live
    camera keeps the plain average - real traffic there actually
    moves.
    """
    if not frame_counts:
        return {name: 0 for name in VEHICLE_CLASSES}
    settled = {}
    for name in VEHICLE_CLASSES:
        seen = sorted((counts.get(name, 0) for counts in frame_counts),
                      reverse=True)
        settled[name] = seen[1] if len(seen) > 1 else seen[0]
    return settled


def _busiest_of(averaged):
    """Which approach is busiest for a set of settled counts.

    Runs the same priority + busiest-first ordering as the live
    cycle, so the side a fresh cycle serves first is exactly the
    side _anchor_live_model warns with the pre-green yellow.
    Returns None when no approach has any vehicles.
    """
    traffic = {
        approach: {"counts": averaged[approach],
                   "queue": total_vehicles(averaged[approach])}
        for approach in APPROACHES
    }
    details = compute_green_times(traffic, CONFIG, mode=STATE["mode"])
    apply_gap_out(details, CONFIG, mode=STATE["mode"])
    phases = build_cycle(details, CONFIG, mode=STATE["mode"])
    return next((p["green"] for p in phases
                 if p["state"] == "green"), None)


def _anchor_live_model(model_state, averaged):
    """Start a fresh live-model cycle from settled counts.

    Sets the traffic, the cycle anchor and the pre-green yellow
    (3s for the incoming busiest side, v0.22). The cycle order
    is busiest-first (build_cycle sorts by priority), so the
    busiest side goes first and every side gets its turn.
    All-zero counts skip the yellow - there is no green to warn
    about. Always returns (averaged, True).
    """
    model_state["last_applied"] = averaged
    STATE["traffic"] = {
        approach: {"counts": averaged[approach],
                   "queue": total_vehicles(averaged[approach])}
        for approach in APPROACHES
    }
    STATE["source"] = "live-model"
    new_total = sum(total_vehicles(averaged[a]) for a in APPROACHES)
    new_busiest = _busiest_of(averaged) if new_total > 0 else None
    if new_busiest and new_total > 0:
        STATE["anchor"] = time.time() + 3
        STATE["pre_green"] = {"approach": new_busiest,
                              "until": STATE["anchor"]}
    else:
        STATE["anchor"] = time.time()
        STATE["pre_green"] = None
    return averaged, True


def _apply_live_model_counts(per_approach_now):
    """Feed one model frame's counts for all approaches at once.

    Returns (settled_per_approach, applied). The counts the
    signals use are the SETTLED counts of the last few frames
    (see _settled_model_counts), applied only when they change,
    never during an override. Approaches the frame does not
    cover (no drawn region) count as empty.

    v0.31 - COMPLETE CYCLE (Tayyab 2026-11-13): once a cycle
    starts it runs to the end and visits ALL FOUR sides,
    busiest-first, before anything restarts it. v0.24 only
    restarted the cycle on a stable busiest change - but two
    sides close in traffic (3 vs 2 cars) flip the busiest back
    and forth as detections flicker, and each flip restarted
    the cycle at the busiest side, so sides 3 and 4 never got
    green (reproduced 2026-11-13: a flip at 44s of a 60s cycle
    restarted at East and West lost its turn). So a running
    cycle is NEVER restarted mid-cycle now: changed counts are
    only recorded, and the NEXT cycle - started by the play-out
    restart below - is built from the latest settled counts, so
    a side the checker loaded with cars goes first in the next
    cycle. The one exception is an empty model: when the
    settled counts are all zero the cycle stops at once (there
    is nothing left to serve), as before.

    v0.28 - the play-out restart is checked on EVERY frame,
    not only when the counts are unchanged. A marginal car that
    flickers in and out makes the settled count alternate every
    frame (6,5,6,5...), so the counts were never unchanged and
    the restart never fired: after the first cycle the model
    sat stopped (all red) forever even though the camera was
    still running and cars were still on the model. At play-out
    no green is running, so restarting with the latest settled
    counts is always right.
    """
    model_state = STATE.setdefault(
        "live_model", {"history": [], "last_applied": None})
    model_state["history"] = (
        model_state["history"] + [per_approach_now])[-LIVE_MODEL_HISTORY_FRAMES:]
    averaged = {}
    for approach in APPROACHES:
        averaged[approach] = _settled_model_counts(
            [frame.get(approach, {}) for frame in model_state["history"]])
    if STATE.get("override"):
        return averaged, False
    # v0.16 + v0.28 (Tayyab's model-demo request): on the model the
    # toy cars never leave, so when the running cycle has fully
    # played out, start it again with the LATEST settled counts -
    # the lights keep rotating until the camera mode is switched
    # off, and green times never go stale. v0.28 moves this check
    # here, on EVERY frame: it used to live inside the "counts
    # unchanged" branch below, and a flickering settled count
    # (6,5,6,5...) meant the counts were never unchanged, so the
    # restart never fired and the model sat stopped after the
    # first cycle. Uploaded videos keep the old stop-when-clear
    # behaviour; this restart only happens while the live model
    # camera is the active source. last_applied is set to the
    # latest counts too, so the frame after a restart does not
    # look like a new change.
    if (model_state["last_applied"] is not None
            and STATE.get("source") == "live-model"):
        details = compute_green_times(STATE["traffic"], CONFIG,
                                      mode=STATE["mode"])
        apply_gap_out(details, CONFIG, mode=STATE["mode"])
        total_cycle = sum(
            p["duration"] for p in build_cycle(
                details, CONFIG, mode=STATE["mode"]))
        if (total_cycle
                and time.time() - STATE.get("anchor", START_TIME)
                >= total_cycle):
            STATE["traffic"] = {
                approach: {"counts": averaged[approach],
                           "queue": total_vehicles(averaged[approach])}
                for approach in APPROACHES
            }
            model_state["last_applied"] = averaged
            STATE["anchor"] = time.time()
            STATE["pre_green"] = None
            return averaged, True
    if model_state["last_applied"] == averaged:
        # The measurement agrees with what is applied.
        return averaged, False
    # The very first measurement applies at once - there is no
    # running cycle to protect yet.
    if model_state["last_applied"] is None:
        return _anchor_live_model(model_state, averaged)
    # v0.31: a running cycle is never restarted mid-cycle, even
    # when the busiest side changes - it must visit all four
    # sides first (see the docstring). Changed counts are only
    # recorded here; the play-out restart above builds the next
    # cycle from them, so a newly busy side goes first next time.
    new_total = sum(total_vehicles(averaged[a]) for a in APPROACHES)
    if new_total == 0:
        # Exception: every road empty - stop at once instead of
        # cycling to the end with no cars left to serve.
        return _anchor_live_model(model_state, averaged)
    current_total = sum(
        total_vehicles(STATE["traffic"][a]["counts"])
        for a in APPROACHES)
    if current_total == 0:
        # The system is stopped (the model was empty) and cars
        # are back - start a fresh cycle at once instead of
        # waiting for a play-out restart that is not running.
        return _anchor_live_model(model_state, averaged)
    model_state["last_applied"] = averaged
    return averaged, False


@app.post("/api/live-model-frame")
async def live_model_frame(regions: str = Form(...),
                           file: UploadFile = File(...)):
    """One camera frame of the whole (model) intersection: count
    every approach from its drawn region and auto-apply."""
    try:
        region_map = json.loads(regions)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400,
                            detail="Regions must be valid JSON.")
    data = await file.read()
    try:
        result = analyze_image_rois_bytes(data, region_map)
    except (ImportError, DetectionUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if result.get("proof_image"):
        STATE.setdefault("proofs", {})["Intersection"] = result["proof_image"]
    per_approach = {approach: res["counts"]
                    for approach, res in result["approaches"].items()}
    averaged, applied = _apply_live_model_counts(per_approach)
    return {
        "counts_now": per_approach,
        "counts_avg": averaged,
        "applied": applied,
        "proof_image": result["proof_image"],
        "proof_frame_vehicles": result["proof_frame_vehicles"],
        "proof_frame_counted": result["proof_frame_counted"],
    }


@app.post("/api/live-model-stop")
def live_model_stop():
    """Forget the model camera's history, so a restart applies its
    first counts immediately."""
    STATE["live_model"] = {"history": [], "last_applied": None}
    return {"live_model": "stopped"}


@app.post("/api/auto-regions")
async def auto_regions(file: UploadFile = File(...)):
    """One camera frame of the whole model in, fitted road regions
    out (v0.29): the dashboard's "Auto-fit regions to roads"
    button sends the current frame, and the detector finds the
    black roads in it and fits one region inside each road - so
    no region spills outside its road and no two regions land on
    the same road. Roads it cannot trace reliably get a box
    placed from the crossing's own geometry instead (v0.30,
    "estimated"); only a road with no road under the box at all
    comes back in "missing" (the dashboard keeps the user's box
    for those). The proof image shows the new regions drawn on
    the frame."""
    data = await file.read()
    try:
        result = auto_regions_from_bytes(data)
    except (ImportError, DetectionUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "regions": result["regions"],
        "found": result["found"],
        "missing": result["missing"],
        "estimated": result["estimated"],
        "proof_image": result["proof_image"],
    }


# ------------------------------------------------------------------
# Printable report (v0.11)
#
# One page that collects what a supervisor or expo judge asks for:
# the current measurement with the recommended signal times, the
# detection proof images, and the latest saved fixed-vs-adaptive
# comparison. The browser's Print -> "Save as PDF" turns it into
# a PDF; no extra library is needed.
# ------------------------------------------------------------------

_SOURCE_LABELS = {
    "demo": "Demo data (no video analyzed yet)",
    "waiting": "Waiting for a measurement — start the live camera or upload a video",
    "video": "Uploaded approach videos",
    "video-roi": "Single intersection video (drawn regions)",
    "live-camera": "Live camera",
    "manual": "Manual counts sent to the API",
}


@app.get("/report", response_class=HTMLResponse)
def report():
    details = compute_green_times(STATE["traffic"], CONFIG, mode=STATE["mode"])
    apply_gap_out(details, CONFIG, mode=STATE["mode"])

    rows = []
    for approach in APPROACHES:
        d = details[approach]
        counts_text = ", ".join(
            f"{name}: {value}" for name, value in d["counts"].items()
            if value) or "none"
        green_text = f"{d['green_time']}s"
        if d.get("gap_out_seconds_saved", 0) > 0:
            green_text += f" (planned {d['green_time_planned']}s, cut short: road clears early)"
        rows.append(
            f"<tr><td>{approach}</td><td>{counts_text}</td>"
            f"<td>{d['queue']}</td><td>{d['density']}</td>"
            f"<td>{d['level']}</td><td>{green_text}</td></tr>")

    proofs = STATE.get("proofs") or {}
    if proofs:
        proof_html = "".join(
            f'<div class="proof"><h3>{name}</h3>'
            f'<img src="data:image/jpeg;base64,{img}"></div>'
            for name, img in proofs.items())
    else:
        proof_html = ("<p>No detection proof image yet — analyze a "
                      "video or start the live camera first.</p>")

    tracking_data = STATE.get("tracking") or {}
    if tracking_data:
        trows = "".join(
            f"<tr><td>{a}</td><td>{t['unique_total']}</td>"
            f"<td>{', '.join(f'{k}: {v}' for k, v in t['unique_counts'].items() if v) or 'none'}</td>"
            f"<td>{'First part of the video only' if t.get('partial') else 'Whole video'}</td></tr>"
            for a, t in tracking_data.items())
        tracking_html = f"""
<h2>Total vehicles that came and passed (tracking)</h2>
<table>
  <tr><th>Approach</th><th>Unique vehicles</th><th>Breakdown</th><th>Coverage</th></tr>
  {trows}
</table>
<p class="muted">Tracking follows every vehicle with an ID and counts each
vehicle once - the flow through the road during the video. This is different
from the per-moment counts above, which measure how crowded the road is.</p>"""
    else:
        tracking_html = ""

    try:
        experiments = storage.list_experiments()
    except Exception:
        experiments = []
    if experiments:
        latest = experiments[0]
        result = latest["result"]
        f, a = result["fixed"], result["adaptive"]
        comparison_html = f"""
        <p>Scenario: <strong>{latest['scenario']}</strong> · run at {latest['created_at']}</p>
        <table>
          <tr><th>Metric</th><th>Fixed-time</th><th>Adaptive</th></tr>
          <tr><td>Average waiting time (s / vehicle)</td><td>{f['avg_waiting_seconds_per_vehicle']}</td><td>{a['avg_waiting_seconds_per_vehicle']}</td></tr>
          <tr><td>Average queue length</td><td>{f['avg_queue_length']}</td><td>{a['avg_queue_length']}</td></tr>
          <tr><td>Maximum queue length</td><td>{f['max_queue_length']}</td><td>{a['max_queue_length']}</td></tr>
          <tr><td>Throughput (vehicles)</td><td>{f['throughput_vehicles']}</td><td>{a['throughput_vehicles']}</td></tr>
        </table>"""
    else:
        comparison_html = ("<p>No comparison run saved yet — run one "
                           "from the dashboard comparison section.</p>")

    generated = time.strftime("%d %B %Y, %H:%M", time.localtime())
    source_label = _SOURCE_LABELS.get(STATE["source"], STATE["source"])
    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Smart Traffic Light Management System — Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 32px; color: #111; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 17px; margin-top: 28px; }}
  .muted {{ color: #555; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  th, td {{ border: 1px solid #999; padding: 6px 8px; text-align: left; font-size: 14px; }}
  .proof img {{ width: 100%; max-width: 640px; border: 1px solid #999; }}
  @media print {{ body {{ margin: 12px; }} }}
</style></head><body>
<h1>AI-Based Smart Traffic Light Management System</h1>
<p class="muted">Demonstration report · Generated {generated} · Data source: {source_label} · Mode: {STATE['mode']}</p>

<h2>Current measurement and recommended signal times</h2>
<table>
  <tr><th>Approach</th><th>Vehicles counted</th><th>Queue</th><th>Density (PCU)</th><th>Level</th><th>Recommended green</th></tr>
  {''.join(rows)}
</table>

<h2>Detection proof — what the AI saw</h2>
{proof_html}
{tracking_html}

<h2>Latest fixed-time vs adaptive comparison (simulation)</h2>
{comparison_html}

<h2>Notes</h2>
<p class="muted">
Counts are the average number of vehicles visible per sampled frame — a
density estimate, the value signal timing needs; the system does not claim
to count every vehicle that passed. The dashboard's falling counts during a
green are a discharge estimate (about one vehicle every 2 seconds), not a
re-measurement. Counting accuracy was measured against manual counts on an
aerial test clip at about 78% overall and about 94% on the busiest approach,
which was identified correctly in every checked frame; accuracy depends on
camera angle, and the system assumes an angled traffic camera like real
installations. This is a demonstration prototype for a single four-way
intersection — not a certified traffic control system.
</p>
</body></html>"""
    return HTMLResponse(body)


@app.post("/api/analyze-videos")
def analyze_videos(
    north: Optional[UploadFile] = File(None),
    east: Optional[UploadFile] = File(None),
    south: Optional[UploadFile] = File(None),
    west: Optional[UploadFile] = File(None),
    tracking: bool = Form(False),
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
    STATE["tracking"] = {}
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
            if result.get("proof_image"):
                STATE.setdefault("proofs", {})[approach] = result["proof_image"]
            if tracking:
                # Optional extra (v0.12): also count the UNIQUE
                # vehicles that came and passed, via tracking.
                # If tracking cannot run, the normal counts still
                # go through and the reason is reported instead.
                try:
                    result["tracking"] = track_video_file(path)
                    STATE["tracking"][approach] = result["tracking"]
                except (ImportError, DetectionUnavailableError,
                        ValueError) as exc:
                    result["tracking_error"] = str(exc)
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
    STATE["override"] = None
    STATE["live"] = {}
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
        "cycle": build_cycle(details, CONFIG, mode=STATE["mode"]),
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
    STATE["override"] = None
    STATE["live"] = {}
    STATE["tracking"] = {}
    STATE["proofs"] = ({"Intersection": result["proof_image"]}
                       if result.get("proof_image") else {})
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
        "cycle": build_cycle(details, CONFIG, mode=STATE["mode"]),
        "recommendation": recommendation,
        "analysis": analysis,
        "missing_approaches": missing,
        "proof_image": result.get("proof_image"),
        "proof_frame_vehicles": result.get("proof_frame_vehicles"),
        "proof_frame_counted": result.get("proof_frame_counted"),
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
