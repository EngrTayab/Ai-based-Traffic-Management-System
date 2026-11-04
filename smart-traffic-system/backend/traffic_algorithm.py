"""
Adaptive traffic signal algorithm.

This module is pure Python. It does NOT need YOLO, OpenCV or a video.
That is deliberate: the signal logic can be tested with simple numbers
(dummy counts) before any camera or video work exists.

Rules come from the project SRS (Section 7):
- Density uses Passenger Car Units (PCU): motorcycle 0.5, car 1.0,
  truck 2.0, bus 2.5
- Priority P = 0.6 * density + 0.4 * queue_length
- Traffic level: Low < 5, Medium 5-15, High > 15 (on density)
- Green time is always clamped between green_min and green_max
- Every green is followed by yellow, then all-red
- Only one approach is ever green at a time (safety rule)
- Fixed-time baseline: every approach gets the same green time
"""

APPROACHES = ["North", "East", "South", "West"]
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

DEFAULT_CONFIG = {
    "pcu_weights": {"motorcycle": 0.5, "car": 1.0, "truck": 2.0, "bus": 2.5},
    "alpha_density": 0.6,
    "beta_queue": 0.4,
    "level_low_max": 5,
    "level_high_min": 15,
    "green_min": 10,
    "green_max": 60,
    "green_total_per_cycle": 120,
    "yellow": 3,
    "all_red": 2,
    "fixed_green": 30,
    "max_red": 180,
    # Queue discharge estimate for the live dashboard display: while
    # an approach is green, roughly one vehicle leaves every
    # "discharge_seconds_per_vehicle" seconds (typical saturation
    # headway is about 2 seconds per vehicle).
    "discharge_seconds_per_vehicle": 2.0,
    # "Get ready" yellow: this many seconds before an approach's
    # green starts, that approach already shows yellow on the
    # dashboard, warning its drivers that their turn is next.
    "ready_yellow_seconds": 3,
    # Gap-out: in adaptive mode, a green that is longer than the
    # time its measured queue needs to clear is cut short, so the
    # next approach gets its turn earlier. See apply_gap_out().
    "gap_out_enabled": True,
}


def load_config(path="config.json"):
    """Load config.json if it exists, otherwise use the SRS defaults."""
    import json
    import os

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config = DEFAULT_CONFIG.copy()
            config.update(json.load(f))
            return config
    return DEFAULT_CONFIG.copy()


def compute_density(counts, config):
    """Weighted traffic density (PCU) for one approach.

    counts example: {"car": 5, "motorcycle": 2, "bus": 1, "truck": 0}
    """
    weights = config["pcu_weights"]
    return sum(counts.get(name, 0) * weights.get(name, 0) for name in VEHICLE_CLASSES)


def classify_level(density, config):
    """Turn a density score into Low / Medium / High."""
    if density < config["level_low_max"]:
        return "Low"
    if density <= config["level_high_min"]:
        return "Medium"
    return "High"


def compute_priority(density, queue_length, config):
    """Priority score P = alpha * density + beta * queue."""
    return config["alpha_density"] * density + config["beta_queue"] * queue_length


def clamp(value, low, high):
    return max(low, min(high, value))


def compute_green_times(traffic, config, mode="adaptive"):
    """Calculate green time (seconds) for every approach.

    traffic example:
    {
        "North": {"counts": {"car": 10, ...}, "queue": 8},
        "East":  {"counts": {"car": 2, ...},  "queue": 1},
        ...
    }
    Returns: {"North": 34.2, "East": 12.0, ...} plus a detail dict.
    """
    details = {}
    for approach in APPROACHES:
        data = traffic.get(approach, {})
        counts = data.get("counts", {})
        queue = data.get("queue", 0)
        density = compute_density(counts, config)
        details[approach] = {
            "counts": {name: counts.get(name, 0) for name in VEHICLE_CLASSES},
            "queue": queue,
            "density": round(density, 2),
            "level": classify_level(density, config),
            "priority": round(compute_priority(density, queue, config), 2),
        }

    green_min = config["green_min"]
    green_max = config["green_max"]

    if mode == "fixed":
        greens = {a: float(config["fixed_green"]) for a in APPROACHES}
    else:
        total_priority = sum(d["priority"] for d in details.values())
        greens = {}
        if total_priority == 0:
            # No traffic measured anywhere: share the cycle evenly.
            even = config["green_total_per_cycle"] / len(APPROACHES)
            for approach in APPROACHES:
                greens[approach] = float(clamp(even, green_min, green_max))
        else:
            budget = config["green_total_per_cycle"] - len(APPROACHES) * green_min
            for approach in APPROACHES:
                share = details[approach]["priority"] / total_priority
                greens[approach] = float(
                    clamp(green_min + share * budget, green_min, green_max)
                )

    for approach in APPROACHES:
        details[approach]["green_time"] = round(greens[approach], 1)
    return details


def apply_gap_out(details, config, mode="adaptive"):
    """Cut greens short once the measured queue has cleared (gap-out).

    Real signal controllers do this: a green is planned from the
    measured traffic, but if the queue drains before the planned
    green is over, the remaining seconds are wasted - nobody is
    left to use them. Gap-out ends the green at the moment the
    queue is estimated to be empty and hands the time to the next
    approach.

    The clearing time uses the same discharge estimate as the live
    dashboard drain: vehicles leave at about one per
    `discharge_seconds_per_vehicle` seconds, so an approach with N
    measured vehicles needs about N * discharge seconds of green.

    Safety / honesty rules:
    - A green is never cut below green_min (the SRS minimum), even
      for an empty approach.
    - A green is never EXTENDED by gap-out, only shortened.
    - Fixed mode is the untouched baseline: no gap-out there, so
      the fixed-vs-adaptive comparison stays honest.
    - This is an estimate from the recorded measurement, like the
      drain display - not a fresh detection.

    Mutates each approach in `details`: "green_time" becomes the
    effective green, "green_time_planned" keeps the original, and
    "gap_out_seconds_saved" records the cut (0 when no cut).
    Returns the total seconds saved in one cycle.
    """
    import math

    total_saved = 0.0
    enabled = mode == "adaptive" and config.get("gap_out_enabled", True)
    discharge = float(config.get("discharge_seconds_per_vehicle", 2.0))
    green_min = float(config["green_min"])
    for approach in APPROACHES:
        d = details[approach]
        planned = float(d["green_time"])
        d["green_time_planned"] = planned
        saved = 0.0
        if enabled:
            vehicles = sum(d["counts"].get(name, 0) for name in VEHICLE_CLASSES)
            drain_needed = math.ceil(vehicles * discharge) if vehicles > 0 else 0
            effective = min(planned, max(green_min, float(drain_needed)))
            if effective < planned:
                saved = round(planned - effective, 1)
                d["green_time"] = round(effective, 1)
        d["gap_out_seconds_saved"] = saved
        total_saved += saved
    return round(total_saved, 1)


def build_cycle(details, config):
    """Build one full signal cycle: N -> E -> S -> W.

    Safety invariant: exactly one approach is green in any phase,
    and yellow + all-red always come between two greens.
    """
    phases = []
    for approach in APPROACHES:
        green_time = int(round(details[approach]["green_time"]))
        phases.append({"green": approach, "state": "green", "duration": green_time})
        phases.append({"green": None, "state": "yellow", "approach": approach,
                       "duration": config["yellow"]})
        phases.append({"green": None, "state": "all_red",
                       "duration": config["all_red"]})
    return phases


def check_no_conflicting_greens(phases):
    """Safety check used by the tests: a phase never has 2+ greens."""
    for phase in phases:
        greens = phase.get("greens")
        if greens is not None and len(greens) > 1:
            return False
        if phase.get("state") == "green" and not phase.get("green"):
            return False
    return True


def estimate_remaining_vehicles(initial_vehicles, green_start, green_duration,
                                cycle_seconds, elapsed_seconds,
                                discharge_seconds=2.0):
    """Estimate how many measured vehicles are still waiting.

    This powers the live dashboard effect Tayyab asked for: while an
    approach has the green light, its queue drains - roughly one
    vehicle leaves every `discharge_seconds` seconds of green. When
    the estimate reaches 0 the road is shown as clear.

    IMPORTANT: this is an ESTIMATE of the queue clearing, not a new
    detection. The recorded videos are not synchronised with the
    signal cycle, so the system cannot truly know how many vehicles
    have left - it models the discharge at a fixed rate.

    green_start / green_duration: this approach's green window
    inside one cycle (seconds). elapsed_seconds: time since the
    traffic was measured (the cycle anchor). Green received in
    earlier cycles counts too, so the estimate keeps falling cycle
    after cycle until it reaches 0 and stays there.
    """
    if initial_vehicles <= 0:
        return 0
    if cycle_seconds <= 0 or green_duration <= 0:
        return initial_vehicles
    elapsed = max(0.0, float(elapsed_seconds))
    full_cycles = int(elapsed // cycle_seconds)
    in_cycle = elapsed - full_cycles * cycle_seconds
    green_elapsed = full_cycles * green_duration
    if in_cycle > green_start:
        green_elapsed += min(in_cycle - green_start, green_duration)
    discharged = int(green_elapsed / discharge_seconds)
    return max(0, initial_vehicles - discharged)


def scale_counts(counts, target_total):
    """Scale per-class counts so they add up to target_total.

    Used to show the draining queue: the class mix stays roughly
    proportional to what was measured, while the total falls.
    Largest-remainder rounding keeps the sum exact.
    """
    total = sum(counts.get(name, 0) for name in VEHICLE_CLASSES)
    if total <= 0 or target_total >= total:
        return {name: counts.get(name, 0) for name in VEHICLE_CLASSES}
    if target_total <= 0:
        return {name: 0 for name in VEHICLE_CLASSES}
    exact = {name: counts.get(name, 0) * target_total / total
             for name in VEHICLE_CLASSES}
    scaled = {name: int(exact[name]) for name in VEHICLE_CLASSES}
    leftover = target_total - sum(scaled.values())
    by_remainder = sorted(VEHICLE_CLASSES,
                          key=lambda n: exact[n] - scaled[n], reverse=True)
    for name in by_remainder[:leftover]:
        scaled[name] += 1
    return scaled
