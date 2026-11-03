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
