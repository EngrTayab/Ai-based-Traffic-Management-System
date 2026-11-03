"""
Fixed-time vs adaptive comparison, done as a simulation.

Why a simulation? We cannot control real traffic lights, so the SRS
(Section 9.3) allows comparing both strategies on the same simulated
traffic arrivals. The numbers this file produces are SIMULATION
estimates from assumed arrival rates - they are not measurements from
real traffic, and they must be described that way in the report.

Model (kept deliberately simple and explainable):
- Each approach receives vehicles at a fixed arrival rate
  (vehicles per minute) for the whole run.
- During its green, an approach clears 1 vehicle every 2 seconds.
- Yellow and all-red clear nobody.
- Waiting time is accumulated as queue_length * seconds waited.
"""

from traffic_algorithm import APPROACHES, compute_green_times

# Assumed arrival rates (vehicles per minute per approach).
# These are demo scenarios, to be replaced/calibrated with real
# counts from the test videos during the evaluation stage.
SCENARIOS = {
    "low":    {"North": 2, "East": 1, "South": 2, "West": 1},
    "medium": {"North": 6, "East": 4, "South": 5, "West": 3},
    "high":   {"North": 12, "East": 8, "South": 10, "West": 6},
}

SERVICE_SECONDS_PER_VEHICLE = 2.0


def _run_one_strategy(arrival_rates, config, mode, duration_seconds=600):
    queues = {a: 0.0 for a in APPROACHES}
    arrived = 0.0
    departed = 0.0
    total_wait_vehicle_seconds = 0.0
    queue_samples = []
    max_queue = 0.0

    def current_traffic():
        # The adaptive controller "sees" the current queue as its input.
        # Counts are approximated by the queue itself (1 vehicle = 1 car PCU).
        return {
            a: {"counts": {"car": int(round(queues[a]))}, "queue": int(round(queues[a]))}
            for a in APPROACHES
        }

    second = 0
    while second < duration_seconds:
        details = compute_green_times(current_traffic(), config, mode=mode)
        for approach in APPROACHES:
            green = int(round(details[approach]["green_time"]))
            for phase_name, phase_seconds in (
                ("green", green),
                ("yellow", config["yellow"]),
                ("all_red", config["all_red"]),
            ):
                for _ in range(int(phase_seconds)):
                    if second >= duration_seconds:
                        break
                    # Arrivals for every approach, every second
                    for a in APPROACHES:
                        add = arrival_rates[a] / 60.0
                        queues[a] += add
                        arrived += add
                    # Departures only from the approach that is green
                    if phase_name == "green" and queues[approach] > 0:
                        cleared = min(queues[approach],
                                      1.0 / SERVICE_SECONDS_PER_VEHICLE)
                        queues[approach] -= cleared
                        departed += cleared
                    total_wait_vehicle_seconds += sum(queues.values())
                    queue_samples.append(sum(queues.values()) / len(APPROACHES))
                    max_queue = max(max_queue, max(queues.values()))
                    second += 1

    avg_wait = (total_wait_vehicle_seconds / arrived) if arrived else 0.0
    avg_queue = (sum(queue_samples) / len(queue_samples)) if queue_samples else 0.0
    return {
        "strategy": mode,
        "duration_seconds": duration_seconds,
        "vehicles_arrived": round(arrived),
        "throughput_vehicles": int(departed),
        "avg_waiting_seconds_per_vehicle": round(avg_wait, 2),
        "avg_queue_length": round(avg_queue, 2),
        "max_queue_length": round(max_queue, 2),
    }


def run_comparison(scenario_name, config, duration_seconds=600):
    """Run BOTH strategies on the SAME scenario and return both results."""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_name}'. "
                         f"Choose one of: {', '.join(SCENARIOS)}")
    rates = SCENARIOS[scenario_name]
    return {
        "scenario": scenario_name,
        "arrival_rates_per_minute": rates,
        "note": "Simulation estimate from assumed arrival rates, "
                "not a measurement from real traffic.",
        "fixed": _run_one_strategy(rates, config, "fixed", duration_seconds),
        "adaptive": _run_one_strategy(rates, config, "adaptive", duration_seconds),
    }
