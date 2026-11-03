"""
Tests for the adaptive algorithm and the simulation.
Run with the standard library only - no pytest needed:

    PowerShell, from the smart-traffic-system folder:
        python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from simulator import run_comparison
from traffic_algorithm import (
    DEFAULT_CONFIG,
    build_cycle,
    check_no_conflicting_greens,
    classify_level,
    compute_density,
    compute_green_times,
)

CONFIG = DEFAULT_CONFIG


def traffic(north_cars=0, east_cars=0, south_cars=0, west_cars=0):
    def one(cars, queue=0):
        return {"counts": {"car": cars}, "queue": queue}
    return {
        "North": one(north_cars, north_cars // 2),
        "East": one(east_cars, east_cars // 2),
        "South": one(south_cars, south_cars // 2),
        "West": one(west_cars, west_cars // 2),
    }


class TestDensityAndLevel(unittest.TestCase):
    def test_pcu_weights(self):
        counts = {"car": 2, "motorcycle": 2, "bus": 1, "truck": 1}
        # 2*1.0 + 2*0.5 + 1*2.5 + 1*2.0 = 7.5
        self.assertEqual(compute_density(counts, CONFIG), 7.5)

    def test_levels(self):
        self.assertEqual(classify_level(3, CONFIG), "Low")
        self.assertEqual(classify_level(10, CONFIG), "Medium")
        self.assertEqual(classify_level(20, CONFIG), "High")


class TestGreenTimes(unittest.TestCase):
    def test_busier_road_gets_more_green(self):
        details = compute_green_times(traffic(20, 2, 10, 1), CONFIG)
        self.assertGreater(details["North"]["green_time"],
                           details["West"]["green_time"])

    def test_green_always_within_limits(self):
        for counts in [(0, 0, 0, 0), (50, 0, 0, 0), (10, 10, 10, 10)]:
            details = compute_green_times(traffic(*counts), CONFIG)
            for approach in details.values():
                self.assertGreaterEqual(approach["green_time"], 10)
                self.assertLessEqual(approach["green_time"], 60)

    def test_fixed_mode_is_equal(self):
        details = compute_green_times(traffic(20, 2, 10, 1), CONFIG, mode="fixed")
        greens = {d["green_time"] for d in details.values()}
        self.assertEqual(greens, {30.0})

    def test_no_conflicting_greens(self):
        details = compute_green_times(traffic(20, 2, 10, 1), CONFIG)
        phases = build_cycle(details, CONFIG)
        self.assertTrue(check_no_conflicting_greens(phases))
        green_phases = [p for p in phases if p["state"] == "green"]
        self.assertEqual(len(green_phases), 4)  # one per approach per cycle


class TestSimulation(unittest.TestCase):
    def test_comparison_runs_for_all_scenarios(self):
        for name in ("low", "medium", "high"):
            result = run_comparison(name, CONFIG, duration_seconds=120)
            for strategy in ("fixed", "adaptive"):
                metrics = result[strategy]
                self.assertGreater(metrics["vehicles_arrived"], 0)
                self.assertGreaterEqual(metrics["avg_queue_length"], 0)
                self.assertGreaterEqual(metrics["throughput_vehicles"], 0)


if __name__ == "__main__":
    unittest.main()
