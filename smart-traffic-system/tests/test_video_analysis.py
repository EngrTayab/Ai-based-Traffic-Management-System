"""
Tests for the video-analysis averaging logic.
These test the counting maths only - they do not need YOLO, OpenCV
or a real video file.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from video_analysis import average_counts, total_vehicles


class TestAverageCounts(unittest.TestCase):
    def test_average_and_rounding(self):
        frames = [{"car": 4, "bus": 1}, {"car": 6}, {"car": 5, "motorcycle": 3}]
        rounded, averages = average_counts(frames)
        self.assertEqual(averages["car"], 5.0)
        self.assertEqual(rounded["car"], 5)
        self.assertEqual(rounded["motorcycle"], 1)

    def test_empty_frames_give_zeros(self):
        rounded, averages = average_counts([])
        self.assertTrue(all(v == 0 for v in rounded.values()))
        self.assertTrue(all(v == 0.0 for v in averages.values()))

    def test_missing_classes_are_zero(self):
        rounded, _ = average_counts([{"truck": 2}])
        self.assertEqual(rounded["car"], 0)
        self.assertEqual(rounded["truck"], 2)

    def test_total_vehicles(self):
        self.assertEqual(total_vehicles({"car": 3, "bus": 1}), 4)
        self.assertEqual(total_vehicles({}), 0)


if __name__ == "__main__":
    unittest.main()
