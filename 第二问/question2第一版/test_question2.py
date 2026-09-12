"""第二问闭式推导、几何算法和输出数据的回归测试。"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from q1_localization import Measurement, solve_localization
from q2_theory import (
    MAX_RECEPTION_RADIUS_M,
    approximate_diameter,
    derive_solution,
    normalized_diameter_objective,
    safe_candidate,
    second_detection_point,
)
from q2_validation import bearing_deg, minimum_enclosing_circle


HERE = Path(__file__).resolve().parent


class TheoryTests(unittest.TestCase):
    def test_closed_form_solution(self) -> None:
        solution = derive_solution()
        self.assertAlmostEqual(solution.forward_ratio, 9.0 / 11.0, places=12)
        self.assertAlmostEqual(
            solution.lateral_ratio, 2.0 * math.sqrt(10.0) / 11.0, places=12
        )
        self.assertAlmostEqual(solution.travel_m, 1000.0, places=9)
        self.assertTrue(safe_candidate(solution.forward_m, solution.lateral_abs_m))
        self.assertAlmostEqual(
            solution.nominal_worst_diameter_m,
            7000.0 * math.tan(math.radians(1.0)),
            places=9,
        )

    def test_stationary_equation_and_unique_feasible_root(self) -> None:
        first_root = 9.0 / 11.0
        second_root = 121.0 / 99.0
        polynomial = lambda value: 99.0 * value**2 - 202.0 * value + 99.0
        self.assertAlmostEqual(polynomial(first_root), 0.0, places=12)
        self.assertGreater(second_root, 1.0)
        self.assertLess(
            normalized_diameter_objective(first_root),
            normalized_diameter_objective(first_root - 0.05),
        )
        self.assertLess(
            normalized_diameter_objective(first_root),
            normalized_diameter_objective(first_root + 0.05),
        )

    def test_worst_nominal_radius_is_upper_endpoint(self) -> None:
        solution = derive_solution()
        radial_grid = np.linspace(0.0, MAX_RECEPTION_RADIUS_M, 3001)
        values = np.array(
            [
                approximate_diameter(
                    radius, solution.forward_m, solution.lateral_abs_m
                )
                for radius in radial_grid
            ]
        )
        self.assertEqual(int(np.argmax(values)), len(radial_grid) - 1)

    def test_coordinate_rotation(self) -> None:
        solution = derive_solution()
        point = second_detection_point(10.0, 20.0, 90.0, 1)
        expected = np.array(
            [
                10.0 - solution.lateral_abs_m,
                20.0 + solution.forward_m,
            ]
        )
        np.testing.assert_allclose(point, expected, atol=1.0e-9)


class GeometryTests(unittest.TestCase):
    def test_exact_nominal_half_plane_check(self) -> None:
        solution = derive_solution()
        second = np.array([solution.forward_m, solution.lateral_abs_m])
        source = np.array([1500.0, 0.0])
        result = solve_localization(
            [
                Measurement(0.0, 0.0, 0.0),
                Measurement(*second, bearing_deg(second, source)),
            ],
            1.0,
        )
        self.assertAlmostEqual(result.diameter, 122.4626359104, places=7)
        self.assertTrue(result.circle_covers_polygon)

    def test_minimum_enclosing_circle(self) -> None:
        triangle = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        center, radius = minimum_enclosing_circle(triangle)
        np.testing.assert_allclose(center, [1.0, 1.0], atol=1.0e-10)
        self.assertAlmostEqual(radius, math.sqrt(2.0), places=10)

    def test_saved_simulation_summary_is_consistent(self) -> None:
        path = HERE / "simulation_summary.json"
        if not path.exists():
            self.skipTest("尚未生成仿真汇总")
        with path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        self.assertEqual(summary["observation_counts"]["no_signal"], 0)
        self.assertEqual(summary["observation_counts"]["invalid_intersection"], 0)
        self.assertAlmostEqual(summary["reception_retained_rate"], 1.0)
        self.assertAlmostEqual(summary["true_source_containment_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
