"""第二问闭式推导、几何算法和输出数据的回归测试。"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from q1_localization import (
    LocalizationError,
    Measurement,
    build_bearing_halfplanes,
    diameter_circle_coverage,
    half_plane_intersection,
    make_line,
    rotating_calipers_diameter,
    solve_localization,
)
from q2_theory import (
    MAX_RECEPTION_RADIUS_M,
    approximate_diameter,
    derive_solution,
    normalized_diameter_objective,
    safe_candidate,
    second_detection_point,
)
from q2_validation import bearing_deg, deterministic_checks, minimum_enclosing_circle


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
    def test_two_orthogonal_bearings(self) -> None:
        result = solve_localization(
            [
                Measurement(0.0, 0.0, 0.0),
                Measurement(1000.0, 1000.0, 270.0),
            ],
            error_deg=1.0,
        )
        self.assertEqual(len(result.vertices), 4)
        self.assertAlmostEqual(result.diameter, 49.385425824, places=6)
        self.assertTrue(result.circle_covers_polygon)

    def test_single_bearing_is_unbounded(self) -> None:
        lines = build_bearing_halfplanes([Measurement(0.0, 0.0, 0.0)])
        with self.assertRaises(LocalizationError):
            half_plane_intersection(lines)

    def test_empty_bounded_intersection_is_rejected(self) -> None:
        origin = np.array([0.0, 0.0])
        lines = [
            make_line(np.array([1.0, 0.0]), np.array([0.0, -1.0])),
            make_line(origin, np.array([0.0, 1.0])),
            make_line(origin, np.array([1.0, 0.0])),
            make_line(np.array([0.0, 1.0]), np.array([-1.0, 0.0])),
        ]
        with self.assertRaises(LocalizationError):
            half_plane_intersection(lines)

    def test_square_half_plane_intersection(self) -> None:
        lines = [
            make_line(np.array([0.0, 0.0]), np.array([1.0, 0.0])),
            make_line(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            make_line(np.array([1.0, 1.0]), np.array([-1.0, 0.0])),
            make_line(np.array([0.0, 1.0]), np.array([0.0, -1.0])),
        ]
        square = half_plane_intersection(lines)
        self.assertEqual(len(square), 4)
        first_edge = square[1] - square[0]
        second_edge = square[2] - square[1]
        signed_turn = float(
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        )
        self.assertGreaterEqual(signed_turn, 0.0)

    def test_half_plane_intersection_keeps_point_degeneracy(self) -> None:
        origin = np.array([0.0, 0.0])
        lines = [
            make_line(origin, np.array([0.0, -1.0])),
            make_line(origin, np.array([0.0, 1.0])),
            make_line(origin, np.array([1.0, 0.0])),
            make_line(origin, np.array([-1.0, 0.0])),
        ]
        vertices = half_plane_intersection(lines)
        np.testing.assert_allclose(vertices, [[0.0, 0.0]], atol=1.0e-10)
        self.assertEqual(rotating_calipers_diameter(vertices).diameter, 0.0)

    def test_half_plane_intersection_keeps_segment_degeneracy(self) -> None:
        origin = np.array([0.0, 0.0])
        lines = [
            make_line(origin, np.array([0.0, -1.0])),
            make_line(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            make_line(origin, np.array([1.0, 0.0])),
            make_line(origin, np.array([-1.0, 0.0])),
        ]
        vertices = half_plane_intersection(lines)
        np.testing.assert_allclose(vertices, [[0.0, 0.0], [1.0, 0.0]], atol=1.0e-10)
        self.assertAlmostEqual(rotating_calipers_diameter(vertices).diameter, 1.0)

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

    def test_equilateral_triangle_diameter_circle_does_not_cover(self) -> None:
        height = math.sqrt(3.0) / 2.0
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, height]])
        diameter = rotating_calipers_diameter(triangle)
        covered, _, radius, _ = diameter_circle_coverage(
            triangle, diameter.point_a, diameter.point_b
        )
        self.assertAlmostEqual(diameter.diameter, 1.0, places=12)
        self.assertAlmostEqual(radius, 0.5, places=12)
        self.assertFalse(covered)

    def test_rectangle_diameter_circle_covers(self) -> None:
        rectangle = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])
        diameter = rotating_calipers_diameter(rectangle)
        covered, _, _, _ = diameter_circle_coverage(
            rectangle, diameter.point_a, diameter.point_b
        )
        self.assertAlmostEqual(diameter.diameter, math.sqrt(13.0), places=12)
        self.assertTrue(covered)

    def test_rotating_calipers_matches_brute_force(self) -> None:
        random = np.random.default_rng(20260911)
        for _ in range(100):
            count = int(random.integers(3, 40))
            angles = np.sort(random.uniform(0.0, 2.0 * math.pi, count))
            polygon = np.column_stack((3.0 * np.cos(angles), 2.0 * np.sin(angles)))
            calculated = rotating_calipers_diameter(polygon).diameter
            expected = max(
                np.linalg.norm(polygon[first] - polygon[second])
                for first in range(count)
                for second in range(first)
            )
            self.assertAlmostEqual(calculated, expected, places=10)

    def test_full_precision_boundary_counterexample(self) -> None:
        checks = deterministic_checks()
        edge = checks["full_precision_error_boundary_counterexample"]
        self.assertAlmostEqual(edge["second_true_bearing_deg"], 321.1607715712, places=9)
        self.assertAlmostEqual(edge["second_reading_deg"], 322.1607715712, places=9)
        self.assertAlmostEqual(edge["exact_diameter_m"], 135.1845601220, places=7)
        self.assertGreater(
            edge["exact_diameter_m"], derive_solution().nominal_worst_diameter_m
        )

    def test_near_source_is_triangle(self) -> None:
        near = deterministic_checks()["near_source_triangle"]
        self.assertEqual(near["vertex_count"], 3)
        self.assertAlmostEqual(near["exact_diameter_m"], 40.1042701001, places=7)
        self.assertAlmostEqual(near["first_order_diameter_m"], 60.2217700765, places=7)

    def test_same_scene_candidate_check(self) -> None:
        comparison = deterministic_checks()["same_nominal_scene_candidate_check"]
        self.assertAlmostEqual(
            comparison["current_exact_diameter_m"], 122.4626359104, places=7
        )
        self.assertAlmostEqual(
            comparison["forty_five_exact_diameter_m"], 126.1164084180, places=7
        )
        self.assertLess(
            comparison["current_exact_diameter_m"],
            comparison["forty_five_exact_diameter_m"],
        )

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
