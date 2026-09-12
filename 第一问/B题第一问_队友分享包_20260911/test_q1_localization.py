"""第一问核心几何算法的回归测试。"""

import math
import unittest

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


class QuestionOneGeometryTests(unittest.TestCase):
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

    def test_equilateral_triangle_is_not_covered(self) -> None:
        height = math.sqrt(3.0) / 2.0
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, height]])
        diameter = rotating_calipers_diameter(triangle)
        covered, _, radius, _ = diameter_circle_coverage(
            triangle, diameter.point_a, diameter.point_b
        )
        self.assertAlmostEqual(diameter.diameter, 1.0, places=12)
        self.assertAlmostEqual(radius, 0.5, places=12)
        self.assertFalse(covered)

    def test_rectangle_diameter(self) -> None:
        rectangle = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])
        diameter = rotating_calipers_diameter(rectangle)
        covered, _, _, _ = diameter_circle_coverage(
            rectangle, diameter.point_a, diameter.point_b
        )
        self.assertAlmostEqual(diameter.diameter, math.sqrt(13.0), places=12)
        self.assertTrue(covered)

    def test_single_bearing_is_unbounded(self) -> None:
        lines = build_bearing_halfplanes([Measurement(0.0, 0.0, 0.0)])
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
        area_factor = abs(
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        )
        self.assertAlmostEqual(area_factor, 1.0)

    def test_rotating_calipers_matches_brute_force(self) -> None:
        random = np.random.default_rng(20260911)
        for _ in range(100):
            count = int(random.integers(3, 40))
            angles = np.sort(random.uniform(0.0, 2.0 * math.pi, count))
            polygon = np.column_stack((3.0 * np.cos(angles), 2.0 * np.sin(angles)))
            calculated = rotating_calipers_diameter(polygon).diameter
            expected = max(
                np.linalg.norm(polygon[i] - polygon[j])
                for i in range(count)
                for j in range(i)
            )
            self.assertAlmostEqual(calculated, expected, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
