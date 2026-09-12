"""第一问核心几何算法的回归测试。"""

import math
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from q1_localization import (
    LocalizationError,
    EmptyLocalizationError,
    UnboundedLocalizationError,
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

    def test_bearing_269_counterexample(self) -> None:
        result = solve_localization(
            [Measurement(0, 0, 0), Measurement(1000, 1000, 269)]
        )
        self.assertFalse(result.circle_covers_polygon)
        self.assertAlmostEqual(result.diameter, 49.3628597593074, places=8)
        self.assertAlmostEqual(
            max(result.vertex_distances_to_center), 25.11594102472149, places=8
        )
        brute = max(
            np.linalg.norm(a - b) for a, b in itertools.combinations(result.vertices, 2)
        )
        self.assertAlmostEqual(result.diameter, brute, places=10)

    def test_singleton_from_valid_bearings(self) -> None:
        result = solve_localization(
            [
                Measurement(-100, 0, 1),
                Measurement(100, 0, 181),
                Measurement(0, -100, 89),
                Measurement(0, 100, 269),
            ]
        )
        self.assertEqual(result.region_type, "point")
        np.testing.assert_allclose(result.vertices, [[0, 0]], atol=1e-10)
        self.assertEqual(result.diameter, 0.0)
        self.assertTrue(result.circle_covers_polygon)

    def test_segment_from_valid_bearings(self) -> None:
        result = solve_localization(
            [
                Measurement(-100, 0, 1),
                Measurement(100, 0, 181),
                Measurement(0, -100, 90),
                Measurement(0, 100, 270),
            ]
        )
        extent = 100 * math.tan(math.radians(1))
        self.assertEqual(result.region_type, "segment")
        np.testing.assert_allclose(
            result.vertices, [[-extent, 0], [extent, 0]], atol=1e-10
        )
        self.assertAlmostEqual(result.diameter, 2 * extent, places=10)
        self.assertTrue(result.circle_covers_polygon)

    def test_singleton_without_opposite_parallel_lines(self) -> None:
        lines = [
            make_line(np.zeros(2), np.array(d)) for d in [(0, -1), (1, 0), (-1, 1)]
        ]
        np.testing.assert_allclose(half_plane_intersection(lines), [[0, 0]], atol=1e-10)

    def test_conflicting_parallel_halfplanes_are_empty(self) -> None:
        lines = [
            make_line(np.array([0, 1]), np.array([1, 0])),
            make_line(np.array([0, 0]), np.array([-1, 0])),
        ]
        with self.assertRaises(EmptyLocalizationError):
            half_plane_intersection(lines)

    def test_conflicting_nonparallel_halfplanes_are_empty(self) -> None:
        # x>=0, y>=0, x+y<=-1.
        lines = [
            make_line(np.array(p), np.array(d))
            for p, d in [
                ((0, 0), (0, -1)),
                ((0, 0), (1, 0)),
                ((-1, 0), (-1, 1)),
            ]
        ]
        with self.assertRaises(EmptyLocalizationError):
            half_plane_intersection(lines)

    def test_unbounded_strip_line_ray_and_plane(self) -> None:
        cases = [
            [],
            [((0, 0), (1, 0))],
            [((0, 0), (1, 0)), ((0, 1), (-1, 0))],
            [((0, 0), (1, 0)), ((0, 0), (-1, 0))],
            [((0, 0), (1, 0)), ((0, 0), (-1, 0)), ((0, 0), (0, -1))],
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(UnboundedLocalizationError):
                half_plane_intersection(
                    [make_line(np.array(p), np.array(d)) for p, d in raw]
                )

    def test_repeated_measurements_do_not_change_region(self) -> None:
        measurements = [Measurement(0, 0, 0), Measurement(1000, 1000, 269)]
        original = solve_localization(measurements)
        repeated = solve_localization(measurements * 4)
        np.testing.assert_allclose(original.vertices, repeated.vertices, atol=1e-9)

    def test_redundant_parallel_constraints(self) -> None:
        raw = [
            ((0, 0), (1, 0)),
            ((1, 0), (0, 1)),
            ((1, 1), (-1, 0)),
            ((0, 1), (0, -1)),
            ((0, -10), (1, 0)),
            ((0, -1), (1, 0)),
        ]
        vertices = half_plane_intersection(
            [make_line(np.array(p), np.array(d)) for p, d in raw]
        )
        np.testing.assert_allclose(
            vertices, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-10
        )

    def test_angle_wrap_and_negative_angles(self) -> None:
        reference = solve_localization(
            [Measurement(0, 0, 0), Measurement(1000, 1000, 270)]
        )
        for first, second in [(360, -90), (-360, 630), (720, 990)]:
            result = solve_localization(
                [Measurement(0, 0, first), Measurement(1000, 1000, second)]
            )
            np.testing.assert_allclose(result.vertices, reference.vertices, atol=1e-9)
        for angle in [-0.5, 359.5, 0.5]:
            point = 100 * np.array(
                [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
            )
            for line in build_bearing_halfplanes([Measurement(0, 0, 0)]):
                self.assertGreaterEqual(
                    float(np.linalg.det(np.vstack([line.d, point - line.p]))), -1e-10
                )

    def test_nearly_parallel_bounded_region(self) -> None:
        # 含很小交角的有界平行四边形；不能误报无界。
        delta = 1e-7
        lines = [
            make_line(np.array(p), np.array(d))
            for p, d in [
                ((0, 0), (1, 0)),
                ((0, 1), (-1, 0)),
                ((0, 0), (-1, -delta)),
                ((1, 0), (1, delta)),
            ]
        ]
        vertices = half_plane_intersection(lines)
        self.assertEqual(len(vertices), 4)
        expected = math.hypot(1 / delta + 1, 1)
        self.assertAlmostEqual(
            rotating_calipers_diameter(vertices).diameter / expected, 1, places=10
        )

    def test_translation_and_rotation_preserve_diameter(self) -> None:
        base = [Measurement(0, 0, 0), Measurement(1000, 1000, 269)]
        reference = solve_localization(base)
        angle = math.radians(43)
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        measurements = []
        for item in base:
            p = rotation @ item.point + np.array([1000000, -1000000])
            measurements.append(Measurement(*p, item.bearing_deg + 43))
        result = solve_localization(measurements)
        self.assertAlmostEqual(result.diameter, reference.diameter, places=7)
        self.assertEqual(result.circle_covers_polygon, reference.circle_covers_polygon)

    def test_invalid_numeric_inputs(self) -> None:
        for error in [0, 90, -1, math.nan, math.inf]:
            with self.subTest(error=error), self.assertRaises(ValueError):
                solve_localization([Measurement(0, 0, 0)], error)
        for measurement in [
            Measurement(math.nan, 0, 0),
            Measurement(0, math.inf, 0),
            Measurement(0, 0, math.inf),
        ]:
            with self.assertRaises(ValueError):
                solve_localization([measurement])
        with self.assertRaises(ValueError):
            solve_localization([])

    def test_random_bearings_against_independent_vertex_enumeration(self) -> None:
        random = np.random.default_rng(91713)
        for trial in range(300):
            count = int(random.integers(3, 11))
            target = random.uniform(-800, 800, 2)
            angles = np.arange(count) * 2 * math.pi / count + random.uniform(
                -0.08, 0.08, count
            )
            angles += random.uniform(0, 2 * math.pi)
            points = target + np.column_stack(
                (np.cos(angles), np.sin(angles))
            ) * random.uniform(100, 900, (count, 1))
            bearings = np.degrees(
                np.arctan2(target[1] - points[:, 1], target[0] - points[:, 0])
            )
            bearings += random.uniform(-0.99, 0.99, count)
            measurements = [Measurement(*p, b) for p, b in zip(points, bearings)]
            # 独立地建立 A*x<=b，并枚举所有两两交点；不调用被测半平面交。
            rows, bounds = [], []
            for p, b in zip(points, bearings):
                lo, hi = np.radians([b - 1, b + 1])
                for row in [
                    np.array([math.sin(lo), -math.cos(lo)]),
                    np.array([-math.sin(hi), math.cos(hi)]),
                ]:
                    rows.append(row)
                    bounds.append(row @ p)
            matrix, rhs = np.asarray(rows), np.asarray(bounds)
            expected = []
            for i, j in itertools.combinations(range(len(rows)), 2):
                if abs(np.linalg.det(matrix[[i, j]])) < 1e-12:
                    continue
                vertex = np.linalg.solve(matrix[[i, j]], rhs[[i, j]])
                if np.all(matrix @ vertex <= rhs + 1e-7) and not any(
                    np.linalg.norm(vertex - p) < 1e-6 for p in expected
                ):
                    expected.append(vertex)
            result = solve_localization(measurements)
            with self.subTest(trial=trial):
                self.assertEqual(len(result.vertices), len(expected))
                for vertex in result.vertices:
                    self.assertLess(
                        min(np.linalg.norm(vertex - p) for p in expected), 1e-6
                    )
                diameter = max(
                    np.linalg.norm(a - b)
                    for a, b in itertools.combinations(expected, 2)
                )
                self.assertAlmostEqual(result.diameter, diameter, places=6)
                self.assertTrue(np.all(matrix @ target <= rhs + 1e-7))

    def test_cli_json_statuses(self) -> None:
        script = Path(__file__).with_name("q1_localization.py")
        cases = [
            ("unbounded", {"measurements": [{"x": 0, "y": 0, "bearing_deg": 0}]}),
            (
                "empty",
                {
                    "measurements": [
                        {"x": 0, "y": 1, "bearing_deg": 1},
                        {"x": 0, "y": 0, "bearing_deg": 181},
                    ]
                },
            ),
            ("invalid_input", {"measurements": []}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for status, payload in cases:
                path.write_text(json.dumps(payload), encoding="utf-8")
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "utf8",
                        "-B",
                        str(script),
                        "--input",
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(json.loads(completed.stdout)["status"], status)
                self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
