"""排序队列补充检查：分类、包络、数值尺度与接口。"""

import itertools
import math
import unittest

import numpy as np

from q1_localization import (
    _Function,
    _max_envelope,
    _feasible_x,
    _result_as_dict,
    EmptyLocalizationError,
    UnboundedLocalizationError,
    Measurement,
    build_bearing_halfplanes,
    halfplanes_from_coefficients,
    half_plane_intersection,
    make_line,
    point_inside,
    rotating_calipers_diameter,
    solve_halfplanes,
    solve_localization,
    solve_payload,
)


def solve(rows):
    return solve_halfplanes(halfplanes_from_coefficients(rows))


class SpecialCaseTests(unittest.TestCase):
    def test_same_direction_stricter_lower(self):
        r = solve([[1, 0, 0], [1, 0, 2], [-1, 0, -4], [0, 1, 0], [0, -1, -3]])
        self.assertAlmostEqual(r.diameter, math.sqrt(13))

    def test_same_direction_stricter_upper(self):
        r = solve([[1, 0, 0], [-1, 0, -4], [0, 1, 0], [0, -1, -8], [0, -1, -3]])
        self.assertAlmostEqual(r.diameter, 5)

    def test_parallel_vertical_conflict(self):
        with self.assertRaises(EmptyLocalizationError):
            solve([[1, 0, 1], [-1, 0, 0]])

    def test_parallel_horizontal_conflict(self):
        with self.assertRaises(EmptyLocalizationError):
            solve([[0, 1, 1], [0, -1, 0]])

    def test_oblique_parallel_conflict(self):
        with self.assertRaises(EmptyLocalizationError):
            solve([[1, 2, 3], [-1, -2, 0]])

    def test_empty_triangle(self):
        with self.assertRaises(EmptyLocalizationError):
            solve([[1, 0, 0], [0, 1, 0], [-1, -1, 1]])

    def test_entire_plane(self):
        with self.assertRaises(UnboundedLocalizationError):
            half_plane_intersection([])

    def test_half_plane(self):
        with self.assertRaises(UnboundedLocalizationError):
            solve([[1, 2, 3]])

    def test_vertical_strip(self):
        with self.assertRaises(UnboundedLocalizationError):
            solve([[1, 0, 0], [-1, 0, -1]])

    def test_oblique_strip(self):
        with self.assertRaises(UnboundedLocalizationError):
            solve([[1, 1, 0], [-1, -1, -2]])

    def test_vertical_line(self):
        with self.assertRaises(UnboundedLocalizationError):
            solve([[1, 0, 2], [-1, 0, -2]])

    def test_oblique_line(self):
        with self.assertRaises(UnboundedLocalizationError):
            solve([[2, 3, 1], [-2, -3, -1]])

    def test_oblique_ray(self):
        with self.assertRaises(UnboundedLocalizationError):
            solve([[1, -1, 0], [-1, 1, 0], [1, 0, 0]])

    def test_unbounded_quadrant_has_vertex(self):
        with self.assertRaises(UnboundedLocalizationError):
            solve([[1, 0, 0], [0, 1, 0]])

    def test_vertical_segment(self):
        r = solve([[1, 0, 2], [-1, 0, -2], [0, 1, -1], [0, -1, -3]])
        self.assertEqual(r.region_type, "segment")
        self.assertAlmostEqual(r.diameter, 4)

    def test_oblique_segment(self):
        r = solve([[1, -1, 0], [-1, 1, 0], [1, 0, 0], [-1, 0, -3]])
        self.assertEqual(r.region_type, "segment")
        self.assertAlmostEqual(r.diameter, 3 * math.sqrt(2))

    def test_point_without_parallel_pairs(self):
        r = solve([[1, 0, 0], [0, 1, 0], [-1, -1, 0]])
        self.assertEqual(r.region_type, "point")
        self.assertEqual(r.diameter, 0)

    def test_thin_triangle(self):
        r = solve([[1, 0, 0], [0, 1, 0], [-1e-9, -1, -1e-6]])
        self.assertEqual(len(r.vertices), 3)
        self.assertAlmostEqual(r.diameter, 1000, places=5)

    def test_near_parallel_distinct_lines_not_dropped(self):
        r = solve([[0, 1, 0], [1, 0, 0], [-1e-11, -1, -1e-8]])
        self.assertEqual(len(r.vertices), 3)
        self.assertAlmostEqual(r.diameter, 1000, places=3)

    def test_seam_direction_duplicates(self):
        lines = build_bearing_halfplanes(
            [Measurement(0, 0, 0), Measurement(1000, 1000, 270)]
        )
        lines += [make_line(l.p, l.d * 4) for l in lines]
        self.assertAlmostEqual(solve_halfplanes(lines).diameter, 49.385425824, places=7)

    def test_input_order_invariance(self):
        rows = [
            [1, 0, 0],
            [-1, 0, -4],
            [0, 1, 0],
            [0, -1, -3],
            [-1, -1, -5],
            [-1, -2, -10],
        ]
        expected = solve(rows).diameter
        for order in np.random.default_rng(21).random((60, len(rows))).argsort(axis=1):
            self.assertAlmostEqual(
                solve([rows[i] for i in order]).diameter, expected, places=8
            )

    def test_many_redundant_constraints(self):
        rows = [[1, 0, 0], [-1, 0, -4], [0, 1, 0], [0, -1, -3]]
        rows += [[1, 0, -i] for i in range(150)]
        self.assertAlmostEqual(solve(rows).diameter, 5)

    def test_envelope_pops_redundant_middle(self):
        env = _max_envelope([_Function(0, 0), _Function(1, -20), _Function(2, -2)])
        self.assertEqual(len(env.lines), 2)
        self.assertEqual(env.starts, [-math.inf, 1])

    def test_envelope_against_direct_max(self):
        rng = np.random.default_rng(811)
        fs = [_Function(float(m), float(b)) for m, b in rng.normal(size=(100, 2))]
        env = _max_envelope(fs)
        for x in np.linspace(-100, 100, 401):
            self.assertAlmostEqual(env.at(x), max(f.at(x) for f in fs), places=10)

    def test_envelope_singleton_feasible_projection(self):
        lo = _max_envelope([_Function(-1, 0), _Function(1, 0)])
        up = _max_envelope([_Function(0, 0)])
        self.assertEqual(_feasible_x(lo, up, -math.inf, math.inf, 1e-10), (0, 0))

    def test_small_scale_calipers(self):
        angles = np.arange(17) * 2 * math.pi / 17
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        points[4] *= 1.03
        from q1_localization import _convex_hull

        for scale in [1, 1e-3, 1e-6]:
            polygon = _convex_hull(list(points * scale), 1e-14)
            expected = max(
                np.linalg.norm(a - b) for a, b in itertools.combinations(polygon, 2)
            )
            self.assertAlmostEqual(
                rotating_calipers_diameter(polygon).diameter,
                expected,
                delta=1e-12 * scale,
            )

    def test_large_translation(self):
        measurements = [Measurement(0, 0, 0), Measurement(1000, 1000, 269)]
        reference = solve_localization(measurements)
        result = solve_localization(
            [Measurement(m.x + 2e6, m.y - 2e6, m.bearing_deg) for m in measurements]
        )
        self.assertAlmostEqual(result.diameter, reference.diameter, places=7)

    def test_rotated_degenerate_segment(self):
        for angle in [0, 13, 45, 89, 123, 179, 271]:
            r = math.radians(angle)
            rotation = np.array(
                [[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]]
            )
            lines = halfplanes_from_coefficients(
                [[1, -1, 0], [-1, 1, 0], [1, 0, 0], [-1, 0, -3]]
            )
            result = solve_halfplanes(
                [make_line(rotation @ l.p, rotation @ l.d) for l in lines]
            )
            self.assertEqual(result.region_type, "segment")
            self.assertAlmostEqual(result.diameter, 3 * math.sqrt(2), places=7)

    def test_invalid_payload_and_directions(self):
        for data in [
            [],
            {},
            {"halfplanes": [[math.nan, 1, 0]]},
            {"measurements": [], "halfplanes": []},
        ]:
            with self.assertRaises((ValueError, TypeError)):
                solve_payload(data)
        for d in [[0, 0], [math.inf, 1]]:
            with self.assertRaises(ValueError):
                make_line([0, 0], d)

    def test_zero_normal_constraints(self):
        with self.assertRaises(EmptyLocalizationError):
            solve([[0, 0, 1]])
        with self.assertRaises(UnboundedLocalizationError):
            solve([[0, 0, 0], [0, 0, -2]])

    def test_output_compatible_and_tagged(self):
        result = solve([[1, 0, 0], [-1, 0, -1], [0, 1, 0], [0, -1, -1]])
        data = _result_as_dict(result)
        self.assertTrue(data["diameter_circle"]["covers_polygon"])
        self.assertIn("upper_lower_deques", data["backend"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
