from __future__ import annotations

import math
import unittest

import numpy as np

from geometry import (
    COVERAGE_LAYOUT_CENTER_HEX,
    COVERAGE_LAYOUT_HEPTAGON,
    Observation,
    _axis_aligned_clear_route,
    _clear_route_worst_time,
    conservative_clear_route,
    coverage_route_length,
    coverage_stations,
    guaranteed_shared_observation,
    localization_polygon,
    minimax_coverage_radius,
    minimum_enclosing_circle,
    observation_feasible_polygon,
    observation_separation_deg,
    second_detection_candidates,
    worst_coverage_distance,
)


class GeometryTests(unittest.TestCase):
    def test_minimax_seven_point_radius(self) -> None:
        expected = 1800.0 / (2.0 * math.cos(math.pi / 7.0))
        self.assertAlmostEqual(minimax_coverage_radius(), expected, places=10)
        self.assertLess(worst_coverage_distance(), 1000.0)
        self.assertEqual(len(coverage_stations()), 7)

    def test_center_hex_layout_is_valid_but_has_longer_skeleton(self) -> None:
        self.assertEqual(len(coverage_stations(COVERAGE_LAYOUT_CENTER_HEX)), 7)
        self.assertLess(worst_coverage_distance(COVERAGE_LAYOUT_CENTER_HEX), 1000.0)
        self.assertGreater(
            coverage_route_length(COVERAGE_LAYOUT_CENTER_HEX),
            coverage_route_length(COVERAGE_LAYOUT_HEPTAGON),
        )

    def test_optimized_second_candidates_satisfy_analytic_guarantee(self) -> None:
        observation = Observation(123.0, -456.0, 37.25)
        for point in second_detection_candidates(observation):
            self.assertAlmostEqual(
                float(np.linalg.norm(point - observation.point)),
                math.hypot(535.0, 105.0),
                places=8,
            )
            self.assertTrue(guaranteed_shared_observation(observation, point))
            self.assertAlmostEqual(
                observation_separation_deg(observation, point),
                math.degrees(math.atan2(105.0, 535.0)),
                places=10,
            )
        self.assertGreater(math.degrees(math.atan2(105.0, 535.0)), 1.0)

    def test_clear_grid_covers_bounding_rectangle(self) -> None:
        polygon = np.array(
            [[-61.0, -44.0], [72.0, -44.0], [72.0, 53.0], [-61.0, 53.0]]
        )
        route, _ = conservative_clear_route(polygon, np.array([0.0, 0.0]))
        for x in np.linspace(-61.0, 72.0, 80):
            for y in np.linspace(-44.0, 53.0, 70):
                nearest = min(float(np.linalg.norm(np.array([x, y]) - point)) for point in route)
                self.assertLessEqual(nearest, 20.0 + 1.0e-8)

    def test_clear_grid_skips_cells_outside_a_slender_polygon(self) -> None:
        polygon = np.array(
            [[-120.0, -10.0], [-110.0, -20.0], [120.0, 10.0], [110.0, 20.0]]
        )
        route, radius = conservative_clear_route(
            polygon, np.array([0.0, 0.0])
        )
        bounding_width = float(np.ptp(polygon[:, 0]))
        bounding_height = float(np.ptp(polygon[:, 1]))
        maximum_side = math.sqrt(2.0) * (20.0 - 1.0e-6)
        full_box_cells = math.ceil(bounding_width / maximum_side) * math.ceil(
            bounding_height / maximum_side
        )
        self.assertGreater(radius, 20.0)
        self.assertLess(len(route) - 1, full_box_cells)
        # 稠密采样凸四边形内部，验证每一点距某个清除点不超过20米。
        first, second, third, fourth = polygon
        for u in np.linspace(0.0, 1.0, 61):
            left = (1.0 - u) * first + u * second
            right = (1.0 - u) * fourth + u * third
            for v in np.linspace(0.0, 1.0, 61):
                point = (1.0 - v) * left + v * right
                nearest = min(
                    float(np.linalg.norm(point - candidate)) for candidate in route
                )
                self.assertLessEqual(nearest, 20.0 + 1.0e-8)

    def test_oriented_grid_preserves_coverage_and_baseline_upper_bound(self) -> None:
        angle = math.radians(27.0)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        local_polygon = np.array(
            [[-130.0, -17.0], [130.0, -17.0], [130.0, 17.0], [-130.0, 17.0]]
        )
        polygon = local_polygon @ rotation.T + np.array([90.0, -45.0])
        start = np.array([-400.0, 250.0])
        baseline, _ = _axis_aligned_clear_route(polygon, start)
        route, radius = conservative_clear_route(polygon, start)
        self.assertGreater(radius, 20.0)
        self.assertLessEqual(
            _clear_route_worst_time(start, route),
            _clear_route_worst_time(start, baseline) + 1.0e-9,
        )
        for u in np.linspace(-130.0, 130.0, 81):
            for v in np.linspace(-17.0, 17.0, 31):
                point = np.array([u, v]) @ rotation.T + np.array([90.0, -45.0])
                nearest = min(
                    float(np.linalg.norm(point - candidate)) for candidate in route
                )
                self.assertLessEqual(nearest, 20.0 + 1.0e-8)

    def test_small_enclosing_circle_needs_only_one_clear_point(self) -> None:
        polygon = np.array(
            [[-19.9, -0.1], [19.9, -0.1], [19.9, 0.1], [-19.9, 0.1]]
        )
        route, radius = conservative_clear_route(polygon, np.array([0.0, 0.0]))
        self.assertLessEqual(radius, 20.001)
        self.assertEqual(len(route), 1)

    def test_mec_construction_tolerance_does_not_understate_radius(self) -> None:
        points = np.array([[-20.0, 0.0], [20.0, 0.0], [0.0, 20.000001]])
        center, radius = minimum_enclosing_circle(points)
        actual = float(np.max(np.linalg.norm(points - center, axis=1)))
        self.assertAlmostEqual(radius, actual, places=12)
        route, conservative_radius = conservative_clear_route(
            points, np.array([0.0, 0.0])
        )
        self.assertGreater(conservative_radius, 20.0)
        self.assertGreater(len(route), 1)

    def test_closed_one_degree_error_boundary_remains_feasible(self) -> None:
        # 真实源在首次接收半径上，两个读数分别取误差闭区间的两端；
        # 这是此前会被数值退化误判为空的确定性边界反例。
        source = np.array([1500.0, 0.0])
        first = Observation(0.0, 0.0, 359.0)
        second_point = second_detection_candidates(first)[0]
        true_second_bearing = math.degrees(
            math.atan2(
                float(source[1] - second_point[1]),
                float(source[0] - second_point[0]),
            )
        ) % 360.0
        second = Observation(
            float(second_point[0]),
            float(second_point[1]),
            (true_second_bearing + 1.0) % 360.0,
        )
        polygon = localization_polygon([first, second])
        self.assertGreaterEqual(len(polygon), 3)
        # 极小外扩后真实源应落在闭合保守区域内的数微米邻域。
        self.assertLess(
            min(float(np.linalg.norm(vertex - source)) for vertex in polygon),
            1.0e-2,
        )

    def test_no_signal_disk_can_prune_whole_grid_cells(self) -> None:
        polygon = np.array(
            [[-40.0, -40.0], [40.0, -40.0], [40.0, 40.0], [-40.0, 40.0]]
        )
        full_route, radius = conservative_clear_route(
            polygon, np.array([0.0, 0.0])
        )
        pruned_route, _ = conservative_clear_route(
            polygon,
            np.array([0.0, 0.0]),
            no_signal_points=[np.array([-900.0, 0.0])],
        )
        self.assertGreater(radius, 20.0)
        self.assertLess(len(pruned_route), len(full_route))

    def test_shared_observation_requires_guaranteed_reception(self) -> None:
        first = Observation(0.0, 0.0, 0.0)
        safe = second_detection_candidates(first)[0]
        self.assertTrue(guaranteed_shared_observation(first, safe))
        # 该点可能离最远源超过1000米，但首次正信号同时证明远源具有
        # 更大的实际接收半径；解析判据仍能保证第二次接收。
        angle = math.radians(50.0)
        analytically_safe = 999.0 * np.array([math.cos(angle), math.sin(angle)])
        self.assertTrue(guaranteed_shared_observation(first, analytically_safe))
        unsafe_angle = math.radians(65.0)
        unsafe = 999.0 * np.array(
            [math.cos(unsafe_angle), math.sin(unsafe_angle)]
        )
        self.assertFalse(guaranteed_shared_observation(first, unsafe))
        self.assertFalse(
            guaranteed_shared_observation(first, np.array([800.0, 0.0]))
        )
        self.assertFalse(
            guaranteed_shared_observation(first, np.array([-1000.0, 0.0]))
        )

    def test_no_signal_adds_closer_to_positive_halfplane(self) -> None:
        first = Observation(0.0, 0.0, 0.0)
        negative = np.array([2000.0, 0.0])
        polygon = observation_feasible_polygon([first], [negative])
        # 在原点收到、在(2000,0)无信号，必有 |G|<|G-(2000,0)|，
        # 即G位于垂直平分线x=1000的原点一侧。
        self.assertLessEqual(float(np.max(polygon[:, 0])), 1000.0 + 1.0e-7)


if __name__ == "__main__":
    unittest.main()
