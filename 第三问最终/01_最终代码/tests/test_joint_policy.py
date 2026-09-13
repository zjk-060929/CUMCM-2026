from __future__ import annotations

import unittest

import numpy as np

from joint_policy import ArcItem, JointScale125Policy, polygon_centroid, rolling_orienteering_tour
from offline_validation import NullLogger, OfflineClient, generate_sources


class JointPolicyTests(unittest.TestCase):
    def test_polygon_centroid(self) -> None:
        square = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        self.assertTrue(np.allclose(polygon_centroid(square), [1.0, 1.0]))

    def test_arc_planner_is_deterministic_and_visits_every_item(self) -> None:
        items = [
            ArcItem("station", 0, np.asarray([[1.0, 0.0], [1.0, 0.0]]), np.asarray([[1.0, 0.0], [1.0, 0.0]])),
            ArcItem("service", 1, np.asarray([[2.0, 1.0], [2.0, -1.0]]), np.asarray([[3.0, 0.0], [3.0, 0.0]])),
            ArcItem("station", 2, np.asarray([[4.0, 0.0], [4.0, 0.0]]), np.asarray([[4.0, 0.0], [4.0, 0.0]])),
        ]
        first = rolling_orienteering_tour(np.zeros(2), items)
        second = rolling_orienteering_tour(np.zeros(2), items)
        self.assertEqual(first.order, second.order)
        self.assertEqual(first.sides, second.sides)
        self.assertEqual(set(first.order), set(range(len(items))))
        self.assertEqual(len(first.sides), len(items))

    def test_joint_policy_clears_an_unseen_offline_world(self) -> None:
        rng = np.random.default_rng(20260915)
        sources = generate_sources(rng)
        error_seed = int(rng.integers(0, 2**31 - 1))
        client = OfflineClient(sources, error_seed=error_seed)
        summary = JointScale125Policy(
            client,  # type: ignore[arg-type]
            NullLogger(),
            {
                "remaining_real_duration_s": 1200,
                "max_virtual_duration_s": 360000,
            },
        ).run()
        self.assertTrue(summary.completed_normally)
        self.assertTrue(summary.search_certified)
        self.assertEqual(client.cleared_count, len(sources))
        self.assertEqual(set(summary.discovered_channels), set(summary.cleared_channels))


if __name__ == "__main__":
    unittest.main()
