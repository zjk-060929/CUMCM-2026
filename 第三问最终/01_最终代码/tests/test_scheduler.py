from __future__ import annotations

import itertools
import unittest

import numpy as np

from scheduler import (
    current_edge_is_cheapest,
    insertion_cost,
    optimal_candidate_route,
)


def brute_force_route(start, candidates_by_channel, terminal=None):
    channels = sorted(candidates_by_channel)
    best = float("inf")
    best_signature = None
    for order in itertools.permutations(channels):
        for sides in itertools.product((0, 1), repeat=len(order)):
            position = np.asarray(start, dtype=float)
            distance = 0.0
            for channel, side in zip(order, sides):
                point = np.asarray(candidates_by_channel[channel][side], dtype=float)
                distance += float(np.linalg.norm(point - position))
                position = point
            if terminal is not None:
                distance += float(np.linalg.norm(np.asarray(terminal) - position))
            signature = (order, sides)
            if distance < best - 1.0e-10 or (
                abs(distance - best) <= 1.0e-10
                and (best_signature is None or signature < best_signature)
            ):
                best = distance
                best_signature = signature
    return best


class SchedulerTests(unittest.TestCase):
    def test_dynamic_programming_matches_exhaustive_enumeration(self) -> None:
        start = np.array([0.0, 0.0])
        terminal = np.array([7.0, -1.0])
        candidates = {
            3: (np.array([1.0, 3.0]), np.array([2.0, -2.0])),
            8: (np.array([4.0, 4.0]), np.array([3.0, -3.0])),
            11: (np.array([6.0, 2.0]), np.array([5.0, -4.0])),
            17: (np.array([8.0, 3.0]), np.array([7.0, -5.0])),
        }
        plan = optimal_candidate_route(start, candidates, terminal)
        expected = brute_force_route(start, candidates, terminal)
        self.assertAlmostEqual(plan.distance_m, expected, places=10)
        self.assertEqual(
            {decision.channel for decision in plan.decisions}, set(candidates)
        )
        self.assertEqual(len(plan.decisions), len(candidates))

    def test_empty_plan_only_contains_terminal_distance(self) -> None:
        plan = optimal_candidate_route(
            np.array([1.0, 2.0]), {}, terminal=np.array([4.0, 6.0])
        )
        self.assertEqual(plan.decisions, ())
        self.assertAlmostEqual(plan.distance_m, 5.0)

    def test_insertion_cost_uses_better_of_two_candidates(self) -> None:
        start = np.array([0.0, 0.0])
        end = np.array([10.0, 0.0])
        candidates = (np.array([5.0, 0.0]), np.array([5.0, 5.0]))
        self.assertAlmostEqual(insertion_cost(start, end, candidates), 0.0)

    def test_current_edge_must_beat_every_future_edge(self) -> None:
        candidates = (np.array([5.0, 1.0]), np.array([5.0, -1.0]))
        is_cheapest, current, best = current_edge_is_cheapest(
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            [(np.array([20.0, 0.0]), np.array([30.0, 0.0]))],
            candidates,
        )
        self.assertTrue(is_cheapest)
        self.assertAlmostEqual(current, best)

        is_cheapest, current, best = current_edge_is_cheapest(
            np.array([20.0, 0.0]),
            np.array([30.0, 0.0]),
            [(np.array([0.0, 0.0]), np.array([10.0, 0.0]))],
            candidates,
        )
        self.assertFalse(is_cheapest)
        self.assertGreater(current, best)


if __name__ == "__main__":
    unittest.main()
