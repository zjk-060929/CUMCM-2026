from __future__ import annotations

import hashlib
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from geometry import Observation, guaranteed_shared_observation, second_detection_candidates
from offline_validation import (
    HiddenSource,
    OfflineClient,
    SpatiallyCorrelatedOfflineClient,
    generate_sources,
)
from policy import PolicyError, SearchFirstPolicy
from protocol import JsonlLogger


class OfflineSimulatorClient:
    """仅供单元测试的内存环境，不创建监听端口。"""

    worst_case_request_wait_s = 0.01

    def __init__(self, source_count: int = 10):
        self.position = np.zeros(2)
        self.receiver_channel = 1
        self.virtual_time_s = 0.0
        self.sources = {}
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for index, channel in enumerate(range(1, source_count + 1)):
            radius = 260.0 + 80.0 * index
            angle = index * golden_angle
            self.sources[channel] = radius * np.array([math.cos(angle), math.sin(angle)])
        self.cleared = set()
        self.last_action_started_monotonic_s = time.monotonic()

    def _move(self, x: float, y: float) -> np.ndarray:
        target = np.array([x, y], dtype=float)
        self.virtual_time_s += float(np.linalg.norm(target - self.position)) / 5.0
        self.position = target
        return target

    @staticmethod
    def _error(channel: int, point: np.ndarray) -> float:
        key = f"{channel}:{point[0]:.6f}:{point[1]:.6f}".encode()
        digest = hashlib.blake2b(key, digest_size=8).digest()
        unit = int.from_bytes(digest, "big") / float(2**64 - 1)
        return 1.8 * unit - 0.9

    def measure(self, x: float, y: float, channel: int):
        point = self._move(x, y)
        if channel != self.receiver_channel:
            self.virtual_time_s += 1.0
        self.receiver_channel = channel
        self.virtual_time_s += 5.0
        source = self.sources.get(channel)
        if source is None or channel in self.cleared:
            result = "no_signal"
            extra = {}
        else:
            distance = float(np.linalg.norm(source - point))
            if distance > 1000.0 + 1.0e-9:
                result = "no_signal"
                extra = {}
            elif distance <= 5.0:
                result = "near"
                extra = {}
            else:
                result = "direction"
                bearing = math.degrees(math.atan2(source[1] - point[1], source[0] - point[0]))
                bearing = round((bearing + self._error(channel, point)) % 360.0, 2) % 360.0
                extra = {"svd_deg": bearing}
        return {
            "accepted": True,
            "real_timestamp_ms": 1,
            "virtual_time_s": self.virtual_time_s,
            "measure_result": result,
            **extra,
        }

    def clear(self, x: float, y: float, channel: int):
        point = self._move(x, y)
        source = self.sources.get(channel)
        success = (
            source is not None
            and channel not in self.cleared
            and float(np.linalg.norm(source - point)) <= 20.0 + 1.0e-9
        )
        self.virtual_time_s += 5.0 if success else 3.0
        if success:
            self.cleared.add(channel)
        return {
            "accepted": True,
            "real_timestamp_ms": 1,
            "virtual_time_s": self.virtual_time_s,
            "clear_result": "success" if success else "no_target_in_range",
        }


class OfflinePolicyTests(unittest.TestCase):
    def test_three_to_five_metre_remeasure_does_not_shrink_region(self) -> None:
        client = OfflineSimulatorClient(source_count=1)
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "correlated-short-baseline.jsonl", "offline")
            policy = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                logger,
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
            )
            first = np.zeros(2)
            response = policy._measure(first, 1)
            self.assertIsNotNone(policy._record_direction(1, first, response))
            close = np.array([3.0, 4.0])
            response = policy._measure(close, 1)
            self.assertIsNone(policy._record_direction(1, close, response))
        self.assertEqual(len(policy.observations[1]), 1)
        self.assertEqual(policy.correlated_bearing_ignored_count, 1)

    def test_spatial_error_is_bounded_and_changes_slowly_over_five_metres(self) -> None:
        source = HiddenSource(1, np.array([800.0, 0.0]), 1200.0)
        client = SpatiallyCorrelatedOfflineClient(
            [source], error_seed=20260913, correlation_scale_m=200.0
        )
        first = client._bearing_error(1, np.array([73.0, -91.0]))
        nearby = client._bearing_error(1, np.array([76.0, -87.0]))
        self.assertLessEqual(abs(first), 0.99)
        self.assertLessEqual(abs(nearby), 0.99)
        self.assertLessEqual(abs(nearby - first), 0.10)

    def test_single_center_clear_failure_triggers_one_local_measurement(self) -> None:
        class FirstClearFailsClient(OfflineSimulatorClient):
            def __init__(self) -> None:
                super().__init__(source_count=1)
                self.sources[1] = np.zeros(2)
                self.injected_failure = False
                self.measure_positions: list[np.ndarray] = []

            def measure(self, x: float, y: float, channel: int):
                self.measure_positions.append(np.array([x, y], dtype=float))
                return super().measure(x, y, channel)

            def clear(self, x: float, y: float, channel: int):
                if not self.injected_failure:
                    self.injected_failure = True
                    self._move(x, y)
                    self.virtual_time_s += 3.0
                    return {
                        "accepted": True,
                        "real_timestamp_ms": 1,
                        "virtual_time_s": self.virtual_time_s,
                        "clear_result": "no_target_in_range",
                    }
                return super().clear(x, y, channel)

        client = FirstClearFailsClient()
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "single-center.jsonl", "offline-single")
            policy = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                logger,
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
            )
            policy.observations[1] = [
                # 具体多边形由下面的几何桩返回；这里只需满足“已有两次观测”。
                Observation(0.0, -100.0, 90.0),
                Observation(-100.0, 0.0, 0.0),
            ]
            policy.discovered.add(1)
            tiny_polygon = np.array(
                [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]], dtype=float
            )
            with patch("policy.localization_polygon", return_value=tiny_polygon), patch(
                "policy.conservative_clear_route",
                return_value=(np.array([[0.0, 0.0]]), 1.5),
            ):
                policy._service_channel(1)

        self.assertTrue(client.injected_failure)
        self.assertIn(1, policy.cleared)
        self.assertEqual(policy.reactive_refinement_count, 1)
        self.assertEqual(len(client.measure_positions), 1)
        self.assertTrue(np.allclose(client.measure_positions[0], np.zeros(2)))

    def test_reached_second_point_can_serve_another_compatible_channel(self) -> None:
        client = OfflineSimulatorClient(source_count=2)
        client.sources[1] = np.array([800.0, 0.0])
        angle = math.radians(30.0)
        client.sources[2] = 800.0 * np.array([math.cos(angle), math.sin(angle)])
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "piggyback.jsonl", "offline-piggyback")
            policy = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                logger,
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
            )
            origin = np.zeros(2)
            for channel in (1, 2):
                response = policy._measure(origin, channel)
                policy._record_direction(channel, origin, response)
            candidate = second_detection_candidates(policy.observations[1][0])[0]
            self.assertTrue(
                guaranteed_shared_observation(policy.observations[2][0], candidate)
            )
            policy._service_channel(1, preferred_candidate_index=0)
        self.assertEqual(len(policy.observations[1]), 2)
        self.assertEqual(len(policy.observations[2]), 2)
        self.assertEqual(policy.piggyback_observation_count, 1)

    def test_second_bearing_returns_control_to_rolling_planner_before_clear(self) -> None:
        client = OfflineSimulatorClient(source_count=10)
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "one-action.jsonl", "offline-one-action")
            policy = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                logger,
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
            )
            origin = np.zeros(2)
            first_response = policy._measure(origin, 1)
            policy._record_direction(1, origin, first_response)
            policy._service_channel(1, preferred_candidate_index=0)
            self.assertEqual(len(policy.observations[1]), 2)
            self.assertNotIn(1, policy.cleared)
            policy._service_channel(1)
        self.assertIn(1, policy.cleared)

    def test_origin_information_scan_has_zero_movement_and_finds_central_sources(self) -> None:
        client = OfflineSimulatorClient(source_count=10)
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "origin.jsonl", "offline-origin")
            policy = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                logger,
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
            )
            policy._scan_origin_information()
        self.assertTrue(np.allclose(policy.state.position, np.zeros(2)))
        self.assertEqual(policy.state.measure_count, 20)
        self.assertEqual(policy.discovered, set(range(1, 11)))
        # 20次测量和19次切频；没有任何移动时间。
        self.assertAlmostEqual(policy.state.virtual_time_s, 119.0, places=8)

    def test_search_first_clears_ten_hidden_sources_without_truth_access(self) -> None:
        client = OfflineSimulatorClient()
        enter_response = {
            "remaining_real_duration_s": 1200,
            "max_virtual_duration_s": 360000,
        }
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "offline.jsonl", "offline")
            policy = SearchFirstPolicy(client, logger, enter_response)  # type: ignore[arg-type]
            summary = policy.run()
        self.assertTrue(summary.completed_normally)
        self.assertEqual(summary.discovered_channels, tuple(range(1, 11)))
        self.assertEqual(summary.cleared_channels, tuple(range(1, 11)))
        self.assertEqual(client.cleared, set(range(1, 11)))
        self.assertTrue(summary.search_certified)
        self.assertEqual(summary.certified_absent_channels, tuple(range(11, 21)))
        self.assertLess(summary.final_virtual_time_s, 360000)

    def test_fewer_than_ten_discoveries_is_not_reported_as_success(self) -> None:
        client = OfflineSimulatorClient(source_count=9)
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "offline.jsonl", "offline")
            policy = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                logger,
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
            )
            with self.assertRaises(PolicyError):
                policy.run()

    def test_sixteen_sources_trigger_valid_early_stop(self) -> None:
        client = OfflineSimulatorClient(source_count=16)
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "offline.jsonl", "offline")
            policy = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                logger,
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
            )
            summary = policy.run()
        self.assertEqual(len(summary.discovered_channels), 16)
        self.assertEqual(len(summary.cleared_channels), 16)
        self.assertTrue(summary.search_certified)
        self.assertEqual(summary.certified_absent_channels, ())

    def test_origin_finds_sixteen_without_walking_to_a_coverage_station(self) -> None:
        client = OfflineSimulatorClient(source_count=16)
        for index, channel in enumerate(sorted(client.sources)):
            angle = 2.0 * math.pi * index / 16.0
            client.sources[channel] = 700.0 * np.array(
                [math.cos(angle), math.sin(angle)]
            )
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "origin16.jsonl", "offline-origin16")
            policy = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                logger,
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
            )
            policy._scan_origin_information()
            time_before = policy.state.virtual_time_s
            policy._scan_coverage_stations()
        self.assertEqual(len(policy.discovered), 16)
        self.assertTrue(policy.search_certified)
        self.assertEqual(policy.stations_visited, 0)
        self.assertTrue(np.allclose(policy.state.position, np.zeros(2)))
        self.assertAlmostEqual(policy.state.virtual_time_s, time_before, places=8)

    def test_time_selected_supplement_controls_known_tail_case(self) -> None:
        rng = np.random.default_rng(20260911)
        for _ in range(202):
            sources = generate_sources(rng)
            error_seed = int(rng.integers(0, 2**31 - 1))
        client = OfflineClient(sources, error_seed=error_seed)
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory) / "tail.jsonl", "offline-tail")
            summary = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                logger,
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
            ).run()
        self.assertTrue(summary.completed_normally)
        self.assertLess(summary.clear_attempt_count, 50)


if __name__ == "__main__":
    unittest.main()
