"""Reproducible, offline-only 500-world iteration runner for Question 3.

The module contains experimental policies only.  It never imports the live
runner or opens a socket.  Every variant sees exactly the same packed worlds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
for directory in (ROOT, CODE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from geometry import (  # noqa: E402
    COVERAGE_LAYOUT_HEPTAGON,
    MIN_RECEPTION_RADIUS_M,
    coverage_stations,
    guaranteed_shared_observation,
    localization_polygon,
    minimum_enclosing_circle,
    second_detection_candidates,
)
from offline_validation import HiddenSource, NullLogger, generate_sources  # noqa: E402
from policy import CHANNELS, MAX_SOURCE_COUNT, MIN_SOURCE_COUNT, PolicyError, RunSummary, SearchFirstPolicy  # noqa: E402
from scratch_q3_global_rolling import AuditClient  # noqa: E402
from scheduler import current_edge_is_cheapest  # noqa: E402
from q3_fast_scheduler import fast_optimal_candidate_route  # noqa: E402
from q3_orienteering_candidate import (  # noqa: E402
    JointScale125Policy,
    OrderedJointScale125Policy,
)
import policy as policy_module  # noqa: E402


def make_worlds(count: int, seed: int) -> list[tuple[int, list[tuple[int, float, float, float]], int]]:
    rng = np.random.default_rng(seed)
    worlds = []
    for index in range(count):
        sources = generate_sources(rng)
        packed = [
            (
                source.channel,
                float(source.point[0]),
                float(source.point[1]),
                source.reception_radius_m,
            )
            for source in sources
        ]
        error_seed = int(rng.integers(0, 2**31 - 1))
        worlds.append((index, packed, error_seed))
    return worlds


def _coverage_cells(side_m: float = 50.0) -> np.ndarray:
    """Return corners of every square that intersects the 1800 m target disk."""

    edges = np.arange(-1800.0, 1800.0 + side_m, side_m, dtype=float)
    cells: list[np.ndarray] = []
    for ix in range(len(edges) - 1):
        x0, x1 = float(edges[ix]), float(edges[ix + 1])
        nearest_x = min(max(0.0, x0), x1)
        for iy in range(len(edges) - 1):
            y0, y1 = float(edges[iy]), float(edges[iy + 1])
            nearest_y = min(max(0.0, y0), y1)
            if nearest_x * nearest_x + nearest_y * nearest_y > 1800.0**2:
                continue
            cells.append(
                np.asarray(
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    dtype=float,
                )
            )
    return np.asarray(cells, dtype=float)


ADAPTIVE_CELL_CORNERS = _coverage_cells()


def exact_disk_covering_radius(points: list[np.ndarray]) -> float:
    """Maximum distance from the 1800 m target disk to its nearest point.

    Interior maxima of the nearest-site distance occur at Voronoi vertices;
    boundary maxima occur at a stationary point of one boundary distance or
    where two such distances are equal.  Enumerating all triples/pairs includes
    every such candidate (plus harmless non-active candidates).
    """

    if not points:
        return math.inf
    centers = np.asarray(points, dtype=float)
    region_radius = 1800.0
    candidates: list[np.ndarray] = [np.zeros(2, dtype=float)]

    # All circumcentres include every finite Voronoi vertex.
    for first, second, third in itertools.combinations(centers, 3):
        matrix = 2.0 * np.asarray([second - first, third - first], dtype=float)
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= 1.0e-9:
            continue
        rhs = np.asarray(
            [float(np.dot(second, second) - np.dot(first, first)),
             float(np.dot(third, third) - np.dot(first, first))],
            dtype=float,
        )
        center = np.linalg.solve(matrix, rhs)
        if float(np.dot(center, center)) <= region_radius**2 + 1.0e-7:
            candidates.append(center)

    boundary_angles: list[float] = []
    for center in centers:
        phi = math.atan2(float(center[1]), float(center[0]))
        boundary_angles.extend((phi, phi + math.pi))
    for first, second in itertools.combinations(centers, 2):
        vector = second - first
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-12:
            continue
        rhs = float(np.dot(second, second) - np.dot(first, first)) / (
            2.0 * region_radius
        )
        ratio = rhs / norm
        if ratio < -1.0 - 1.0e-12 or ratio > 1.0 + 1.0e-12:
            continue
        ratio = min(1.0, max(-1.0, ratio))
        phi = math.atan2(float(vector[1]), float(vector[0]))
        delta = math.acos(ratio)
        boundary_angles.extend((phi - delta, phi + delta))
    for angle in boundary_angles:
        candidates.append(
            region_radius * np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        )

    values = [
        float(np.min(np.linalg.norm(centers - candidate, axis=1)))
        for candidate in candidates
    ]
    return max(values)


class GatedOpportunityPolicy(SearchFirstPolicy):
    """Reject individually 'cheapest' insertions that are still large detours."""

    name = "scratch_gated_opportunity"
    maximum_insertion_m = 250.0
    maximum_services_per_edge = 100

    def _service_opportunities(self, station_index: int, stations: list[np.ndarray]) -> None:
        next_station = stations[station_index + 1]
        future_stations = stations[station_index + 1 :]
        future_edges = [
            (stations[index], stations[index + 1])
            for index in range(station_index + 1, len(stations) - 1)
        ]
        serviced = 0
        while serviced < self.maximum_services_per_edge:
            pending = set(self.observations) - self.cleared
            pending -= {
                channel
                for channel in pending
                if len(self.observations[channel]) == 1
                and any(
                    guaranteed_shared_observation(self.observations[channel][0], point)
                    for point in future_stations
                )
            }
            eligible: set[int] = set()
            for channel, candidates in self._candidate_map(pending).items():
                is_cheapest, current_cost, _ = current_edge_is_cheapest(
                    self.state.position,
                    next_station,
                    future_edges,
                    candidates,
                )
                if is_cheapest and current_cost <= self.maximum_insertion_m:
                    eligible.add(channel)
            if not eligible:
                return
            plan = self._rolling_plan(eligible, terminal=next_station, phase="gated_opportunity")
            first = plan.decisions[0]
            self._service_channel(first.channel, first.candidate_index)
            serviced += 1


class Gate100Policy(GatedOpportunityPolicy):
    name = "scratch_opportunity_gate100"
    maximum_insertion_m = 100.0


class Gate250Policy(GatedOpportunityPolicy):
    name = "scratch_opportunity_gate250"
    maximum_insertion_m = 250.0


class Gate500Policy(GatedOpportunityPolicy):
    name = "scratch_opportunity_gate500"
    maximum_insertion_m = 500.0


class OnePerEdgePolicy(GatedOpportunityPolicy):
    name = "scratch_opportunity_one_per_edge"
    maximum_insertion_m = math.inf
    maximum_services_per_edge = 1


class FullProxyOpportunityPolicy(SearchFirstPolicy):
    """Gate a one-bearing task by a proxy for both localization and clearance."""

    name = "scratch_full_proxy_opportunity"
    maximum_insertion_m = 500.0

    def _proxy_sequences(self, channel: int) -> list[tuple[np.ndarray, ...]]:
        observations = self.observations[channel]
        if len(observations) >= 2:
            polygon = localization_polygon(observations, self.no_signal_points[channel])
            center, _ = minimum_enclosing_circle(polygon)
            return [(center,)]
        first = observations[0]
        angle = math.radians(first.bearing_deg)
        expected_target = first.point + 855.2631578947 * np.asarray(
            [math.cos(angle), math.sin(angle)]
        )
        return [(candidate, expected_target) for candidate in second_detection_candidates(first)]

    @staticmethod
    def _sequence_insertion(start: np.ndarray, end: np.ndarray, sequence: tuple[np.ndarray, ...]) -> float:
        length = float(np.linalg.norm(sequence[0] - start))
        for before, after in zip(sequence, sequence[1:]):
            length += float(np.linalg.norm(after - before))
        length += float(np.linalg.norm(end - sequence[-1]))
        return length - float(np.linalg.norm(end - start))

    def _edge_cost(self, start: np.ndarray, end: np.ndarray, channel: int) -> float:
        return min(self._sequence_insertion(start, end, sequence) for sequence in self._proxy_sequences(channel))

    def _service_opportunities(self, station_index: int, stations: list[np.ndarray]) -> None:
        next_station = stations[station_index + 1]
        future_stations = stations[station_index + 1 :]
        future_edges = [
            (stations[index], stations[index + 1])
            for index in range(station_index + 1, len(stations) - 1)
        ]
        while True:
            pending = set(self.observations) - self.cleared
            pending -= {
                channel
                for channel in pending
                if len(self.observations[channel]) == 1
                and any(
                    guaranteed_shared_observation(self.observations[channel][0], point)
                    for point in future_stations
                )
            }
            costs: dict[int, float] = {}
            for channel in pending:
                current = self._edge_cost(self.state.position, next_station, channel)
                future = [self._edge_cost(start, end, channel) for start, end in future_edges]
                if current <= min([current, *future]) + 1.0e-8 and current <= self.maximum_insertion_m:
                    costs[channel] = current
            if not costs:
                return
            # Let the existing exact candidate DP choose among tasks that pass
            # the full-service proxy gate.
            plan = self._rolling_plan(set(costs), terminal=next_station, phase="full_proxy_opportunity")
            first = plan.decisions[0]
            self._service_channel(first.channel, first.candidate_index)


class FullProxy250Policy(FullProxyOpportunityPolicy):
    name = "scratch_full_proxy250"
    maximum_insertion_m = 250.0


class FullProxy500Policy(FullProxyOpportunityPolicy):
    name = "scratch_full_proxy500"
    maximum_insertion_m = 500.0


class FullProxy1000Policy(FullProxyOpportunityPolicy):
    name = "scratch_full_proxy1000"
    maximum_insertion_m = 1000.0


class AdaptiveOrientationPolicy(SearchFirstPolicy):
    """Rotate/reverse the valid heptagon using only origin observations.

    Every tested orientation is the same analytically valid seven-disk cover.
    The score inserts a conservative two-stop proxy (safe second point followed
    by the conditional radial target proxy) into the coverage skeleton.  It is
    used only to choose route orientation; feedback and certificates are still
    handled by the unchanged production policy.
    """

    name = "scratch_adaptive_heptagon_orientation"
    orientation_step_deg = 5.0
    expected_origin_range_m = 855.2631578947

    @staticmethod
    def _sequence_insertion(
        start: np.ndarray,
        end: np.ndarray,
        sequence: tuple[np.ndarray, ...],
    ) -> float:
        distance = float(np.linalg.norm(sequence[0] - start))
        for before, after in zip(sequence, sequence[1:]):
            distance += float(np.linalg.norm(after - before))
        distance += float(np.linalg.norm(end - sequence[-1]))
        return distance - float(np.linalg.norm(end - start))

    def _orientation_score(self, stations: list[np.ndarray]) -> float:
        edges = list(zip([np.zeros(2), *stations[:-1]], stations))
        score = sum(float(np.linalg.norm(end - start)) for start, end in edges)
        for channel in sorted(self.observations):
            if channel in self.cleared or len(self.observations[channel]) != 1:
                continue
            first = self.observations[channel][0]
            # Only origin observations exist when this score is evaluated.
            angle = math.radians(first.bearing_deg)
            proxy = first.point + self.expected_origin_range_m * np.asarray(
                [math.cos(angle), math.sin(angle)], dtype=float
            )
            candidates = second_detection_candidates(first)
            score += min(
                self._sequence_insertion(start, end, (candidate, proxy))
                for start, end in edges
                for candidate in candidates
            )
        return score

    def _scan_origin_information(self) -> None:
        super()._scan_origin_information()
        if len(self.discovered) == MAX_SOURCE_COUNT:
            return
        radius = float(np.linalg.norm(self.coverage_points[0]))
        choices: list[tuple[float, list[np.ndarray]]] = []
        steps = int(round(360.0 / self.orientation_step_deg))
        for step in range(steps):
            theta = math.radians(step * self.orientation_step_deg)
            for direction in (1.0, -1.0):
                stations = [
                    radius
                    * np.asarray(
                        [
                            math.cos(theta + direction * 2.0 * math.pi * index / 7.0),
                            math.sin(theta + direction * 2.0 * math.pi * index / 7.0),
                        ],
                        dtype=float,
                    )
                    for index in range(7)
                ]
                choices.append((self._orientation_score(stations), stations))
        _, self.coverage_points = min(choices, key=lambda item: item[0])


class DynamicCoverageOrderPolicy(SearchFirstPolicy):
    """Recompute the shortest remaining seven-station order after each detour."""

    name = "scratch_dynamic_coverage_order"

    def _shortest_remaining_order(self, remaining: set[int]) -> list[int]:
        if not remaining:
            return []
        best_distance = math.inf
        best_order: tuple[int, ...] | None = None
        for order in itertools.permutations(sorted(remaining)):
            before = self.state.position
            distance = 0.0
            for index in order:
                after = self.coverage_points[index]
                distance += float(np.linalg.norm(after - before))
                before = after
            if distance < best_distance - 1.0e-9:
                best_distance = distance
                best_order = order
        assert best_order is not None
        return list(best_order)

    def _scan_coverage_stations(self) -> None:
        if len(self.discovered) == MAX_SOURCE_COUNT:
            self._update_search_certificate()
            return
        remaining = set(range(len(self.coverage_points)))
        while remaining:
            order = self._shortest_remaining_order(remaining)
            station_index = order[0]
            station = self.coverage_points[station_index]
            unknown = [channel for channel in CHANNELS if channel not in self.discovered]
            if self.state.receiver_channel in unknown:
                unknown.remove(self.state.receiver_channel)
                unknown.insert(0, self.state.receiver_channel)
            for channel in unknown:
                response = self._measure(station, channel)
                result = response["measure_result"]
                if result == "direction":
                    self._record_direction(channel, station, response)
                elif result == "near":
                    self._clear_near(channel, station)
                else:
                    self._record_no_signal(
                        channel,
                        station,
                        phase="dynamic_coverage_order",
                        station_index=station_index,
                    )
                if len(self.discovered) == MAX_SOURCE_COUNT:
                    break
            self.stations_visited += 1
            remaining.remove(station_index)
            if self.shared_observations:
                self._collect_shared_observations(station, station_index)
            if len(self.discovered) == MAX_SOURCE_COUNT:
                break
            if self.opportunistic_during_search and remaining:
                future_order = self._shortest_remaining_order(remaining)
                synthetic_route = [station] + [
                    self.coverage_points[index] for index in future_order
                ]
                self._service_opportunities(0, synthetic_route)
        self._update_search_certificate()


class AdaptiveSubsetCoveragePolicy(SearchFirstPolicy):
    """Use profitable clear-stop scans to replace a subset of fixed stations."""

    name = "scratch_adaptive_subset_coverage"
    certificate_tolerance_m = 1.0e-6
    minimum_predicted_saving_s = 1.0
    maximum_adaptive_scans = 4

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._common_scan_points: list[np.ndarray] = []
        self._remaining_indices: set[int] = set(range(len(self.coverage_points)))
        self.adaptive_scan_count = 0
        self.fallback_scan_count = 0

    @staticmethod
    def _path_distance(start: np.ndarray, order: tuple[int, ...], stations: list[np.ndarray]) -> float:
        distance = 0.0
        before = start
        for index in order:
            after = stations[index]
            distance += float(np.linalg.norm(after - before))
            before = after
        return distance

    def _best_feasible_fixed_route(
        self,
        scan_points: list[np.ndarray],
        remaining: set[int],
    ) -> tuple[list[int], float, float]:
        """Return lowest estimated-time fixed subset and its geometric radius."""

        unknown_count = len(set(CHANNELS) - self.discovered)
        best_score = math.inf
        best_distance = math.inf
        best_radius = math.inf
        best_order: tuple[int, ...] | None = None
        ordered_remaining = sorted(remaining)
        for subset_size in range(len(ordered_remaining) + 1):
            for subset in itertools.combinations(ordered_remaining, subset_size):
                centers = [*scan_points, *[self.coverage_points[index] for index in subset]]
                radius = exact_disk_covering_radius(centers)
                if radius > MIN_RECEPTION_RADIUS_M - self.certificate_tolerance_m:
                    continue
                if not subset:
                    permutations = [()]
                else:
                    permutations = itertools.permutations(subset)
                for order in permutations:
                    distance = self._path_distance(self.state.position, order, self.coverage_points)
                    # At a fresh station the receiver is normally on a known
                    # channel, so switching+measurement is bounded by 6 s per
                    # still-unknown channel.  This is an action-time estimate,
                    # not an accuracy/time weighting coefficient.
                    score = distance / 5.0 + 6.0 * unknown_count * len(order)
                    key = (score, distance, len(order), order)
                    best_key = (
                        best_score,
                        best_distance,
                        len(best_order) if best_order is not None else 10**9,
                        best_order or (),
                    )
                    if key < best_key:
                        best_score = score
                        best_distance = distance
                        best_radius = radius
                        best_order = order
        if best_order is None:
            raise PolicyError("剩余七点中不存在保证覆盖子集")
        return list(best_order), best_score, best_radius

    def _scan_unknown_here(self, *, phase: str) -> None:
        point = self.state.position.copy()
        unknown = [channel for channel in CHANNELS if channel not in self.discovered]
        if self.state.receiver_channel in unknown:
            unknown.remove(self.state.receiver_channel)
            unknown.insert(0, self.state.receiver_channel)
        for channel in unknown:
            response = self._measure(point, channel)
            result = response["measure_result"]
            if result == "direction":
                self._record_direction(channel, point, response)
            elif result == "near":
                self._clear_near(channel, point)
            else:
                self._record_no_signal(channel, point, phase=phase)
            if len(self.discovered) == MAX_SOURCE_COUNT:
                break

    def _maybe_scan_service_stop(self) -> None:
        if (
            len(self.discovered) == MAX_SOURCE_COUNT
            or self.adaptive_scan_count >= self.maximum_adaptive_scans
        ):
            return
        point = self.state.position.copy()
        if any(float(np.linalg.norm(point - old)) <= 1.0e-7 for old in self._common_scan_points):
            return
        before_order, before_score, _ = self._best_feasible_fixed_route(
            self._common_scan_points,
            self._remaining_indices,
        )
        after_points = [*self._common_scan_points, point]
        after_order, after_future_score, _ = self._best_feasible_fixed_route(
            after_points,
            self._remaining_indices,
        )
        scan_cost = 6.0 * len(set(CHANNELS) - self.discovered)
        predicted_saving = before_score - (scan_cost + after_future_score)
        if predicted_saving < self.minimum_predicted_saving_s:
            return
        self._scan_unknown_here(phase="adaptive_subset_service_stop")
        self.adaptive_scan_count += 1
        self._common_scan_points.append(point)
        # Re-evaluate with the actual discoveries and safely discard stations
        # outside a newly certified subset.
        selected, _, radius = self._best_feasible_fixed_route(
            self._common_scan_points,
            self._remaining_indices,
        )
        self._remaining_indices = set(selected)
        self.logger.write(
            "adaptive_subset_selected",
            predicted_saving_s=predicted_saving,
            remaining_station_indices=[index + 1 for index in selected],
            certified_covering_radius_m=radius,
        )

    def _service_one_opportunity(self, order: list[int]) -> bool:
        if not order:
            return False
        stations = [self.coverage_points[index] for index in order]
        next_station = stations[0]
        future_stations = stations
        future_edges = list(zip(stations[:-1], stations[1:]))
        pending = set(self.observations) - self.cleared
        pending -= {
            channel
            for channel in pending
            if len(self.observations[channel]) == 1
            and any(
                guaranteed_shared_observation(self.observations[channel][0], point)
                for point in future_stations
            )
        }
        eligible: set[int] = set()
        for channel, candidates in self._candidate_map(pending).items():
            is_cheapest, _, _ = current_edge_is_cheapest(
                self.state.position,
                next_station,
                future_edges,
                candidates,
            )
            if is_cheapest:
                eligible.add(channel)
        if not eligible:
            return False
        plan = self._rolling_plan(eligible, terminal=next_station, phase="adaptive_subset")
        first = plan.decisions[0]
        cleared_before = len(self.cleared)
        self._service_channel(first.channel, first.candidate_index)
        if len(self.cleared) > cleared_before:
            self._maybe_scan_service_stop()
        return True

    def _update_search_certificate(self) -> None:
        if len(self.discovered) == MAX_SOURCE_COUNT:
            self.search_certified = True
            self.certified_absent_channels = set()
            return
        radius = exact_disk_covering_radius(self._common_scan_points)
        self.search_certified = radius <= MIN_RECEPTION_RADIUS_M - self.certificate_tolerance_m
        self.certified_absent_channels = (
            set(CHANNELS) - self.discovered if self.search_certified else set()
        )

    def _scan_coverage_stations(self) -> None:
        if len(self.discovered) == MAX_SOURCE_COUNT:
            self._update_search_certificate()
            return
        while self._remaining_indices and len(self.discovered) < MAX_SOURCE_COUNT:
            order, _, _ = self._best_feasible_fixed_route(
                self._common_scan_points,
                self._remaining_indices,
            )
            while order and self._service_one_opportunity(order):
                if len(self.discovered) == MAX_SOURCE_COUNT:
                    break
                order, _, _ = self._best_feasible_fixed_route(
                    self._common_scan_points,
                    self._remaining_indices,
                )
            if len(self.discovered) == MAX_SOURCE_COUNT or not order:
                break
            station_index = order[0]
            station = self.coverage_points[station_index]
            unknown = [channel for channel in CHANNELS if channel not in self.discovered]
            if self.state.receiver_channel in unknown:
                unknown.remove(self.state.receiver_channel)
                unknown.insert(0, self.state.receiver_channel)
            for channel in unknown:
                response = self._measure(station, channel)
                result = response["measure_result"]
                if result == "direction":
                    self._record_direction(channel, station, response)
                elif result == "near":
                    self._clear_near(channel, station)
                else:
                    self._record_no_signal(
                        channel,
                        station,
                        phase="adaptive_subset_fixed",
                        station_index=station_index,
                    )
                if len(self.discovered) == MAX_SOURCE_COUNT:
                    break
            self.stations_visited += 1
            self.fallback_scan_count += 1
            self._remaining_indices.remove(station_index)
            self._common_scan_points.append(station.copy())
            if self.shared_observations:
                self._collect_shared_observations(station, station_index)
            self._update_search_certificate()
            if self.search_certified:
                break
        self._update_search_certificate()

    def _scan_origin_information(self) -> None:
        super()._scan_origin_information()
        if len(self.discovered) < MAX_SOURCE_COUNT:
            self._common_scan_points.append(np.zeros(2, dtype=float))


class IntegratedAdaptiveCoveragePolicy(SearchFirstPolicy):
    """Keep baseline clustered opportunities, but reuse clear stops for coverage."""

    name = "scratch_integrated_adaptive_coverage"
    adaptive_gain_fraction = 0.15
    maximum_adaptive_scans = 4

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._adaptive_covered = np.zeros(len(ADAPTIVE_CELL_CORNERS), dtype=bool)
        self._adaptive_scan_points: list[np.ndarray] = []
        self.adaptive_scan_count = 0
        self.fallback_scan_count = 0

    @staticmethod
    def _cells_covered_by(point: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(ADAPTIVE_CELL_CORNERS - point, axis=2)
        return np.max(distances, axis=1) < MIN_RECEPTION_RADIUS_M - 1.0e-7

    def _register_coverage_point(self, point: np.ndarray) -> None:
        self._adaptive_covered |= self._cells_covered_by(np.asarray(point, dtype=float))
        self._adaptive_scan_points.append(np.asarray(point, dtype=float).copy())

    def _scan_unknown_service_stop(self) -> None:
        if len(self.discovered) == MAX_SOURCE_COUNT or self.adaptive_scan_count >= self.maximum_adaptive_scans:
            return
        point = self.state.position.copy()
        if any(float(np.linalg.norm(point - old)) <= 1.0e-7 for old in self._adaptive_scan_points):
            return
        covered = self._cells_covered_by(point)
        gain = int(np.count_nonzero(covered & ~self._adaptive_covered))
        uncovered = int(np.count_nonzero(~self._adaptive_covered))
        if gain < max(1, int(math.ceil(self.adaptive_gain_fraction * uncovered))):
            return
        unknown = [channel for channel in CHANNELS if channel not in self.discovered]
        if self.state.receiver_channel in unknown:
            unknown.remove(self.state.receiver_channel)
            unknown.insert(0, self.state.receiver_channel)
        for channel in unknown:
            response = self._measure(point, channel)
            result = response["measure_result"]
            if result == "direction":
                self._record_direction(channel, point, response)
            elif result == "near":
                self._clear_near(channel, point)
            else:
                self._record_no_signal(channel, point, phase="integrated_clear_stop")
            if len(self.discovered) == MAX_SOURCE_COUNT:
                break
        self.adaptive_scan_count += 1
        self._register_coverage_point(point)
        if len(self.discovered) < MAX_SOURCE_COUNT and bool(np.all(self._adaptive_covered)):
            self.search_certified = True
            self.certified_absent_channels = set(CHANNELS) - self.discovered

    def _service_opportunities(self, station_index: int, stations: list[np.ndarray]) -> None:
        next_station = stations[station_index + 1]
        future_stations = stations[station_index + 1 :]
        future_edges = [
            (stations[index], stations[index + 1])
            for index in range(station_index + 1, len(stations) - 1)
        ]
        while not self.search_certified:
            pending = set(self.observations) - self.cleared
            pending -= {
                channel
                for channel in pending
                if len(self.observations[channel]) == 1
                and any(
                    guaranteed_shared_observation(self.observations[channel][0], point)
                    for point in future_stations
                )
            }
            eligible: set[int] = set()
            for channel, candidates in self._candidate_map(pending).items():
                is_cheapest, _, _ = current_edge_is_cheapest(
                    self.state.position, next_station, future_edges, candidates
                )
                if is_cheapest:
                    eligible.add(channel)
            if not eligible:
                return
            plan = self._rolling_plan(eligible, terminal=next_station, phase="integrated_coverage")
            first = plan.decisions[0]
            cleared_before = len(self.cleared)
            self._service_channel(first.channel, first.candidate_index)
            if len(self.cleared) > cleared_before:
                self._scan_unknown_service_stop()

    def _scan_coverage_stations(self) -> None:
        if len(self.discovered) == MAX_SOURCE_COUNT:
            self._update_search_certificate()
            return
        for station_index, station in enumerate(self.coverage_points):
            unknown = [channel for channel in CHANNELS if channel not in self.discovered]
            if self.state.receiver_channel in unknown:
                unknown.remove(self.state.receiver_channel)
                unknown.insert(0, self.state.receiver_channel)
            for channel in unknown:
                response = self._measure(station, channel)
                result = response["measure_result"]
                if result == "direction":
                    self._record_direction(channel, station, response)
                elif result == "near":
                    self._clear_near(channel, station)
                else:
                    self._record_no_signal(
                        channel,
                        station,
                        phase="integrated_fixed_station",
                        station_index=station_index,
                    )
                if len(self.discovered) == MAX_SOURCE_COUNT:
                    break
            self.stations_visited += 1
            self.fallback_scan_count += 1
            self._register_coverage_point(station)
            if self.shared_observations:
                self._collect_shared_observations(station, station_index)
            if len(self.discovered) == MAX_SOURCE_COUNT:
                self._update_search_certificate()
                break
            if bool(np.all(self._adaptive_covered)):
                self.search_certified = True
                self.certified_absent_channels = set(CHANNELS) - self.discovered
                break
            if station_index + 1 < len(self.coverage_points):
                self._service_opportunities(station_index, self.coverage_points)
                if self.search_certified:
                    break
        if not self.search_certified:
            self._update_search_certificate()

    def run(self) -> RunSummary:
        # Mark the origin only after the inherited information scan really has
        # queried it; count-16 termination remains handled by the base policy.
        self._scan_origin_information()
        self._register_coverage_point(np.zeros(2, dtype=float))
        self._scan_coverage_stations()
        if not self.search_certified:
            raise PolicyError("联合覆盖没有形成搜索证书")
        if len(self.discovered) < MIN_SOURCE_COUNT:
            raise PolicyError("联合覆盖发现数低于题面下限")
        pending = set(self.observations) - self.cleared
        while pending:
            plan = self._rolling_plan(pending, terminal=None, phase="integrated_tail")
            first = plan.decisions[0]
            self._service_channel(first.channel, first.candidate_index)
            pending = set(self.observations) - self.cleared
        return RunSummary(
            strategy=self.name,
            completed_normally=True,
            stop_reason="联合覆盖证书成立且全部清除",
            stations_visited=self.stations_visited,
            discovered_channels=tuple(sorted(self.discovered)),
            cleared_channels=tuple(sorted(self.cleared)),
            pending_channels=(),
            measure_count=self.state.measure_count,
            clear_attempt_count=self.state.clear_attempt_count,
            clear_success_count=self.state.clear_success_count,
            no_signal_measure_count=self.no_signal_measure_count,
            shared_observation_count=self.shared_observation_count,
            search_certified=True,
            certified_absent_channels=tuple(sorted(self.certified_absent_channels)),
            final_virtual_time_s=self.state.virtual_time_s,
            piggyback_observation_count=self.piggyback_observation_count,
            reactive_refinement_count=self.reactive_refinement_count,
        )


class IntegratedGain20Policy(IntegratedAdaptiveCoveragePolicy):
    name = "scratch_integrated_gain20"
    adaptive_gain_fraction = 0.20
    maximum_adaptive_scans = 3


class IntegratedGain10Policy(IntegratedAdaptiveCoveragePolicy):
    name = "scratch_integrated_gain10"
    adaptive_gain_fraction = 0.10
    maximum_adaptive_scans = 5


class IntegratedImmediateCertificatePolicy(IntegratedAdaptiveCoveragePolicy):
    """Scan a clear stop only when that one scan completes the certificate.

    Round 7 showed that partial coverage gains are usually sunk measurement
    cost: they rarely remove a fixed station.  This variant therefore pays for
    an all-channel scan only if the current service stop covers every cell that
    is still uncovered by already scanned points.
    """

    name = "scratch_integrated_immediate_certificate"
    maximum_adaptive_scans = 1

    def _scan_unknown_service_stop(self) -> None:
        if len(self.discovered) == MAX_SOURCE_COUNT or self.adaptive_scan_count >= 1:
            return
        point = self.state.position.copy()
        if any(float(np.linalg.norm(point - old)) <= 1.0e-7 for old in self._adaptive_scan_points):
            return
        covered = self._cells_covered_by(point)
        if not bool(np.all(self._adaptive_covered | covered)):
            return
        unknown = [channel for channel in CHANNELS if channel not in self.discovered]
        if self.state.receiver_channel in unknown:
            unknown.remove(self.state.receiver_channel)
            unknown.insert(0, self.state.receiver_channel)
        for channel in unknown:
            response = self._measure(point, channel)
            result = response["measure_result"]
            if result == "direction":
                self._record_direction(channel, point, response)
            elif result == "near":
                self._clear_near(channel, point)
            else:
                self._record_no_signal(channel, point, phase="integrated_immediate_certificate")
            if len(self.discovered) == MAX_SOURCE_COUNT:
                break
        self.adaptive_scan_count += 1
        self._register_coverage_point(point)
        if len(self.discovered) < MAX_SOURCE_COUNT:
            self.search_certified = True
            self.certified_absent_channels = set(CHANNELS) - self.discovered


class AdaptiveCascadePolicy(SearchFirstPolicy):
    """Let useful target-service stops replace part of the fixed search skeleton.

    Every point admitted as an adaptive coverage point scans *all currently
    undiscovered channels*.  Hence a channel still undiscovered at termination
    has a no-signal observation at every admitted point.  A conservative square
    certificate declares coverage only when every square intersecting the target
    disk lies wholly inside at least one 1000 m scan disk.  The analytic seven
    point layout remains a fallback.
    """

    name = "scratch_adaptive_cascade"
    adaptive_gain_fraction = 0.05
    speculative_shared = False
    scan_after_localization = True

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["opportunistic_during_search"] = False
        kwargs["coverage_layout"] = COVERAGE_LAYOUT_HEPTAGON
        super().__init__(*args, **kwargs)
        self._adaptive_covered = np.zeros(len(ADAPTIVE_CELL_CORNERS), dtype=bool)
        self._adaptive_scan_points: list[np.ndarray] = []
        self._fallback_remaining = set(range(len(self.coverage_points)))
        self.adaptive_scan_count = 0
        self.fallback_scan_count = 0

    @staticmethod
    def _cells_covered_by(point: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(ADAPTIVE_CELL_CORNERS - point, axis=2)
        return np.max(distances, axis=1) < MIN_RECEPTION_RADIUS_M - 1.0e-7

    def _coverage_gain(self, point: np.ndarray) -> tuple[np.ndarray, int]:
        covered = self._cells_covered_by(np.asarray(point, dtype=float))
        gain = int(np.count_nonzero(covered & ~self._adaptive_covered))
        return covered, gain

    def _scan_unknown_at(
        self,
        point: np.ndarray,
        *,
        phase: str,
        fallback_index: int | None = None,
    ) -> None:
        point = np.asarray(point, dtype=float)
        unknown = [channel for channel in CHANNELS if channel not in self.discovered]
        if self.state.receiver_channel in unknown:
            unknown.remove(self.state.receiver_channel)
            unknown.insert(0, self.state.receiver_channel)
        for channel in unknown:
            response = self._measure(point, channel)
            result = response["measure_result"]
            if result == "direction":
                self._record_direction(channel, point, response)
            elif result == "near":
                self._clear_near(channel, point)
            else:
                self._record_no_signal(
                    channel,
                    point,
                    phase=phase,
                    station_index=fallback_index,
                )
            if len(self.discovered) == MAX_SOURCE_COUNT:
                break

        covered, _ = self._coverage_gain(point)
        self._adaptive_covered |= covered
        self._adaptive_scan_points.append(point.copy())
        if fallback_index is None:
            self.adaptive_scan_count += 1
        else:
            self.fallback_scan_count += 1
            self._fallback_remaining.discard(fallback_index)
            self.stations_visited += 1

        # A guaranteed extra bearing costs no movement and can remove an entire
        # dedicated second-observation trip later.
        self._collect_shared_at_adaptive_point(point)

    def _collect_shared_at_adaptive_point(self, point: np.ndarray) -> None:
        candidates = []
        for channel in sorted(self.observations):
            if channel in self.cleared or len(self.observations[channel]) != 1:
                continue
            first = self.observations[channel][0]
            if float(np.linalg.norm(point - first.point)) <= 1.0e-7:
                continue
            if guaranteed_shared_observation(first, point):
                candidates.append(channel)
            elif self.speculative_shared:
                # Experimental zero-movement attempt.  Failure is retained as
                # valid negative evidence; it never removes the safe fallback.
                displacement = point - first.point
                angle = math.radians(first.bearing_deg)
                forward = float(np.dot(displacement, [math.cos(angle), math.sin(angle)]))
                lateral = float(np.dot(displacement, [-math.sin(angle), math.cos(angle)]))
                separation = math.degrees(math.atan2(abs(lateral), forward))
                proxy_target = first.point + 855.2631578947 * np.asarray(
                    [math.cos(angle), math.sin(angle)]
                )
                if 5.0 <= separation <= 80.0 and float(np.linalg.norm(point - proxy_target)) <= 900.0:
                    candidates.append(channel)
        if self.state.receiver_channel in candidates:
            candidates.remove(self.state.receiver_channel)
            candidates.insert(0, self.state.receiver_channel)
        for channel in candidates:
            response = self._measure(point, channel)
            result = response["measure_result"]
            if result == "direction":
                self._record_direction(channel, point, response)
                self.shared_observation_count += 1
            elif result == "near":
                self._clear_near(channel, point)
            else:
                self._record_no_signal(channel, point, phase="adaptive_shared")

    def _admit_service_stop(self) -> None:
        if len(self.discovered) == MAX_SOURCE_COUNT:
            return
        point = self.state.position.copy()
        if any(float(np.linalg.norm(point - old)) <= 1.0e-7 for old in self._adaptive_scan_points):
            return
        _, gain = self._coverage_gain(point)
        uncovered = int(np.count_nonzero(~self._adaptive_covered))
        required = max(1, int(math.ceil(self.adaptive_gain_fraction * uncovered)))
        if gain >= required:
            self._scan_unknown_at(point, phase="adaptive_service_stop")

    def _choose_fallback(self) -> int:
        unknown_count = len(set(CHANNELS) - self.discovered)
        best: tuple[float, float, int] | None = None
        for index in sorted(self._fallback_remaining):
            point = self.coverage_points[index]
            _, gain = self._coverage_gain(point)
            cost_s = self._move_time(point) + max(1, unknown_count) * 6.0
            score = gain / max(cost_s, 1.0e-9)
            candidate = (score, -self._move_time(point), -index)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise PolicyError("自适应覆盖仍未认证，但七点兜底已经耗尽")
        return -best[2]

    def _update_adaptive_certificate(self) -> None:
        if len(self.discovered) == MAX_SOURCE_COUNT:
            self.search_certified = True
            self.certified_absent_channels = set()
            return
        if bool(np.all(self._adaptive_covered)):
            self.search_certified = True
            self.certified_absent_channels = set(CHANNELS) - self.discovered
            return
        if not self._fallback_remaining:
            # The seven analytic heptagon disks cover D.  Every still unknown
            # channel was scanned at every fallback station.
            required = set(range(len(self.coverage_points)))
            undiscovered = set(CHANNELS) - self.discovered
            certified = {
                channel
                for channel in undiscovered
                if self.coverage_no_signal_indices[channel] == required
            }
            if certified == undiscovered:
                self.search_certified = True
                self.certified_absent_channels = certified

    def run(self) -> RunSummary:
        self._scan_origin_information()
        # The origin scan reaches every channel unless the count-16 certificate
        # has already ended the search.
        origin = np.zeros(2, dtype=float)
        self._adaptive_covered |= self._cells_covered_by(origin)
        self._adaptive_scan_points.append(origin)
        self._update_adaptive_certificate()

        while not self.search_certified:
            pending = set(self.observations) - self.cleared
            if pending:
                plan = self._rolling_plan(pending, terminal=None, phase="adaptive_cascade")
                first = plan.decisions[0]
                cleared_before = len(self.cleared)
                self._service_channel(first.channel, first.candidate_index)
                if self.scan_after_localization or len(self.cleared) > cleared_before:
                    self._admit_service_stop()
            else:
                index = self._choose_fallback()
                point = self.coverage_points[index]
                self._scan_unknown_at(
                    point,
                    phase="adaptive_fallback",
                    fallback_index=index,
                )
            self._update_adaptive_certificate()

        if len(self.discovered) < MIN_SOURCE_COUNT:
            raise PolicyError("自适应覆盖完成但发现目标数低于题面下限")
        pending = set(self.observations) - self.cleared
        while pending:
            plan = self._rolling_plan(pending, terminal=None, phase="adaptive_tail")
            first = plan.decisions[0]
            self._service_channel(first.channel, first.candidate_index)
            pending = set(self.observations) - self.cleared

        return RunSummary(
            strategy=self.name,
            completed_normally=True,
            stop_reason="自适应覆盖证书成立且全部目标已清除",
            stations_visited=self.stations_visited,
            discovered_channels=tuple(sorted(self.discovered)),
            cleared_channels=tuple(sorted(self.cleared)),
            pending_channels=(),
            measure_count=self.state.measure_count,
            clear_attempt_count=self.state.clear_attempt_count,
            clear_success_count=self.state.clear_success_count,
            no_signal_measure_count=self.no_signal_measure_count,
            shared_observation_count=self.shared_observation_count,
            search_certified=True,
            certified_absent_channels=tuple(sorted(self.certified_absent_channels)),
            final_virtual_time_s=self.state.virtual_time_s,
            piggyback_observation_count=self.piggyback_observation_count,
            reactive_refinement_count=self.reactive_refinement_count,
        )


class AdaptiveGain10Policy(AdaptiveCascadePolicy):
    name = "scratch_adaptive_cascade_gain10"
    adaptive_gain_fraction = 0.10


class AdaptiveGain30Policy(AdaptiveCascadePolicy):
    name = "scratch_adaptive_cascade_gain30"
    adaptive_gain_fraction = 0.30


class AdaptiveGain20Policy(AdaptiveCascadePolicy):
    name = "scratch_adaptive_cascade_gain20"
    adaptive_gain_fraction = 0.20


class AdaptiveGain05Policy(AdaptiveCascadePolicy):
    name = "scratch_adaptive_cascade_gain05"
    adaptive_gain_fraction = 0.05


class AdaptiveGain02Policy(AdaptiveCascadePolicy):
    name = "scratch_adaptive_cascade_gain02"
    adaptive_gain_fraction = 0.02


class AdaptiveGain02SpecPolicy(AdaptiveGain02Policy):
    name = "scratch_adaptive_cascade_gain02_spec"
    speculative_shared = True


class AdaptiveClearGain20Policy(AdaptiveGain20Policy):
    name = "scratch_adaptive_clear_stops_gain20"
    scan_after_localization = False


class AdaptiveClearGain10Policy(AdaptiveGain10Policy):
    name = "scratch_adaptive_clear_stops_gain10"
    scan_after_localization = False


VARIANTS: dict[str, tuple[type[SearchFirstPolicy], dict[str, Any]]] = {
    "baseline": (SearchFirstPolicy, {}),
    "no_opportunity": (SearchFirstPolicy, {"opportunistic_during_search": False}),
    "no_origin": (SearchFirstPolicy, {"origin_information_scan": False}),
    "no_origin_no_opportunity": (
        SearchFirstPolicy,
        {"origin_information_scan": False, "opportunistic_during_search": False},
    ),
    "center_hex": (SearchFirstPolicy, {"coverage_layout": "center_hex_1150"}),
    "gate100": (Gate100Policy, {}),
    "gate250": (Gate250Policy, {}),
    "gate500": (Gate500Policy, {}),
    "one_per_edge": (OnePerEdgePolicy, {}),
    "full_proxy250": (FullProxy250Policy, {}),
    "full_proxy500": (FullProxy500Policy, {}),
    "full_proxy1000": (FullProxy1000Policy, {}),
    "adaptive_orientation": (AdaptiveOrientationPolicy, {}),
    "dynamic_coverage_order": (DynamicCoverageOrderPolicy, {}),
    "adaptive_subset": (AdaptiveSubsetCoveragePolicy, {}),
    "joint_scale125": (JointScale125Policy, {}),
    "ordered_joint125": (OrderedJointScale125Policy, {}),
    "integrated_gain20": (IntegratedGain20Policy, {}),
    "integrated_gain10": (IntegratedGain10Policy, {}),
    "integrated_immediate": (IntegratedImmediateCertificatePolicy, {}),
    "adaptive_gain30": (AdaptiveGain30Policy, {}),
    "adaptive_gain20": (AdaptiveGain20Policy, {}),
    "adaptive_gain10": (AdaptiveGain10Policy, {}),
    "adaptive_gain05": (AdaptiveGain05Policy, {}),
    "adaptive_gain02": (AdaptiveGain02Policy, {}),
    "adaptive_gain02_spec": (AdaptiveGain02SpecPolicy, {}),
    "adaptive_clear_gain20": (AdaptiveClearGain20Policy, {}),
    "adaptive_clear_gain10": (AdaptiveClearGain10Policy, {}),
}


def run_one(task: tuple[str, tuple[int, list[tuple[int, float, float, float]], int]]) -> dict[str, Any]:
    # Exact same recurrence as the production scheduler, compiled only to make
    # the requested 10 x 500 offline experiment practical.
    policy_module.optimal_candidate_route = fast_optimal_candidate_route
    variant, world = task
    index, packed, error_seed = world
    sources = [
        HiddenSource(channel, np.asarray([x, y]), reception_radius)
        for channel, x, y, reception_radius in packed
    ]
    client = AuditClient(sources, error_seed)
    policy_type, kwargs = VARIANTS[variant]
    started = time.perf_counter()
    try:
        policy = policy_type(
            client,  # type: ignore[arg-type]
            NullLogger(),  # type: ignore[arg-type]
            {"remaining_real_duration_s": 1200, "max_virtual_duration_s": 360000},
            **kwargs,
        )
        summary = policy.run()
        success = bool(
            summary.search_certified
            and summary.completed_normally
            and client.cleared_count == len(sources)
        )
        return {
            "scenario": index,
            "success": success,
            "source_count": len(sources),
            "time_s": summary.final_virtual_time_s,
            "time_per_source_s": summary.final_virtual_time_s / len(sources),
            "movement_m": client.movement_m,
            "measure_count": summary.measure_count,
            "clear_count": summary.clear_attempt_count,
            "shared_count": summary.shared_observation_count,
            "piggyback_count": summary.piggyback_observation_count,
            "reactive_count": summary.reactive_refinement_count,
            "stations_visited": summary.stations_visited,
            "adaptive_scan_count": int(getattr(policy, "adaptive_scan_count", 0)),
            "fallback_scan_count": int(getattr(policy, "fallback_scan_count", 0)),
            "wall_s": time.perf_counter() - started,
            "error": None,
        }
    except Exception as error:
        return {
            "scenario": index,
            "success": False,
            "source_count": len(sources),
            "wall_s": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    good = [row for row in rows if row["success"]]
    if not good:
        return {"successes": 0, "scenarios": len(rows), "failures": rows}
    values = lambda key: np.asarray([row[key] for row in good], dtype=float)
    times = values("time_s")
    per_source = values("time_per_source_s")
    worst = sorted(good, key=lambda row: row["time_per_source_s"], reverse=True)[:10]
    return {
        "successes": len(good),
        "scenarios": len(rows),
        "mean_total_s": float(np.mean(times)),
        "median_total_s": float(np.median(times)),
        "p90_total_s": float(np.percentile(times, 90)),
        "p95_total_s": float(np.percentile(times, 95)),
        "max_total_s": float(np.max(times)),
        "mean_per_source_s": float(np.mean(per_source)),
        "median_per_source_s": float(np.median(per_source)),
        "p90_per_source_s": float(np.percentile(per_source, 90)),
        "mean_movement_m": float(np.mean(values("movement_m"))),
        "mean_measure_count": float(np.mean(values("measure_count"))),
        "mean_clear_count": float(np.mean(values("clear_count"))),
        "mean_shared_count": float(np.mean(values("shared_count"))),
        "mean_piggyback_count": float(np.mean(values("piggyback_count"))),
        "mean_reactive_count": float(np.mean(values("reactive_count"))),
        "mean_stations_visited": float(np.mean(values("stations_visited"))),
        "mean_adaptive_scan_count": float(np.mean(values("adaptive_scan_count"))),
        "mean_fallback_scan_count": float(np.mean(values("fallback_scan_count"))),
        "worst_10": worst,
        "failures": [row for row in rows if not row["success"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--scenarios", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worlds = make_worlds(args.scenarios, args.seed)
    tasks = [(args.variant, world) for world in worlds]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run_one, tasks, chunksize=1))
    result = {
        "mode": "offline_only_no_network",
        "variant": args.variant,
        "seed": args.seed,
        "scenarios": args.scenarios,
        "workers": args.workers,
        "elapsed_wall_s": time.perf_counter() - started,
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))
    if result["summary"]["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
