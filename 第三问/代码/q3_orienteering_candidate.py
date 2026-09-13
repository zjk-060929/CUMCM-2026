"""Offline-only Q3 joint search/service orienteering experiment.

This file deliberately leaves the submitted policy untouched.  The candidate
reorders only actions supported by observations already returned by the mock
interface: unvisited guaranteed-search stations and pending channel-service
actions.  Hidden source coordinates are used solely by ``OfflineClient`` to
produce feedback and are never inspected by the policy.

The main idea is a rolling, directed-arc tour.  A one-bearing task has two safe
second-observation entries and an evidence-only proxy exit (the centroid of its
current feasible region); a localized task and a search station have identical
entry/exit points.  Multi-start greedy construction plus deterministic 2-opt
chooses the next action.  Feedback is then incorporated and the tour is rebuilt.
All seven search stations are still visited unless 16 distinct sources have
already been found, so the original strict search certificate is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
for directory in (ROOT, CODE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from geometry import (  # noqa: E402
    guaranteed_shared_observation,
    localization_polygon,
    minimum_enclosing_circle,
    observation_feasible_polygon,
    second_detection_candidates,
)
from offline_validation import HiddenSource, NullLogger, generate_sources  # noqa: E402
from policy import (  # noqa: E402
    CHANNELS,
    MAX_SOURCE_COUNT,
    MIN_SOURCE_COUNT,
    PolicyError,
    RunSummary,
    SearchFirstPolicy,
)
from scratch_q3_global_rolling import AuditClient  # noqa: E402
from scheduler import current_edge_is_cheapest  # noqa: E402


@dataclass(frozen=True)
class ArcItem:
    kind: str
    identifier: int
    entries: np.ndarray
    exits: np.ndarray


@dataclass(frozen=True)
class ArcTour:
    order: tuple[int, ...]
    sides: tuple[int, ...]
    distance_m: float
    objective_m: float


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    """Area centroid, with a stable fallback for degenerate polygons."""

    vertices = np.asarray(polygon, dtype=float)
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    area2 = float(np.sum(cross))
    if abs(area2) <= 1.0e-12:
        return np.mean(vertices, axis=0)
    return np.asarray(
        [
            np.sum((vertices[:, 0] + following[:, 0]) * cross),
            np.sum((vertices[:, 1] + following[:, 1]) * cross),
        ],
        dtype=float,
    ) / (3.0 * area2)


def best_sides_for_order(
    start: np.ndarray,
    items: list[ArcItem],
    order: list[int] | tuple[int, ...],
) -> tuple[float, tuple[int, ...]]:
    """Exact two-choice DP for one fixed item order."""

    if not order:
        return 0.0, ()
    first = items[order[0]]
    costs = np.linalg.norm(first.entries - start, axis=1) + np.linalg.norm(
        first.exits - first.entries, axis=1
    )
    parents: list[np.ndarray] = []
    for previous_index, current_index in zip(order, order[1:]):
        previous = items[previous_index]
        current = items[current_index]
        transition = np.linalg.norm(
            previous.exits[:, None, :] - current.entries[None, :, :], axis=2
        )
        internal = np.linalg.norm(current.exits - current.entries, axis=1)
        alternatives = costs[:, None] + transition + internal[None, :]
        parent = np.argmin(alternatives, axis=0).astype(np.int8)
        costs = alternatives[parent, np.arange(len(current.entries))]
        parents.append(parent)
    side = int(np.argmin(costs))
    sides = [side]
    for parent in reversed(parents):
        side = int(parent[side])
        sides.append(side)
    sides.reverse()
    return float(np.min(costs)), tuple(sides)


def route_delay_penalty(
    items: list[ArcItem], order: list[int] | tuple[int, ...], station_delay_m: float
) -> float:
    """Small information-delay cost: postpone mandatory scans only deliberately."""

    pending_services = 0
    penalty = 0.0
    for item_index in order:
        if items[item_index].kind == "service":
            pending_services += 1
        else:
            penalty += station_delay_m * pending_services
    return penalty


def rolling_orienteering_tour(
    start: np.ndarray,
    items: list[ArcItem],
    *,
    station_delay_m: float,
    two_opt_passes: int = 2,
) -> ArcTour:
    """Deterministic multi-start greedy tour followed by directed-arc 2-opt."""

    if not items:
        return ArcTour((), (), 0.0, 0.0)

    candidate_orders: list[list[int]] = []
    # Every item is tried as a first action.  Subsequent nearest choice uses the
    # best entry/exit arc available for that item; exact side choices are solved
    # after the complete order is built.
    for first_index in range(len(items)):
        order = [first_index]
        remaining = set(range(len(items))) - {first_index}
        _, first_sides = best_sides_for_order(start, items, order)
        cursor = items[first_index].exits[first_sides[0]]
        while remaining:
            choices: list[tuple[float, int, int]] = []
            for item_index in sorted(remaining):
                item = items[item_index]
                arc = np.linalg.norm(item.entries - cursor, axis=1) + np.linalg.norm(
                    item.exits - item.entries, axis=1
                )
                side = int(np.argmin(arc))
                choices.append((float(arc[side]), item_index, side))
            _, chosen, side = min(choices)
            order.append(chosen)
            remaining.remove(chosen)
            cursor = items[chosen].exits[side]
        candidate_orders.append(order)

    best_order: list[int] | None = None
    best_sides: tuple[int, ...] = ()
    best_distance = float("inf")
    best_objective = float("inf")
    for order in candidate_orders:
        distance, sides = best_sides_for_order(start, items, order)
        objective = distance + route_delay_penalty(items, order, station_delay_m)
        # Best-improvement pair swaps provide a compact two-step/local beam.
        for _ in range(two_opt_passes):
            improved: tuple[float, float, list[int], tuple[int, ...]] | None = None
            for left in range(len(order) - 1):
                for right in range(left + 1, len(order)):
                    trial = order.copy()
                    trial[left], trial[right] = trial[right], trial[left]
                    trial_distance, trial_sides = best_sides_for_order(start, items, trial)
                    trial_objective = trial_distance + route_delay_penalty(
                        items, trial, station_delay_m
                    )
                    if trial_objective + 1.0e-8 < objective:
                        record = (trial_objective, trial_distance, trial, trial_sides)
                        if improved is None or record[0] < improved[0] - 1.0e-8:
                            improved = record
            if improved is None:
                break
            objective, distance, order, sides = improved
        key = (objective, distance, tuple(order), sides)
        best_key = (
            best_objective,
            best_distance,
            tuple(best_order) if best_order is not None else (),
            best_sides,
        )
        if best_order is None or key < best_key:
            best_objective, best_distance = objective, distance
            best_order, best_sides = order.copy(), sides
    assert best_order is not None
    return ArcTour(tuple(best_order), best_sides, best_distance, best_objective)


def ordered_station_insertion_tour(
    start: np.ndarray,
    station_items: list[ArcItem],
    service_items: list[ArcItem],
) -> tuple[list[ArcItem], tuple[int, ...], float]:
    """Greedily insert service arcs while preserving the station subsequence."""

    route = list(station_items)
    remaining = list(service_items)
    while remaining:
        best: tuple[float, int, int, tuple[int, ...]] | None = None
        for service_index, service in enumerate(remaining):
            for position in range(len(route) + 1):
                trial = [*route[:position], service, *route[position:]]
                distance, sides = best_sides_for_order(
                    start, trial, tuple(range(len(trial)))
                )
                record = (distance, service_index, position, sides)
                if best is None or record < best:
                    best = record
        assert best is not None
        _, service_index, position, _ = best
        route.insert(position, remaining.pop(service_index))
    distance, sides = best_sides_for_order(start, route, tuple(range(len(route))))
    return route, sides, distance


class JointOrienteeringPolicy(SearchFirstPolicy):
    """Globally interleave mandatory search stations and observable tasks."""

    name = "scratch_joint_orienteering_delay80"
    station_delay_m = 80.0
    use_proxy_exit = True
    proxy_scale = 1.0
    two_opt_passes = 2

    def _scan_origin_information(self) -> None:
        super()._scan_origin_information()
        self.origin_discovered_count = len(self.discovered)

    def _service_arc(self, channel: int) -> ArcItem:
        observations = self.observations[channel]
        if len(observations) >= 2:
            polygon = localization_polygon(observations, self.no_signal_points[channel])
            center, _ = minimum_enclosing_circle(polygon)
            values = np.vstack([center, center])
            return ArcItem("service", channel, values, values.copy())
        entries = np.asarray(second_detection_candidates(observations[0]), dtype=float)
        if self.use_proxy_exit:
            feasible = observation_feasible_polygon(
                observations, self.no_signal_points[channel]
            )
            centroid = polygon_centroid(feasible)
            proxy = observations[0].point + self.proxy_scale * (
                centroid - observations[0].point
            )
            exits = np.vstack([proxy, proxy])
        else:
            exits = entries.copy()
        return ArcItem("service", channel, entries, exits)

    def _planning_items(self, remaining_stations: set[int]) -> list[ArcItem]:
        station_items = [
            ArcItem(
                "station",
                index,
                np.vstack([self.coverage_points[index], self.coverage_points[index]]),
                np.vstack([self.coverage_points[index], self.coverage_points[index]]),
            )
            for index in sorted(remaining_stations)
        ]
        pending = set(self.observations) - self.cleared
        # A future mandatory station that is a guaranteed second observation is
        # strictly preferable to paying for a separate safe second point now.
        deferred = {
            channel
            for channel in pending
            if len(self.observations[channel]) == 1
            and any(
                guaranteed_shared_observation(
                    self.observations[channel][0], self.coverage_points[index]
                )
                for index in remaining_stations
            )
        }
        services = [self._service_arc(channel) for channel in sorted(pending - deferred)]
        return [*station_items, *services]

    def _visit_station(self, station_index: int) -> None:
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
                    phase="joint_orienteering_station",
                    station_index=station_index,
                )
            if len(self.discovered) == MAX_SOURCE_COUNT:
                break
        self.stations_visited += 1
        if self.shared_observations:
            self._collect_shared_observations(station, station_index)

    def _scan_coverage_stations(self) -> None:
        remaining_stations = set(range(len(self.coverage_points)))
        while remaining_stations and len(self.discovered) < MAX_SOURCE_COUNT:
            items = self._planning_items(remaining_stations)
            tour = rolling_orienteering_tour(
                self.state.position,
                items,
                station_delay_m=self.station_delay_m,
                two_opt_passes=self.two_opt_passes,
            )
            selected = items[tour.order[0]]
            side = tour.sides[0]
            if selected.kind == "station":
                self._visit_station(selected.identifier)
                remaining_stations.remove(selected.identifier)
            else:
                self._service_channel(selected.identifier, preferred_candidate_index=side)
        if len(self.discovered) == MAX_SOURCE_COUNT:
            self._update_search_certificate()
            return
        self._update_search_certificate()


class JointDistanceOnlyPolicy(JointOrienteeringPolicy):
    name = "scratch_joint_orienteering_distance_only"
    station_delay_m = 0.0


class JointReward50Policy(JointOrienteeringPolicy):
    name = "scratch_joint_orienteering_service_reward50"
    station_delay_m = -50.0


class JointReward100Policy(JointOrienteeringPolicy):
    name = "scratch_joint_orienteering_service_reward100"
    station_delay_m = -100.0


class JointScale075Policy(JointDistanceOnlyPolicy):
    name = "scratch_joint_orienteering_proxy_scale075"
    proxy_scale = 0.75


class JointScale125Policy(JointDistanceOnlyPolicy):
    name = "scratch_joint_orienteering_proxy_scale125"
    proxy_scale = 1.25


class JointScale150Policy(JointDistanceOnlyPolicy):
    name = "scratch_joint_orienteering_proxy_scale150"
    proxy_scale = 1.50


class JointScale175Policy(JointDistanceOnlyPolicy):
    name = "scratch_joint_orienteering_proxy_scale175"
    proxy_scale = 1.75


class JointThreePassPolicy(JointDistanceOnlyPolicy):
    name = "scratch_joint_orienteering_three_pass"
    two_opt_passes = 3


class JointDelay200Policy(JointOrienteeringPolicy):
    name = "scratch_joint_orienteering_delay200"
    station_delay_m = 200.0


class JointPointPolicy(JointOrienteeringPolicy):
    name = "scratch_joint_orienteering_point_no_proxy"
    station_delay_m = 80.0
    use_proxy_exit = False


class OrderedJointPolicy(JointOrienteeringPolicy):
    """Joint insertion planning with the proven cyclic station order fixed."""

    name = "scratch_ordered_joint_proxy100"

    def _scan_coverage_stations(self) -> None:
        remaining = list(range(len(self.coverage_points)))
        while remaining and len(self.discovered) < MAX_SOURCE_COUNT:
            all_items = self._planning_items(set(remaining))
            stations_by_id = {
                item.identifier: item for item in all_items if item.kind == "station"
            }
            station_items = [stations_by_id[index] for index in remaining]
            service_items = [item for item in all_items if item.kind == "service"]
            route, sides, _ = ordered_station_insertion_tour(
                self.state.position, station_items, service_items
            )
            selected, side = route[0], sides[0]
            if selected.kind == "station":
                self._visit_station(selected.identifier)
                if selected.identifier != remaining[0]:
                    raise RuntimeError("ordered planner violated station precedence")
                remaining.pop(0)
            else:
                self._service_channel(selected.identifier, side)
        self._update_search_certificate()


class OrderedJointScale125Policy(OrderedJointPolicy):
    name = "scratch_ordered_joint_proxy125"
    proxy_scale = 1.25


class FrozenBatchPolicy(SearchFirstPolicy):
    """Freeze an edge's service window before leaving its search station.

    The baseline recomputes eligibility after every service.  Once the robot is
    off the search skeleton this can make progressively more remote tasks look
    cheapest on the *current* edge, creating an avalanche and a long return.
    Here assignment is evaluated once at the real skeleton vertex.  The entire
    assigned set is then served as one optimized batch and newly eligible tasks
    wait for the next genuine search edge (or the final exact tail tour).
    """

    name = "scratch_frozen_edge_batch"

    def _service_opportunities(self, station_index: int, stations: list[np.ndarray]) -> None:
        next_station = stations[station_index + 1]
        future_stations = stations[station_index + 1 :]
        future_edges = [
            (stations[index], stations[index + 1])
            for index in range(station_index + 1, len(stations) - 1)
        ]
        pending = set(self.observations) - self.cleared
        pending -= {
            channel
            for channel in pending
            if len(self.observations[channel]) == 1
            and any(
                guaranteed_shared_observation(
                    self.observations[channel][0], future_station
                )
                for future_station in future_stations
            )
        }
        batch = {
            channel
            for channel, candidates in self._candidate_map(pending).items()
            if current_edge_is_cheapest(
                self.state.position,
                next_station,
                future_edges,
                candidates,
            )[0]
        }
        while batch:
            batch &= set(self.observations) - self.cleared
            if not batch:
                return
            plan = self._rolling_plan(
                batch,
                terminal=next_station,
                phase=f"frozen_batch_edge_{station_index + 1}",
            )
            first = plan.decisions[0]
            self._service_channel(first.channel, first.candidate_index)


class SeededWindowPolicy(SearchFirstPolicy):
    """Keep the full initial edge batch, but gate path-dependent expansion."""

    name = "scratch_seeded_window_expand500"
    expansion_gate_m = 500.0

    def _eligible_on_edge(
        self,
        next_station: np.ndarray,
        future_stations: list[np.ndarray],
        future_edges: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[set[int], dict[int, float]]:
        pending = set(self.observations) - self.cleared
        pending -= {
            channel
            for channel in pending
            if len(self.observations[channel]) == 1
            and any(
                guaranteed_shared_observation(self.observations[channel][0], station)
                for station in future_stations
            )
        }
        eligible: set[int] = set()
        costs: dict[int, float] = {}
        for channel, candidates in self._candidate_map(pending).items():
            cheapest, current, _ = current_edge_is_cheapest(
                self.state.position, next_station, future_edges, candidates
            )
            if cheapest:
                eligible.add(channel)
                costs[channel] = current
        return eligible, costs

    def _service_opportunities(self, station_index: int, stations: list[np.ndarray]) -> None:
        next_station = stations[station_index + 1]
        future_stations = stations[station_index + 1 :]
        future_edges = [
            (stations[index], stations[index + 1])
            for index in range(station_index + 1, len(stations) - 1)
        ]
        batch, _ = self._eligible_on_edge(next_station, future_stations, future_edges)
        initial = True
        while batch:
            batch &= set(self.observations) - self.cleared
            if not batch:
                return
            plan = self._rolling_plan(batch, terminal=next_station, phase="seeded_window")
            first = plan.decisions[0]
            self._service_channel(first.channel, first.candidate_index)
            eligible, costs = self._eligible_on_edge(
                next_station, future_stations, future_edges
            )
            # Initial assignments are never thresholded.  Only the channels
            # pulled in by being off-skeleton must be cheap enough to expand the
            # current service window.
            additions = {
                channel
                for channel in eligible - batch
                if costs[channel] <= self.expansion_gate_m
            }
            batch |= additions
            initial = False


class SeededWindow250Policy(SeededWindowPolicy):
    name = "scratch_seeded_window_expand250"
    expansion_gate_m = 250.0


class SeededWindow1000Policy(SeededWindowPolicy):
    name = "scratch_seeded_window_expand1000"
    expansion_gate_m = 1000.0


class FrozenBatchArcPolicy(FrozenBatchPolicy):
    """Frozen service windows plus evidence-only directed service exits."""

    name = "scratch_frozen_edge_batch_arc"

    def _rolling_plan(self, channels: set[int], terminal: np.ndarray | None, phase: str):
        del phase
        items = [self._service_arc_for_batch(channel) for channel in sorted(channels)]
        tour = rolling_orienteering_tour(
            self.state.position,
            items,
            station_delay_m=0.0,
            two_opt_passes=1,
        )
        # SearchFirstPolicy needs only the ordered channel/candidate choices.
        from scheduler import RoutePlan, ServiceDecision

        decisions = tuple(
            ServiceDecision(
                items[item_index].identifier,
                side,
                items[item_index].entries[side].copy(),
            )
            for item_index, side in zip(tour.order, tour.sides)
        )
        return RoutePlan(decisions, tour.distance_m)

    def _service_arc_for_batch(self, channel: int) -> ArcItem:
        observations = self.observations[channel]
        if len(observations) >= 2:
            polygon = localization_polygon(observations, self.no_signal_points[channel])
            center, _ = minimum_enclosing_circle(polygon)
            values = np.vstack([center, center])
            return ArcItem("service", channel, values, values.copy())
        entries = np.asarray(second_detection_candidates(observations[0]), dtype=float)
        feasible = observation_feasible_polygon(observations, self.no_signal_points[channel])
        proxy = polygon_centroid(feasible)
        return ArcItem("service", channel, entries, np.vstack([proxy, proxy]))


class AdaptiveRingOrientationPolicy(SearchFirstPolicy):
    """Choose the cyclic start/direction from the origin scan evidence.

    All fourteen Hamiltonian paths around the same seven-point ring have the
    identical mandatory skeleton length.  Their endpoints and directed edges,
    however, have very different insertion cost for the observed sources.  We
    therefore score them with directed evidence-only service arcs and then run
    the unchanged proven opportunity policy on the best orientation.
    """

    name = "scratch_adaptive_ring_orientation_arc"
    proxy_exit = True

    def _origin_service_arcs(self) -> list[ArcItem]:
        result: list[ArcItem] = []
        for channel in sorted(set(self.observations) - self.cleared):
            observations = self.observations[channel]
            if len(observations) >= 2:
                polygon = localization_polygon(observations, self.no_signal_points[channel])
                center, _ = minimum_enclosing_circle(polygon)
                values = np.vstack([center, center])
                result.append(ArcItem("service", channel, values, values.copy()))
                continue
            entries = np.asarray(second_detection_candidates(observations[0]), dtype=float)
            if self.proxy_exit:
                feasible = observation_feasible_polygon(
                    observations, self.no_signal_points[channel]
                )
                proxy = polygon_centroid(feasible)
                exits = np.vstack([proxy, proxy])
            else:
                exits = entries.copy()
            result.append(ArcItem("service", channel, entries, exits))
        return result

    @staticmethod
    def _arc_insertion(start: np.ndarray, end: np.ndarray, item: ArcItem) -> float:
        direct = float(np.linalg.norm(end - start))
        values = (
            np.linalg.norm(item.entries - start, axis=1)
            + np.linalg.norm(item.exits - item.entries, axis=1)
            + np.linalg.norm(end - item.exits, axis=1)
            - direct
        )
        return float(np.min(values))

    @staticmethod
    def _arc_open_tail(start: np.ndarray, item: ArcItem) -> float:
        values = np.linalg.norm(item.entries - start, axis=1) + np.linalg.norm(
            item.exits - item.entries, axis=1
        )
        return float(np.min(values))

    def _choose_ring_orientation(self) -> list[np.ndarray]:
        original = [np.asarray(point, dtype=float).copy() for point in self.coverage_points]
        arcs = self._origin_service_arcs()
        choices: list[tuple[float, tuple[int, ...]]] = []
        for direction in (1, -1):
            for start in range(len(original)):
                indices = tuple(
                    (start + direction * offset) % len(original)
                    for offset in range(len(original))
                )
                points = [original[index] for index in indices]
                edges = list(zip(points, points[1:]))
                score = 0.0
                for item in arcs:
                    options = [
                        self._arc_insertion(before, after, item)
                        for before, after in edges
                    ]
                    options.append(self._arc_open_tail(points[-1], item))
                    score += min(options)
                choices.append((score, indices))
        _, best = min(choices)
        return [original[index] for index in best]

    def run(self) -> RunSummary:
        self._scan_origin_information()
        if len(self.discovered) < MAX_SOURCE_COUNT:
            self.coverage_points = self._choose_ring_orientation()
        self._scan_coverage_stations()
        if not self.search_certified:
            raise PolicyError("保证性搜索尚无完整排查证书")
        if len(self.discovered) < MIN_SOURCE_COUNT:
            raise PolicyError("保证性覆盖后的发现数低于题面下限")
        pending = set(self.observations) - self.cleared
        while pending:
            plan = self._rolling_plan(pending, terminal=None, phase="oriented_tail")
            decision = plan.decisions[0]
            self._service_channel(decision.channel, decision.candidate_index)
            pending = set(self.observations) - self.cleared
        return RunSummary(
            strategy=self.name,
            completed_normally=True,
            stop_reason="自适应环方向完成并取得严格搜索证书",
            stations_visited=self.stations_visited,
            discovered_channels=tuple(sorted(self.discovered)),
            cleared_channels=tuple(sorted(self.cleared)),
            pending_channels=(),
            measure_count=self.state.measure_count,
            clear_attempt_count=self.state.clear_attempt_count,
            clear_success_count=self.state.clear_success_count,
            no_signal_measure_count=self.no_signal_measure_count,
            shared_observation_count=self.shared_observation_count,
            search_certified=self.search_certified,
            certified_absent_channels=tuple(sorted(self.certified_absent_channels)),
            final_virtual_time_s=self.state.virtual_time_s,
            piggyback_observation_count=self.piggyback_observation_count,
            reactive_refinement_count=self.reactive_refinement_count,
        )


class AdaptiveRingPointPolicy(AdaptiveRingOrientationPolicy):
    name = "scratch_adaptive_ring_orientation_point"
    proxy_exit = False


class InstrumentedBaselinePolicy(SearchFirstPolicy):
    """Bit-for-bit baseline actions with one read-only observable diagnostic."""

    name = SearchFirstPolicy.name

    def _scan_origin_information(self) -> None:
        super()._scan_origin_information()
        self.origin_discovered_count = len(self.discovered)


POLICIES = {
    "baseline": InstrumentedBaselinePolicy,
    "window250": SeededWindow250Policy,
    "window500": SeededWindowPolicy,
    "window1000": SeededWindow1000Policy,
    "ring_arc": AdaptiveRingOrientationPolicy,
    "ring_point": AdaptiveRingPointPolicy,
    "frozen_batch": FrozenBatchPolicy,
    "frozen_arc": FrozenBatchArcPolicy,
    "joint0": JointDistanceOnlyPolicy,
    "joint_reward50": JointReward50Policy,
    "joint_reward100": JointReward100Policy,
    "joint_scale075": JointScale075Policy,
    "joint_scale125": JointScale125Policy,
    "joint_scale150": JointScale150Policy,
    "joint_scale175": JointScale175Policy,
    "joint_3pass": JointThreePassPolicy,
    "joint80": JointOrienteeringPolicy,
    "joint200": JointDelay200Policy,
    "joint_point": JointPointPolicy,
    "ordered_joint": OrderedJointPolicy,
    "ordered_joint125": OrderedJointScale125Policy,
}


def make_worlds(scenarios: int, seed: int):
    rng = np.random.default_rng(seed)
    worlds = []
    for index in range(scenarios):
        sources = generate_sources(rng)
        packed = [
            (source.channel, float(source.point[0]), float(source.point[1]), source.reception_radius_m)
            for source in sources
        ]
        worlds.append((index, packed, int(rng.integers(0, 2**31 - 1))))
    return worlds


def run_one(task: tuple[str, int, list[tuple[int, float, float, float]], int]):
    variant, index, packed, error_seed = task
    sources = [
        HiddenSource(channel, np.asarray([x, y]), reception)
        for channel, x, y, reception in packed
    ]
    client = AuditClient(sources, error_seed)
    policy = POLICIES[variant](
        client,
        NullLogger(),
        {"remaining_real_duration_s": 1200, "max_virtual_duration_s": 360000},
    )
    try:
        summary = policy.run()
        success = client.cleared_count == len(sources) and summary.search_certified
        error = None
    except Exception as exc:
        summary = policy.interrupted_summary(str(exc))
        success = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "variant": variant,
        "scenario": index,
        "sources": len(sources),
        "success": success,
        "error": error,
        "time_s": summary.final_virtual_time_s,
        "per_source_s": summary.final_virtual_time_s / len(sources),
        "movement_m": client.movement_m,
        "measure": summary.measure_count,
        "clear": summary.clear_attempt_count,
        "stations": summary.stations_visited,
        "origin_discovered": getattr(policy, "origin_discovered_count", None),
    }


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = np.asarray([row["time_s"] for row in rows])
    return {
        "success": f"{sum(row['success'] for row in rows)}/{len(rows)}",
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
        "p90_s": float(np.percentile(times, 90)),
        "mean_per_source_s": float(np.mean([row["per_source_s"] for row in rows])),
        "mean_movement_m": float(np.mean([row["movement_m"] for row in rows])),
        "mean_measure": float(np.mean([row["measure"] for row in rows])),
        "mean_clear": float(np.mean([row["clear"] for row in rows])),
        "mean_stations": float(np.mean([row["stations"] for row in rows])),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--variants", nargs="+", choices=sorted(POLICIES), default=list(POLICIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tracked = ["geometry.py", "policy.py", "scheduler.py", "offline_validation.py"]
    before = {name: sha256(CODE / name) for name in tracked}
    worlds = make_worlds(args.scenarios, args.seed)
    tasks = [
        (variant, index, packed, error_seed)
        for index, packed, error_seed in worlds
        for variant in args.variants
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        raw = list(pool.map(run_one, tasks, chunksize=1))
    grouped = {
        variant: sorted(
            (row for row in raw if row["variant"] == variant),
            key=lambda row: row["scenario"],
        )
        for variant in args.variants
    }
    comparisons: dict[str, Any] = {}
    if "baseline" in grouped:
        for variant, rows in grouped.items():
            if variant == "baseline":
                continue
            delta = np.asarray(
                [new["time_s"] - old["time_s"] for old, new in zip(grouped["baseline"], rows)]
            )
            comparisons[variant] = {
                "mean_delta_s": float(np.mean(delta)),
                "median_delta_s": float(np.median(delta)),
                "p10_delta_s": float(np.percentile(delta, 10)),
                "p90_delta_s": float(np.percentile(delta, 90)),
                "wins_ties_losses": [
                    int(np.sum(delta < -1.0e-8)),
                    int(np.sum(np.abs(delta) <= 1.0e-8)),
                    int(np.sum(delta > 1.0e-8)),
                ],
            }
    after = {name: sha256(CODE / name) for name in tracked}
    result = {
        "mode": "offline_only_no_network_paired",
        "seed": args.seed,
        "scenarios": args.scenarios,
        "elapsed_wall_s": time.perf_counter() - started,
        "authoritative_hashes_unchanged": before == after,
        "summaries": {variant: describe(rows) for variant, rows in grouped.items()},
        "paired_candidate_minus_baseline": comparisons,
        "rows": grouped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))
    if any(not row["success"] for row in raw):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
