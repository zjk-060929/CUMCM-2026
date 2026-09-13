"""问题3的联合滚动有向弧策略。

本模块只使用已经由接口返回的测量信息。七个保证性搜索站与待服务频道
共同参与路线规划；规划器每次只执行首个动作，收到反馈后立即重算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import (
    guaranteed_shared_observation,
    localization_polygon,
    minimum_enclosing_circle,
    observation_feasible_polygon,
    second_detection_candidates,
)
from policy import CHANNELS, MAX_SOURCE_COUNT, SearchFirstPolicy


@dataclass(frozen=True)
class ArcItem:
    """一个动作的两个可选入口以及相应预测出口。"""

    kind: str
    identifier: int
    entries: np.ndarray
    exits: np.ndarray


@dataclass(frozen=True)
class ArcTour:
    """有向弧路线及其二选一结果。"""

    order: tuple[int, ...]
    sides: tuple[int, ...]
    distance_m: float


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    """计算可行域面积质心；退化时稳定地回退到顶点均值。"""

    vertices = np.asarray(polygon, dtype=float)
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    area_twice = float(np.sum(cross))
    if abs(area_twice) <= 1.0e-12:
        return np.mean(vertices, axis=0)
    return np.asarray(
        [
            np.sum((vertices[:, 0] + following[:, 0]) * cross),
            np.sum((vertices[:, 1] + following[:, 1]) * cross),
        ],
        dtype=float,
    ) / (3.0 * area_twice)


def best_sides_for_order(
    start: np.ndarray,
    items: list[ArcItem],
    order: list[int] | tuple[int, ...],
) -> tuple[float, tuple[int, ...]]:
    """给定访问顺序后，用二状态动态规划精确选择每条弧的入口。"""

    if not order:
        return 0.0, ()
    first = items[order[0]]
    costs = np.linalg.norm(first.entries - start, axis=1) + np.linalg.norm(
        first.exits - first.entries,
        axis=1,
    )
    parents: list[np.ndarray] = []
    for previous_index, current_index in zip(order, order[1:]):
        previous = items[previous_index]
        current = items[current_index]
        transitions = np.linalg.norm(
            previous.exits[:, None, :] - current.entries[None, :, :],
            axis=2,
        )
        internal = np.linalg.norm(current.exits - current.entries, axis=1)
        alternatives = costs[:, None] + transitions + internal[None, :]
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


def rolling_orienteering_tour(
    start: np.ndarray,
    items: list[ArcItem],
    *,
    two_opt_passes: int = 2,
) -> ArcTour:
    """多起点最近弧构造，并以成对交换作确定性局部改进。"""

    if not items:
        return ArcTour((), (), 0.0)
    initial_orders: list[list[int]] = []
    for first_index in range(len(items)):
        order = [first_index]
        remaining = set(range(len(items))) - {first_index}
        _, first_sides = best_sides_for_order(start, items, order)
        cursor = items[first_index].exits[first_sides[0]]
        while remaining:
            choices: list[tuple[float, int, int]] = []
            for item_index in sorted(remaining):
                item = items[item_index]
                arc_costs = np.linalg.norm(item.entries - cursor, axis=1) + np.linalg.norm(
                    item.exits - item.entries,
                    axis=1,
                )
                side = int(np.argmin(arc_costs))
                choices.append((float(arc_costs[side]), item_index, side))
            _, chosen, side = min(choices)
            order.append(chosen)
            remaining.remove(chosen)
            cursor = items[chosen].exits[side]
        initial_orders.append(order)

    best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    for initial in initial_orders:
        order = initial.copy()
        distance, sides = best_sides_for_order(start, items, order)
        for _ in range(two_opt_passes):
            improvement: tuple[float, list[int], tuple[int, ...]] | None = None
            for left in range(len(order) - 1):
                for right in range(left + 1, len(order)):
                    trial = order.copy()
                    trial[left], trial[right] = trial[right], trial[left]
                    trial_distance, trial_sides = best_sides_for_order(start, items, trial)
                    candidate = (trial_distance, trial, trial_sides)
                    if trial_distance + 1.0e-8 < distance and (
                        improvement is None or candidate[0] < improvement[0] - 1.0e-8
                    ):
                        improvement = candidate
            if improvement is None:
                break
            distance, order, sides = improvement
        candidate_key = (distance, tuple(order), sides)
        if best is None or candidate_key < best:
            best = candidate_key
    assert best is not None
    return ArcTour(best[1], best[2], best[0])


class JointScale125Policy(SearchFirstPolicy):
    """空间相关性两方位修正版：联合规划七点和频道服务动作。"""

    name = "p3_joint_orienteering_spatial_exact2_v6"
    proxy_scale = 1.25
    two_opt_passes = 2

    def _service_arc(self, channel: int) -> ArcItem:
        observations = self.observations[channel]
        if len(observations) >= 2:
            polygon = localization_polygon(observations, self.no_signal_points[channel])
            center, _ = minimum_enclosing_circle(polygon)
            points = np.vstack([center, center])
            return ArcItem("service", channel, points, points.copy())

        entries = np.asarray(second_detection_candidates(observations[0]), dtype=float)
        feasible = observation_feasible_polygon(
            observations,
            self.no_signal_points[channel],
        )
        centroid = polygon_centroid(feasible)
        proxy = observations[0].point + self.proxy_scale * (
            centroid - observations[0].point
        )
        exits = np.vstack([proxy, proxy])
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
        deferred = {
            channel
            for channel in pending
            if len(self.observations[channel]) == 1
            and any(
                guaranteed_shared_observation(
                    self.observations[channel][0],
                    self.coverage_points[index],
                )
                and self._direction_has_effective_geometry(
                    channel, self.coverage_points[index]
                )[0]
                for index in remaining_stations
            )
        }
        service_items = [
            self._service_arc(channel) for channel in sorted(pending - deferred)
        ]
        return [*station_items, *service_items]

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
                two_opt_passes=self.two_opt_passes,
            )
            selected = items[tour.order[0]]
            side = tour.sides[0]
            if selected.kind == "station":
                self._visit_station(selected.identifier)
                remaining_stations.remove(selected.identifier)
            else:
                self._service_channel(
                    selected.identifier,
                    preferred_candidate_index=side,
                )
        self._update_search_certificate()
