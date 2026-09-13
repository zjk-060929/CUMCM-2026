"""问题 3 演练策略：七点保证发现、机会式插入与滚动动态规划。"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from geometry import (
    COVERAGE_LAYOUT_HEPTAGON,
    GeometryError,
    Observation,
    conservative_clear_route,
    coverage_stations,
    guaranteed_shared_observation,
    localization_polygon,
    minimum_enclosing_circle,
    observation_separation_deg,
    second_detection_candidates,
)
from protocol import JsonlLogger, SimulatorClient
from scheduler import (
    RoutePlan,
    current_edge_is_cheapest,
    optimal_candidate_route,
)


DOG_SPEED_MPS = 5.0
CHANNELS = tuple(range(1, 21))
MIN_SOURCE_COUNT = 10
MAX_SOURCE_COUNT = 16
# 由客户端允许的最坏配置推导：两个串行动作（当前动作和 /exit），
# 每个动作最多 3 次尝试，每次最多 5 秒，即 2×3×5=30 秒。
MIN_REAL_RESERVE_S = 30.0
# 共站补测至少保留10度交会角，即测角误差半宽的10倍。它不是正确性
# 门槛（解析保证只需大于1度），而是避免用近共线方向换取表面路程收益
# 的定位质量门槛；5/10/15/20/30度离线消融后采用综合耗时最小的10度。
MIN_PIGGYBACK_SEPARATION_DEG = 10.0


class PolicyError(RuntimeError):
    """策略无法在可证明安全的状态下继续。"""


class TimeBudgetReached(PolicyError):
    """为正常 /exit 保留时间而主动停止新动作。"""


@dataclass
class RobotState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    receiver_channel: int = 1
    virtual_time_s: float = 0.0
    measure_count: int = 0
    clear_attempt_count: int = 0
    clear_success_count: int = 0


@dataclass(frozen=True)
class RunSummary:
    strategy: str
    completed_normally: bool
    stop_reason: str
    stations_visited: int
    discovered_channels: tuple[int, ...]
    cleared_channels: tuple[int, ...]
    pending_channels: tuple[int, ...]
    measure_count: int
    clear_attempt_count: int
    clear_success_count: int
    no_signal_measure_count: int
    shared_observation_count: int
    search_certified: bool
    certified_absent_channels: tuple[int, ...]
    final_virtual_time_s: float
    piggyback_observation_count: int = 0
    reactive_refinement_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TimeBudget:
    """由 /enter 实际返回值构造，不把 1200 秒写死。"""

    def __init__(
        self,
        remaining_real_duration_s: float,
        max_virtual_duration_s: float,
        client: SimulatorClient,
    ):
        if remaining_real_duration_s < 0 or max_virtual_duration_s <= 0:
            raise ValueError("模拟器返回了非法时间上限")
        if client.last_action_started_monotonic_s is None:
            raise ValueError("缺少 /enter 请求开始时刻")
        # 以客户端首次发送 /enter 的时刻为基准，比按响应到达时刻计算更保守。
        self.real_deadline = (
            client.last_action_started_monotonic_s + float(remaining_real_duration_s)
        )
        self.max_virtual_duration_s = float(max_virtual_duration_s)
        self.request_wait_s = client.worst_case_request_wait_s

    def check_new_action(self, predicted_virtual_finish_s: float) -> None:
        # 一个新动作最多等待 request_wait_s，之后还需为 /exit 保留同样的最坏等待时间。
        required_real_s = max(MIN_REAL_RESERVE_S, 2.0 * self.request_wait_s)
        if self.real_deadline - time.monotonic() <= required_real_s:
            raise TimeBudgetReached("现实时间仅够保留 /exit，不再发送新的测量或清除动作")
        if predicted_virtual_finish_s >= self.max_virtual_duration_s:
            raise TimeBudgetReached("下一动作可能触及虚拟时间上限")

    def remaining_real_s(self) -> float:
        return max(0.0, self.real_deadline - time.monotonic())


class SearchFirstPolicy:
    """只适用于问题 3 的全向源混合滚动策略。

    保留旧类名是为了兼容既有运行入口；策略本身已不再是纯粹的
    search-first。默认启用机会式处理，尾段使用精确状态压缩动态规划。
    """

    name = "p3_origin_safe_piggyback_reactive_rolling_dp_v3"

    def __init__(
        self,
        client: SimulatorClient,
        logger: JsonlLogger,
        enter_response: dict[str, Any],
        *,
        opportunistic_during_search: bool = True,
        coverage_layout: str = COVERAGE_LAYOUT_HEPTAGON,
        shared_observations: bool = True,
        origin_information_scan: bool = True,
    ):
        self.client = client
        self.logger = logger
        self.state = RobotState()
        self.budget = TimeBudget(
            float(enter_response["remaining_real_duration_s"]),
            float(enter_response["max_virtual_duration_s"]),
            client,
        )
        self.observations: dict[int, list[Observation]] = {}
        self.no_signal_points: dict[int, list[np.ndarray]] = {
            channel: [] for channel in CHANNELS
        }
        self.coverage_no_signal_indices: dict[int, set[int]] = {
            channel: set() for channel in CHANNELS
        }
        self.discovered: set[int] = set()
        self.cleared: set[int] = set()
        self.stations_visited = 0
        self.no_signal_measure_count = 0
        self.shared_observation_count = 0
        self.piggyback_observation_count = 0
        self.reactive_refinement_count = 0
        self.search_certified = False
        self.certified_absent_channels: set[int] = set()
        self.opportunistic_during_search = bool(opportunistic_during_search)
        self.shared_observations = bool(shared_observations)
        self.origin_information_scan = bool(origin_information_scan)
        self.coverage_layout = coverage_layout
        self.coverage_points = coverage_stations(coverage_layout)

    def _move_time(self, target: np.ndarray) -> float:
        return float(np.linalg.norm(target - self.state.position)) / DOG_SPEED_MPS

    def _measure(self, target: np.ndarray, channel: int) -> dict[str, Any]:
        predicted = (
            self.state.virtual_time_s
            + self._move_time(target)
            + (1.0 if channel != self.state.receiver_channel else 0.0)
            + 5.0
        )
        self.budget.check_new_action(predicted)
        response = self.client.measure(float(target[0]), float(target[1]), channel)
        self.state.position = np.array(target, dtype=float)
        self.state.receiver_channel = channel
        self.state.virtual_time_s = float(response["virtual_time_s"])
        self.state.measure_count += 1
        self.logger.write(
            "policy_measure",
            channel=channel,
            position=target.tolist(),
            result=response["measure_result"],
            bearing_deg=response.get("svd_deg"),
            virtual_time_s=self.state.virtual_time_s,
        )
        return response

    def _clear(self, target: np.ndarray, channel: int) -> bool:
        # 成功清除耗时 5 秒，比未发现的 3 秒更长，故用 5 秒作上界。
        predicted = self.state.virtual_time_s + self._move_time(target) + 5.0
        self.budget.check_new_action(predicted)
        response = self.client.clear(float(target[0]), float(target[1]), channel)
        self.state.position = np.array(target, dtype=float)
        self.state.virtual_time_s = float(response["virtual_time_s"])
        self.state.clear_attempt_count += 1
        success = response["clear_result"] == "success"
        if success:
            self.state.clear_success_count += 1
            self.cleared.add(channel)
        self.logger.write(
            "policy_clear",
            channel=channel,
            position=target.tolist(),
            result=response["clear_result"],
            virtual_time_s=self.state.virtual_time_s,
        )
        return success

    def _record_direction(
        self, channel: int, point: np.ndarray, response: dict[str, Any]
    ) -> Observation:
        observation = Observation(
            float(point[0]), float(point[1]), float(response["svd_deg"])
        )
        self.observations.setdefault(channel, []).append(observation)
        self.discovered.add(channel)
        return observation

    def _record_no_signal(
        self,
        channel: int,
        point: np.ndarray,
        *,
        phase: str,
        station_index: int | None = None,
    ) -> None:
        vector = np.asarray(point, dtype=float).copy()
        if not any(
            float(np.linalg.norm(vector - previous)) <= 1.0e-8
            for previous in self.no_signal_points[channel]
        ):
            self.no_signal_points[channel].append(vector)
        if station_index is not None:
            self.coverage_no_signal_indices[channel].add(station_index)
        self.no_signal_measure_count += 1
        self.logger.write(
            "no_signal_constraint_added",
            channel=channel,
            phase=phase,
            position=vector.tolist(),
            exclusion_radius_m=1000.0,
            station_index=(station_index + 1 if station_index is not None else None),
        )

    def _clear_near(self, channel: int, point: np.ndarray) -> None:
        self.discovered.add(channel)
        if not self._clear(point, channel):
            raise PolicyError(
                f"频道 {channel} 返回 near，但在同一点调用 /clear 没有成功"
            )

    def _scan_origin_information(self) -> None:
        """在起点零移动扫描20频道，为后续保底站批量提供第一方位线。"""

        if not self.origin_information_scan:
            return
        origin = np.zeros(2, dtype=float)
        self.logger.write(
            "origin_information_scan_start",
            position=origin.tolist(),
            channels=list(CHANNELS),
        )
        for channel in CHANNELS:
            response = self._measure(origin, channel)
            result = response["measure_result"]
            if result == "direction":
                self._record_direction(channel, origin, response)
            elif result == "near":
                self._clear_near(channel, origin)
            else:
                self._record_no_signal(channel, origin, phase="origin_information")
            if len(self.discovered) == MAX_SOURCE_COUNT:
                break
        self.logger.write(
            "origin_information_scan_complete",
            discovered_channels=sorted(self.discovered),
            cleared_channels=sorted(self.cleared),
            virtual_time_s=self.state.virtual_time_s,
        )

    def _scan_coverage_stations(self) -> None:
        # 若原点零移动扫描已经发现题面上限16个频道，数量上限本身就是
        # 完整搜索证书；此时再走到第一个保底点只会增加无意义移动。
        if len(self.discovered) == MAX_SOURCE_COUNT:
            self.logger.write(
                "coverage_skipped",
                reason="原点已发现题面上限16个频道",
            )
            self._update_search_certificate()
            return

        stations = self.coverage_points
        for station_index, station in enumerate(stations):
            unknown = [channel for channel in CHANNELS if channel not in self.discovered]
            if self.state.receiver_channel in unknown:
                unknown.remove(self.state.receiver_channel)
                unknown.insert(0, self.state.receiver_channel)

            self.logger.write(
                "station_start",
                station_index=station_index + 1,
                position=station.tolist(),
                channels_to_scan=unknown,
            )
            for channel in unknown:
                response = self._measure(station, channel)
                result = response["measure_result"]
                if result == "direction":
                    self._record_direction(channel, station, response)
                elif result == "near":
                    self._clear_near(channel, station)
                elif result == "no_signal":
                    self._record_no_signal(
                        channel,
                        station,
                        phase="coverage",
                        station_index=station_index,
                    )

                # 题面给出的干扰源数量上限是 16；发现 16 个后可证明没有第 17 个。
                if len(self.discovered) == MAX_SOURCE_COUNT:
                    break

            self.stations_visited += 1
            self.logger.write(
                "station_complete",
                station_index=station_index + 1,
                discovered_channels=sorted(self.discovered),
                cleared_channels=sorted(self.cleared),
                virtual_time_s=self.state.virtual_time_s,
            )
            if self.shared_observations:
                self._collect_shared_observations(station, station_index)
            if len(self.discovered) == MAX_SOURCE_COUNT:
                self.logger.write("coverage_early_stop", reason="已发现题面上限 16 个频道")
                break
            if self.opportunistic_during_search and station_index + 1 < len(stations):
                self._service_opportunities(station_index, stations)

        self._update_search_certificate()

    def _collect_shared_observations(
        self, station: np.ndarray, station_index: int
    ) -> None:
        """在既有停靠点为其他频道顺带取得可证明有效的第二条示向度。"""

        candidates = [
            channel
            for channel in sorted(self.observations)
            if channel not in self.cleared
            and len(self.observations[channel]) == 1
            and guaranteed_shared_observation(self.observations[channel][0], station)
        ]
        if self.state.receiver_channel in candidates:
            candidates.remove(self.state.receiver_channel)
            candidates.insert(0, self.state.receiver_channel)
        for channel in candidates:
            response = self._measure(station, channel)
            result = response["measure_result"]
            if result == "direction":
                self._record_direction(channel, station, response)
                self.shared_observation_count += 1
                self.logger.write(
                    "shared_observation_collected",
                    channel=channel,
                    station_index=station_index + 1,
                    position=station.tolist(),
                )
            elif result == "near":
                self._clear_near(channel, station)
            else:
                # 几何判据本应保证最小接收半径内可接收；仍记录反馈并交由
                # 后续安全候选处理，避免把接口异常伪装成正观测。
                self._record_no_signal(
                    channel,
                    station,
                    phase="shared_observation",
                )
                self.logger.write(
                    "shared_observation_unexpected_no_signal",
                    channel=channel,
                    station_index=station_index + 1,
                )

    def _update_search_certificate(self) -> None:
        if len(self.discovered) == MAX_SOURCE_COUNT:
            self.search_certified = True
            self.certified_absent_channels = set()
            self.logger.write(
                "search_certificate",
                certified=True,
                basis="已发现题面上限16个互异频道",
                absent_channels=[],
            )
            return

        required = set(range(len(self.coverage_points)))
        self.certified_absent_channels = {
            channel
            for channel in CHANNELS
            if channel not in self.discovered
            and self.coverage_no_signal_indices[channel] == required
        }
        undiscovered = set(CHANNELS) - self.discovered
        self.search_certified = (
            self.stations_visited == len(self.coverage_points)
            and undiscovered == self.certified_absent_channels
        )
        self.logger.write(
            "search_certificate",
            certified=self.search_certified,
            basis="每个未发现频道的1000米无信号排除圆覆盖整个目标域",
            absent_channels=sorted(self.certified_absent_channels),
            uncertified_channels=sorted(undiscovered - self.certified_absent_channels),
        )

    def _candidate_map(self, channels: set[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for channel in channels:
            observations = self.observations[channel]
            if len(observations) >= 2:
                polygon = localization_polygon(
                    observations, self.no_signal_points[channel]
                )
                center, _ = minimum_enclosing_circle(polygon)
                result[channel] = (center.copy(), center.copy())
            else:
                result[channel] = second_detection_candidates(observations[0])
        return result

    @staticmethod
    def _clear_route_upper_time(start: np.ndarray, route: np.ndarray) -> float:
        """按最后一个候选点才成功，计算有限清除路线的时间上界。"""

        if len(route) == 0:
            return math.inf
        movement_m = float(np.linalg.norm(route[0] - start))
        if len(route) > 1:
            movement_m += float(
                np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1))
            )
        return movement_m / DOG_SPEED_MPS + 3.0 * (len(route) - 1) + 5.0

    def _supplemental_measurement_choice(
        self,
        channel: int,
        polygon: np.ndarray,
        direct_route: np.ndarray,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """比较直接有限覆盖与再测一次后清除的模型化时间。

        后一种方案的未来示向度尚未知，因此在当前定位多边形的全部顶点、
        边中点和最小包围圆圆心上，分别施加 -1、0、+1 度误差进行有限
        情景评价。该评价只决定是否补测；无论选择哪条分支，最终清除仍由
        20 m 有限覆盖兜底保证。
        """

        observations = self.observations[channel]
        if len(observations) < 2:
            return None, {"reason": "insufficient_observations"}
        direct_cost = self._clear_route_upper_time(self.state.position, direct_route)
        if len(direct_route) == 1:
            return None, {
                "direct_clear_upper_time_s": direct_cost,
                "reason": "one_clear_action_already_guarantees_success",
            }
        center, _ = minimum_enclosing_circle(polygon)
        midpoints = 0.5 * (polygon + np.roll(polygon, -1, axis=0))
        raw_targets = np.vstack([polygon, midpoints, center])
        # 外切圆多边形和角带边界会产生恰好共线的浮点退化；向内缩
        # 1e-6 只用于数值稳定，不改变一度误差的建模范围。
        possible_targets = center + (1.0 - 1.0e-6) * (raw_targets - center)
        prior_points = [observation.point for observation in observations]

        unused_candidates = [
            candidate
            for candidate in second_detection_candidates(observations[0])
            if not any(
                float(np.linalg.norm(candidate - point)) <= 1.0e-7
                for point in prior_points
            )
        ]
        switch_time = 1.0 if channel != self.state.receiver_channel else 0.0
        if unused_candidates:
            optimistic_supplement = min(
                self._move_time(candidate) + switch_time + 5.0 + 5.0
                for candidate in unused_candidates
            )
            if optimistic_supplement >= direct_cost - 1.0e-9:
                return None, {
                    "direct_clear_upper_time_s": direct_cost,
                    "supplement_then_clear_lower_bound_s": optimistic_supplement,
                    "reason": "supplement_lower_bound_not_better",
                }

        choices: list[tuple[float, np.ndarray, int]] = []
        for candidate in unused_candidates:
            worst_future = 0.0
            evaluated = 0
            feasible = True
            for target in possible_targets:
                true_bearing = math.degrees(
                    math.atan2(
                        float(target[1] - candidate[1]),
                        float(target[0] - candidate[0]),
                    )
                )
                for error_deg in (-1.0 + 1.0e-6, 0.0, 1.0 - 1.0e-6):
                    hypothetical = Observation(
                        float(candidate[0]),
                        float(candidate[1]),
                        (true_bearing + error_deg) % 360.0,
                    )
                    try:
                        future_polygon = localization_polygon(
                            [*observations, hypothetical],
                            self.no_signal_points[channel],
                        )
                    except GeometryError:
                        feasible = False
                        break
                    future_route, _ = conservative_clear_route(
                        future_polygon,
                        candidate,
                        no_signal_points=self.no_signal_points[channel],
                    )
                    worst_future = max(
                        worst_future,
                        self._clear_route_upper_time(candidate, future_route),
                    )
                    evaluated += 1
                if not feasible:
                    break
            if feasible and evaluated:
                total = (
                    self._move_time(candidate)
                    + switch_time
                    + 5.0
                    + worst_future
                )
                choices.append((total, candidate.copy(), evaluated))

        if not choices:
            return None, {
                "direct_clear_upper_time_s": direct_cost,
                "reason": "no_finite_supplement_candidate",
            }
        best_total, best_candidate, evaluated = min(choices, key=lambda item: item[0])
        details = {
            "direct_clear_upper_time_s": direct_cost,
            "supplement_then_clear_estimated_upper_time_s": best_total,
            "finite_scenarios_evaluated": evaluated,
            "candidate": best_candidate.tolist(),
        }
        if best_total + 1.0e-9 < direct_cost:
            return best_candidate, details
        details["reason"] = "direct_clear_has_no_larger_estimated_cost"
        return None, details

    def _rolling_plan(
        self,
        channels: set[int],
        terminal: np.ndarray | None,
        phase: str,
    ) -> RoutePlan:
        started = time.monotonic()
        plan = optimal_candidate_route(
            self.state.position,
            self._candidate_map(channels),
            terminal=terminal,
        )
        elapsed = time.monotonic() - started
        self.logger.write(
            "rolling_dp_plan",
            phase=phase,
            task_count=len(channels),
            planned_distance_m=plan.distance_m,
            planned_channels=[decision.channel for decision in plan.decisions],
            planned_candidate_indices=[
                decision.candidate_index for decision in plan.decisions
            ],
            planned_action_types=[
                (
                    "clear_from_localized_region"
                    if len(self.observations[decision.channel]) >= 2
                    else "obtain_second_bearing_then_clear"
                )
                for decision in plan.decisions
            ],
            planned_travel_time_s=plan.distance_m / DOG_SPEED_MPS,
            planned_fixed_service_lower_bound_s=sum(
                5.0 if len(self.observations[channel]) >= 2 else 10.0
                for channel in channels
            ),
            computation_time_s=elapsed,
            remaining_real_time_s=self.budget.remaining_real_s(),
        )
        return plan

    def _service_opportunities(
        self, station_index: int, stations: list[np.ndarray]
    ) -> None:
        """只处理当前边最便宜且没有未来免费第二观测站的已知任务。"""

        next_station = stations[station_index + 1]
        future_stations = stations[station_index + 1 :]
        future_edges = [
            (stations[index], stations[index + 1])
            for index in range(station_index + 1, len(stations) - 1)
        ]
        while True:
            pending = set(self.observations) - self.cleared
            deferred_for_shared = {
                channel
                for channel in pending
                if len(self.observations[channel]) == 1
                and any(
                    guaranteed_shared_observation(
                        self.observations[channel][0], candidate_station
                    )
                    for candidate_station in future_stations
                )
            }
            pending -= deferred_for_shared
            eligible: set[int] = set()
            insertion_details: dict[int, dict[str, float]] = {}
            for channel, candidates in self._candidate_map(pending).items():
                is_cheapest, current_cost, best_cost = current_edge_is_cheapest(
                    self.state.position,
                    next_station,
                    future_edges,
                    candidates,
                )
                insertion_details[channel] = {
                    "current_insertion_m": current_cost,
                    "best_remaining_insertion_m": best_cost,
                }
                if is_cheapest:
                    eligible.add(channel)

            self.logger.write(
                "opportunity_evaluation",
                station_index=station_index + 1,
                pending_channels=sorted(pending),
                deferred_for_future_shared_observation=sorted(deferred_for_shared),
                eligible_channels=sorted(eligible),
                insertion_details=insertion_details,
            )
            if not eligible:
                return

            plan = self._rolling_plan(
                eligible,
                terminal=next_station,
                phase=f"coverage_edge_{station_index + 1}",
            )
            decision = plan.decisions[0]
            self.logger.write(
                "opportunistic_service_selected",
                channel=decision.channel,
                preferred_candidate_index=decision.candidate_index,
                next_station=next_station.tolist(),
            )
            self._service_channel(
                decision.channel,
                preferred_candidate_index=decision.candidate_index,
            )

    def _service_channel(
        self, channel: int, preferred_candidate_index: int | None = None
    ) -> None:
        observations = self.observations[channel]
        polygon: np.ndarray | None = None
        last_geometry_error: GeometryError | None = None
        newly_localized = False

        if len(observations) >= 2:
            polygon = localization_polygon(
                observations, self.no_signal_points[channel]
            )
        else:
            first = observations[0]
            candidates = second_detection_candidates(first)
            if preferred_candidate_index not in (None, 0, 1):
                raise ValueError("第二检测点候选编号只能是0或1")
            if preferred_candidate_index is None:
                candidate_order = sorted(
                    range(2),
                    key=lambda index: float(
                        np.linalg.norm(candidates[index] - self.state.position)
                    ),
                )
            else:
                candidate_order = [preferred_candidate_index, 1 - preferred_candidate_index]

            for attempt_index, candidate_index in enumerate(candidate_order, 1):
                candidate = candidates[candidate_index]
                response = self._measure(candidate, channel)
                result = response["measure_result"]
                if result == "near":
                    self._clear_near(channel, candidate)
                    return
                if result == "no_signal":
                    self._record_no_signal(
                        channel,
                        candidate,
                        phase="second_observation",
                    )
                    self.logger.write(
                        "second_point_no_signal",
                        channel=channel,
                        attempt_index=attempt_index,
                        candidate_index=candidate_index,
                        position=candidate.tolist(),
                    )
                    continue

                self._record_direction(channel, candidate, response)
                try:
                    polygon = localization_polygon(
                        observations, self.no_signal_points[channel]
                    )
                    newly_localized = True
                    break
                except GeometryError as error:
                    last_geometry_error = error
                    self.logger.write(
                        "localization_retry",
                        channel=channel,
                        attempt_index=attempt_index,
                        candidate_index=candidate_index,
                        error=error,
                    )

        if polygon is None:
            detail = f": {last_geometry_error}" if last_geometry_error else ""
            raise PolicyError(f"频道 {channel} 的安全第二点未形成定位区域{detail}")

        # 第二条方向线到手后先把控制权交回全局滚动DP。下一轮会把该频道
        # 表示为“访问定位区域包围圆中心”的清除任务，并与其余补测/清除
        # 任务统一排序。这只改变动作次序，不删减任何保证性动作。
        if newly_localized:
            center, enclosing_radius = minimum_enclosing_circle(polygon)
            self.logger.write(
                "second_observation_complete_replan",
                channel=channel,
                polygon_vertices=polygon.tolist(),
                enclosing_center=center.tolist(),
                enclosing_radius_m=enclosing_radius,
            )
            self._collect_piggyback_observations(
                self.state.position,
                serviced_channel=channel,
            )
            return

        route, enclosing_radius = conservative_clear_route(
            polygon,
            self.state.position,
            no_signal_points=self.no_signal_points[channel],
        )
        supplement, comparison = self._supplemental_measurement_choice(
            channel,
            polygon,
            route,
        )
        self.logger.write(
            "clear_or_supplement_comparison",
            channel=channel,
            comparison_scope="before_post_failure_reactive_refinement",
            **comparison,
            selected=("supplement" if supplement is not None else "direct_clear"),
        )
        if supplement is not None:
            response = self._measure(supplement, channel)
            result = response["measure_result"]
            if result == "near":
                self._clear_near(channel, supplement)
                return
            if result == "direction":
                self._record_direction(channel, supplement, response)
                polygon = localization_polygon(
                    observations, self.no_signal_points[channel]
                )
            else:
                self._record_no_signal(
                    channel,
                    supplement,
                    phase="time_selected_supplement",
                )
            route, enclosing_radius = conservative_clear_route(
                polygon,
                self.state.position,
                no_signal_points=self.no_signal_points[channel],
            )
        self.logger.write(
            "localization_complete",
            channel=channel,
            observation_count=len(observations),
            polygon_vertices=polygon.tolist(),
            enclosing_radius_m=enclosing_radius,
            clear_route_points=len(route),
            clear_mode=(
                "single_minimum_enclosing_circle_center"
                if enclosing_radius <= 20.0 + 1.0e-8
                else "finite_20m_cover_with_no_signal_pruning"
            ),
            no_signal_constraints=len(self.no_signal_points[channel]),
        )
        for point_index, point in enumerate(route):
            if self._clear(point, channel):
                return
            # 正常模型下，半径不超过20 m的单中心清除必然成功；若接口
            # 边界或瞬时反馈仍返回失败，也原地补测一次，而不是直接中止。
            if point_index == 0:
                if self._reactive_refinement_after_center_failure(
                    channel,
                    point,
                ):
                    return
        raise PolicyError(f"频道 {channel} 的保守覆盖路线走完后仍未清除")

    def _collect_piggyback_observations(
        self,
        point: np.ndarray,
        *,
        serviced_channel: int,
    ) -> None:
        """在已经到达的第二观测点为其他兼容频道零移动补测。"""

        candidates = [
            other
            for other in sorted(set(self.observations) - self.cleared)
            if other != serviced_channel
            and len(self.observations[other]) == 1
            and guaranteed_shared_observation(self.observations[other][0], point)
            and observation_separation_deg(self.observations[other][0], point)
            >= MIN_PIGGYBACK_SEPARATION_DEG
        ]
        if self.state.receiver_channel in candidates:
            candidates.remove(self.state.receiver_channel)
            candidates.insert(0, self.state.receiver_channel)
        for channel in candidates:
            response = self._measure(point, channel)
            result = response["measure_result"]
            if result == "direction":
                self._record_direction(channel, point, response)
                self.piggyback_observation_count += 1
                self.logger.write(
                    "piggyback_observation_collected",
                    channel=channel,
                    serviced_channel=serviced_channel,
                    position=point.tolist(),
                    separation_deg=observation_separation_deg(
                        self.observations[channel][0], point
                    ),
                )
            elif result == "near":
                self._clear_near(channel, point)
                self.piggyback_observation_count += 1
            else:
                # 在题设误差和固定接收半径模型下，解析判据已保证不会走到
                # 此分支；仍保留证据并交给下一轮安全规划处理接口异常。
                self._record_no_signal(
                    channel,
                    point,
                    phase="piggyback_second_observation",
                )
                self.logger.write(
                    "piggyback_observation_unexpected_no_signal",
                    channel=channel,
                    serviced_channel=serviced_channel,
                    position=point.tolist(),
                )

    def _reactive_refinement_after_center_failure(
        self,
        channel: int,
        failed_center: np.ndarray,
    ) -> bool:
        """包围圆中心清除失败后原地补测，再覆盖缩小后的可行域。

        首次失败本身证明真实源不在中心的20米清除圆内。原地测向只增加
        可靠信息：direction加入新的正负1度半平面，no_signal则通过同频
        道固定接收半径加入距离支配半平面。若新路线因接口异常未能完成，
        调用者仍继续原有限覆盖路线的其余点，因此确定性兜底不被替换。
        """

        self.reactive_refinement_count += 1
        response = self._measure(failed_center, channel)
        result = response["measure_result"]
        if result == "near":
            # 在可靠模型中，刚刚clear失败后不应在同一点返回near；重试一次
            # 可以处理瞬时清除异常，若仍失败则回到原覆盖路线。
            success = self._clear(failed_center, channel)
            self.logger.write(
                "reactive_refinement_inconsistent_near",
                channel=channel,
                position=failed_center.tolist(),
                retry_success=success,
            )
            return success
        if result == "direction":
            self._record_direction(channel, failed_center, response)
        else:
            self._record_no_signal(
                channel,
                failed_center,
                phase="post_mec_clear_failure",
            )

        try:
            refined_polygon = localization_polygon(
                self.observations[channel],
                self.no_signal_points[channel],
            )
            refined_route, refined_radius = conservative_clear_route(
                refined_polygon,
                self.state.position,
                no_signal_points=self.no_signal_points[channel],
            )
        except GeometryError as error:
            self.logger.write(
                "reactive_refinement_geometry_fallback",
                channel=channel,
                error=str(error),
            )
            return False

        self.logger.write(
            "reactive_refinement_route",
            channel=channel,
            feedback=result,
            polygon_vertices=refined_polygon.tolist(),
            enclosing_radius_m=refined_radius,
            route_points=len(refined_route),
        )
        for point in refined_route:
            # 同一个中心刚刚得到可靠clear失败，无需原地重复调用。
            if float(np.linalg.norm(point - failed_center)) <= 1.0e-7:
                continue
            if self._clear(point, channel):
                return True
        self.logger.write(
            "reactive_refinement_old_route_fallback",
            channel=channel,
            reason="缩小域路线未成功，继续原有限覆盖路线的未访问点",
        )
        return False

    def run(self) -> RunSummary:
        self._scan_origin_information()
        self._scan_coverage_stations()
        if not self.search_certified:
            raise PolicyError("保证性搜索尚无完整排查证书，拒绝报告任务完成")
        if len(self.discovered) < MIN_SOURCE_COUNT:
            raise PolicyError(
                f"完成七点覆盖后仅发现 {len(self.discovered)} 个频道，"
                "低于问题3规定的最少10个；停止并保留日志检查"
            )
        pending = set(self.observations) - self.cleared
        while pending:
            plan = self._rolling_plan(pending, terminal=None, phase="tail_clearance")
            decision = plan.decisions[0]
            self._service_channel(
                decision.channel,
                preferred_candidate_index=decision.candidate_index,
            )
            pending = set(self.observations) - self.cleared

        summary = RunSummary(
            strategy=self.name,
            completed_normally=True,
            stop_reason=(
                "已发现并清除题面上限16个目标"
                if len(self.discovered) == MAX_SOURCE_COUNT
                else "所有目标已清除且其余频道均取得覆盖排查证书"
            ),
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
        self.logger.write("policy_complete", summary=summary.as_dict())
        return summary

    def interrupted_summary(self, reason: str) -> RunSummary:
        pending = set(self.observations) - self.cleared
        return RunSummary(
            strategy=self.name,
            completed_normally=False,
            stop_reason=reason,
            stations_visited=self.stations_visited,
            discovered_channels=tuple(sorted(self.discovered)),
            cleared_channels=tuple(sorted(self.cleared)),
            pending_channels=tuple(sorted(pending)),
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
