"""问题3的相关误差稳健机会式多观测实验策略（非当前默认）。

已有两条有效方向线后，仅当第三/第四次观测在代表性误差边界下仍能
相对直接清除节省至少 50 s 时才执行。额外观测点不能是原地小幅挪动。

100组筛选中的虚拟时间收益不足0.2%，而本地计算显著增加，故当前
``config.json`` 不启用本策略；保留本文件只供后续离线研究。
"""

from __future__ import annotations

import math

import numpy as np

from geometry import (
    GeometryError,
    Observation,
    conservative_clear_route,
    guaranteed_shared_observation,
    localization_polygon,
    minimum_enclosing_circle,
)
from joint_policy import ArcItem, JointScale125Policy


class CorrelatedAdaptivePolicy(JointScale125Policy):
    """最多四次定位、按总时间净收益触发的联合滚动策略。"""

    name = "p3_joint_correlated_adaptive4_margin50_v1"
    target_observation_count = 4
    minimum_separation_m = 150.0
    minimum_crossing_deg = 10.0
    minimum_predicted_saving_s = 50.0
    candidate_shortlist = 10

    def _supplemental_measurement_choice(self, channel, polygon, direct_route):
        """关闭旧补测判据，避免两个判据重复触发。"""

        del channel, polygon, direct_route
        return None, {"reason": "controlled_by_correlated_adaptive_policy"}

    @staticmethod
    def _acute_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
        norm = float(np.linalg.norm(first) * np.linalg.norm(second))
        if norm <= 1.0e-12:
            return 0.0
        cosine = abs(float(np.dot(first, second))) / norm
        return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))

    def _extra_candidates(self, channel: int) -> list[np.ndarray]:
        observations = self.observations[channel]
        first = observations[0]
        angle = math.radians(first.bearing_deg)
        forward = np.asarray([math.cos(angle), math.sin(angle)])
        lateral = np.asarray([-math.sin(angle), math.cos(angle)])
        used = [observation.point for observation in observations]
        used.extend(self.no_signal_points[channel])
        candidates: list[np.ndarray] = []
        for beta_deg in (12.0, 22.0, 35.0, 48.0, 58.0):
            maximum_length = min(
                999.0,
                2000.0 * math.cos(math.radians(beta_deg + 1.0)) - 1.0,
            )
            for fraction in (0.55, 0.78, 0.97):
                length = fraction * maximum_length
                beta = math.radians(beta_deg)
                for side in (-1.0, 1.0):
                    point = first.point + length * (
                        math.cos(beta) * forward
                        + side * math.sin(beta) * lateral
                    )
                    if not guaranteed_shared_observation(first, point):
                        continue
                    if min(
                        float(np.linalg.norm(point - old)) for old in used
                    ) < self.minimum_separation_m:
                        continue
                    if any(
                        float(np.linalg.norm(point - old)) <= 1.0e-7
                        for old in candidates
                    ):
                        continue
                    candidates.append(point)
        return candidates

    def _candidate_scores(self, channel: int) -> list[tuple[float, np.ndarray]]:
        observations = self.observations[channel]
        polygon = localization_polygon(
            observations,
            self.no_signal_points[channel],
        )
        proxy, _ = minimum_enclosing_circle(polygon)
        old_vectors = [proxy - observation.point for observation in observations]
        rough: list[tuple[float, float, np.ndarray]] = []
        for point in self._extra_candidates(channel):
            new_vector = proxy - point
            best_crossing = max(
                (
                    self._acute_angle_deg(new_vector, vector)
                    for vector in old_vectors
                ),
                default=0.0,
            )
            if best_crossing < self.minimum_crossing_deg:
                continue
            move_and_measure = self._move_time(point) + (
                1.0 if channel != self.state.receiver_channel else 0.0
            ) + 5.0
            rough.append(
                (move_and_measure + 80.0 / best_crossing, -best_crossing, point)
            )
        rough.sort(
            key=lambda item: (
                item[0],
                item[1],
                float(item[2][0]),
                float(item[2][1]),
            )
        )

        scored: list[tuple[float, np.ndarray]] = []
        for _, _, point in rough[: self.candidate_shortlist]:
            bearing = math.degrees(
                math.atan2(
                    float(proxy[1] - point[1]),
                    float(proxy[0] - point[0]),
                )
            ) % 360.0
            hypothetical = Observation(float(point[0]), float(point[1]), bearing)
            try:
                future_polygon = localization_polygon(
                    [*observations, hypothetical],
                    self.no_signal_points[channel],
                )
                future_route, _ = conservative_clear_route(
                    future_polygon,
                    point,
                    no_signal_points=self.no_signal_points[channel],
                )
            except GeometryError:
                continue
            score = (
                self._move_time(point)
                + (1.0 if channel != self.state.receiver_channel else 0.0)
                + 5.0
                + self._clear_route_upper_time(point, future_route)
            )
            scored.append((score, point.copy()))
        return sorted(
            scored,
            key=lambda item: (item[0], float(item[1][0]), float(item[1][1])),
        )

    def _robust_candidate_scores(
        self,
        channel: int,
    ) -> list[tuple[float, np.ndarray]]:
        observations = self.observations[channel]
        polygon = localization_polygon(
            observations,
            self.no_signal_points[channel],
        )
        proxy, _ = minimum_enclosing_circle(polygon)
        robust: list[tuple[float, np.ndarray]] = []
        for _, point in self._candidate_scores(channel):
            center_bearing = math.degrees(
                math.atan2(
                    float(proxy[1] - point[1]),
                    float(proxy[0] - point[0]),
                )
            ) % 360.0
            possible_times: list[float] = []
            for error_deg in (-1.0, 0.0, 1.0):
                hypothetical = Observation(
                    float(point[0]),
                    float(point[1]),
                    (center_bearing + error_deg) % 360.0,
                )
                try:
                    future_polygon = localization_polygon(
                        [*observations, hypothetical],
                        self.no_signal_points[channel],
                    )
                    future_route, _ = conservative_clear_route(
                        future_polygon,
                        point,
                        no_signal_points=self.no_signal_points[channel],
                    )
                except GeometryError:
                    possible_times = []
                    break
                possible_times.append(
                    self._move_time(point)
                    + (1.0 if channel != self.state.receiver_channel else 0.0)
                    + 5.0
                    + self._clear_route_upper_time(point, future_route)
                )
            if possible_times:
                robust.append((max(possible_times), point.copy()))
        return sorted(
            robust,
            key=lambda item: (item[0], float(item[1][0]), float(item[1][1])),
        )

    def _planned_extra_point(self, channel: int) -> np.ndarray | None:
        observations = self.observations[channel]
        if not 2 <= len(observations) < self.target_observation_count:
            return None
        polygon = localization_polygon(
            observations,
            self.no_signal_points[channel],
        )
        direct_route, _ = conservative_clear_route(
            polygon,
            self.state.position,
            no_signal_points=self.no_signal_points[channel],
        )
        direct = self._clear_route_upper_time(self.state.position, direct_route)
        scores = self._robust_candidate_scores(channel)
        if not scores:
            return None
        estimated, point = scores[0]
        if estimated + self.minimum_predicted_saving_s < direct:
            return point
        return None

    def _service_arc(self, channel: int) -> ArcItem:
        if 2 <= len(self.observations[channel]) < self.target_observation_count:
            point = self._planned_extra_point(channel)
            if point is not None:
                values = np.vstack([point, point])
                return ArcItem("service", channel, values, values.copy())
        return super()._service_arc(channel)

    def _service_channel(
        self,
        channel: int,
        preferred_candidate_index: int | None = None,
    ) -> None:
        if 2 <= len(self.observations[channel]) < self.target_observation_count:
            point = self._planned_extra_point(channel)
            if point is not None:
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
                        phase="correlated_adaptive_extra_bearing",
                    )
                return
        return super()._service_channel(channel, preferred_candidate_index)
