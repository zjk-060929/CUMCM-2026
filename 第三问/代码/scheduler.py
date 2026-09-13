"""问题 3 的纯几何调度器。

本模块不访问模拟器，只依据当前证据生成的两点候选（单方位任务为
左右两个安全第二检测点，已定位任务为两个相同的包围圆中心），求解
带二选一节点的最短开放路线。策略只执行计划的首个动作；取得新方向
线、完成清除或得到异常反馈后都会重新调用，因此属于逐动作滚动规划。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


EPS = 1.0e-9


@dataclass(frozen=True)
class ServiceDecision:
    """动态规划路线中的一个频道及其首选第二检测点。"""

    channel: int
    candidate_index: int
    point: np.ndarray


@dataclass(frozen=True)
class RoutePlan:
    """给定候选点和可选终点时的最短几何路线。"""

    decisions: tuple[ServiceDecision, ...]
    distance_m: float


def _validated_candidates(
    candidates_by_channel: Mapping[int, Sequence[np.ndarray]],
) -> tuple[tuple[int, ...], np.ndarray]:
    channels = tuple(sorted(candidates_by_channel))
    if len(channels) > 16:
        raise ValueError("问题3最多只有16个干扰源")
    points = np.empty((len(channels), 2, 2), dtype=float)
    for index, channel in enumerate(channels):
        candidates = tuple(candidates_by_channel[channel])
        if len(candidates) != 2:
            raise ValueError(f"频道 {channel} 必须恰有两个第二检测点候选")
        for side, point in enumerate(candidates):
            vector = np.asarray(point, dtype=float)
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"频道 {channel} 的候选点不是有限二维坐标")
            points[index, side] = vector
    return channels, points


def optimal_candidate_route(
    start: np.ndarray,
    candidates_by_channel: Mapping[int, Sequence[np.ndarray]],
    terminal: np.ndarray | None = None,
) -> RoutePlan:
    """用 Held-Karp 状态压缩 DP 求二选一任务的最短路线。

    路线从 ``start`` 出发，每个频道的两个候选点中恰访问一个；若给出
    ``terminal``，路线最后还必须到达该点。返回完整最优顺序和距离。
    """

    start = np.asarray(start, dtype=float)
    if start.shape != (2,) or not np.all(np.isfinite(start)):
        raise ValueError("起点必须是有限二维坐标")
    if terminal is not None:
        terminal = np.asarray(terminal, dtype=float)
        if terminal.shape != (2,) or not np.all(np.isfinite(terminal)):
            raise ValueError("终点必须是有限二维坐标")

    channels, points = _validated_candidates(candidates_by_channel)
    task_count = len(channels)
    if task_count == 0:
        distance = 0.0 if terminal is None else float(np.linalg.norm(terminal - start))
        return RoutePlan((), distance)

    state_count = 1 << task_count
    costs = np.full((state_count, task_count, 2), np.inf, dtype=float)
    previous_task = np.full((state_count, task_count, 2), -1, dtype=np.int8)
    previous_side = np.full((state_count, task_count, 2), -1, dtype=np.int8)

    transitions = np.linalg.norm(
        points[:, :, np.newaxis, np.newaxis, :]
        - points[np.newaxis, np.newaxis, :, :, :],
        axis=-1,
    )
    for task in range(task_count):
        bit = 1 << task
        costs[bit, task] = np.linalg.norm(points[task] - start, axis=1)

    # 对每个状态按“最后完成的任务”分解。相同距离时保留编号和候选序
    # 较小的前驱，使不同电脑得到完全一致的计划。
    for mask in range(1, state_count):
        if mask & (mask - 1) == 0:
            continue
        remaining_bits = mask
        while remaining_bits:
            last_bit = remaining_bits & -remaining_bits
            last_task = last_bit.bit_length() - 1
            predecessor_mask = mask ^ last_bit
            predecessor_costs = costs[predecessor_mask]
            for side in range(2):
                alternatives = predecessor_costs + transitions[:, :, last_task, side]
                flat_index = int(np.argmin(alternatives))
                best = float(alternatives.flat[flat_index])
                predecessor_task, predecessor_candidate = divmod(flat_index, 2)
                costs[mask, last_task, side] = best
                previous_task[mask, last_task, side] = predecessor_task
                previous_side[mask, last_task, side] = predecessor_candidate
            remaining_bits ^= last_bit

    full_mask = state_count - 1
    final_costs = costs[full_mask].copy()
    if terminal is not None:
        final_costs += np.linalg.norm(points - terminal, axis=2)
    final_flat = int(np.argmin(final_costs))
    last_task, last_side = divmod(final_flat, 2)
    distance = float(final_costs.flat[final_flat])

    reversed_decisions: list[ServiceDecision] = []
    mask = full_mask
    while last_task >= 0:
        reversed_decisions.append(
            ServiceDecision(
                channel=channels[last_task],
                candidate_index=last_side,
                point=points[last_task, last_side].copy(),
            )
        )
        predecessor_task = int(previous_task[mask, last_task, last_side])
        predecessor_candidate = int(previous_side[mask, last_task, last_side])
        mask ^= 1 << last_task
        last_task, last_side = predecessor_task, predecessor_candidate

    return RoutePlan(tuple(reversed(reversed_decisions)), distance)


def insertion_cost(
    start: np.ndarray,
    end: np.ndarray,
    candidates: Sequence[np.ndarray],
) -> float:
    """把一个二选一检测任务插入有向边后增加的最短路程。"""

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direct = float(np.linalg.norm(end - start))
    values = [
        float(np.linalg.norm(point - start) + np.linalg.norm(end - point) - direct)
        for point in candidates
    ]
    if len(values) != 2:
        raise ValueError("机会式任务必须恰有两个候选点")
    return min(values)


def current_edge_is_cheapest(
    current_start: np.ndarray,
    current_end: np.ndarray,
    future_edges: Sequence[tuple[np.ndarray, np.ndarray]],
    candidates: Sequence[np.ndarray],
) -> tuple[bool, float, float]:
    """判断当前边是否为剩余保证性搜索骨架上的最便宜插入边。"""

    current = insertion_cost(current_start, current_end, candidates)
    future = [insertion_cost(start, end, candidates) for start, end in future_edges]
    best = min([current, *future])
    tolerance = EPS * max(1.0, abs(current), abs(best))
    return current <= best + tolerance, current, best
