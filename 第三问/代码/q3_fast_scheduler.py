"""Numba acceleration of the exact two-candidate Held--Karp scheduler.

The recurrence and deterministic tie order match ``代码/scheduler.py``.  This
changes only offline wall-clock runtime, never the simulator's virtual time.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numba import njit

from scheduler import RoutePlan, ServiceDecision


@njit(cache=True)
def _held_karp(
    start: np.ndarray,
    points: np.ndarray,
    terminal: np.ndarray,
    has_terminal: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = points.shape[0]
    state_count = 1 << n
    costs = np.full((state_count, n, 2), np.inf)
    previous_task = np.full((state_count, n, 2), -1, dtype=np.int8)
    previous_side = np.full((state_count, n, 2), -1, dtype=np.int8)

    for task in range(n):
        for side in range(2):
            dx = points[task, side, 0] - start[0]
            dy = points[task, side, 1] - start[1]
            costs[1 << task, task, side] = np.sqrt(dx * dx + dy * dy)

    for mask in range(1, state_count):
        if mask & (mask - 1) == 0:
            continue
        for last in range(n):
            bit = 1 << last
            if (mask & bit) == 0:
                continue
            predecessor_mask = mask ^ bit
            for side in range(2):
                best = np.inf
                best_task = -1
                best_side = -1
                for before in range(n):
                    if (predecessor_mask & (1 << before)) == 0:
                        continue
                    for before_side in range(2):
                        dx = points[last, side, 0] - points[before, before_side, 0]
                        dy = points[last, side, 1] - points[before, before_side, 1]
                        candidate = costs[predecessor_mask, before, before_side] + np.sqrt(
                            dx * dx + dy * dy
                        )
                        if candidate < best:
                            best = candidate
                            best_task = before
                            best_side = before_side
                costs[mask, last, side] = best
                previous_task[mask, last, side] = best_task
                previous_side[mask, last, side] = best_side

    full = state_count - 1
    best = np.inf
    last_task = -1
    last_side = -1
    for task in range(n):
        for side in range(2):
            candidate = costs[full, task, side]
            if has_terminal:
                dx = terminal[0] - points[task, side, 0]
                dy = terminal[1] - points[task, side, 1]
                candidate += np.sqrt(dx * dx + dy * dy)
            if candidate < best:
                best = candidate
                last_task = task
                last_side = side

    task_order = np.empty(n, dtype=np.int16)
    side_order = np.empty(n, dtype=np.int8)
    mask = full
    output = n - 1
    while last_task >= 0:
        task_order[output] = last_task
        side_order[output] = last_side
        before_task = int(previous_task[mask, last_task, last_side])
        before_side = int(previous_side[mask, last_task, last_side])
        mask ^= 1 << last_task
        last_task = before_task
        last_side = before_side
        output -= 1
    return task_order, side_order, float(best)


def fast_optimal_candidate_route(
    start: np.ndarray,
    candidates_by_channel: Mapping[int, Sequence[np.ndarray]],
    terminal: np.ndarray | None = None,
) -> RoutePlan:
    start = np.asarray(start, dtype=np.float64)
    channels = tuple(sorted(candidates_by_channel))
    if not channels:
        distance = 0.0 if terminal is None else float(np.linalg.norm(np.asarray(terminal) - start))
        return RoutePlan((), distance)
    points = np.empty((len(channels), 2, 2), dtype=np.float64)
    for task, channel in enumerate(channels):
        candidates = tuple(candidates_by_channel[channel])
        if len(candidates) != 2:
            raise ValueError(f"频道 {channel} 必须恰有两个候选点")
        points[task, 0] = np.asarray(candidates[0], dtype=float)
        points[task, 1] = np.asarray(candidates[1], dtype=float)
    terminal_array = (
        np.zeros(2, dtype=np.float64)
        if terminal is None
        else np.asarray(terminal, dtype=np.float64)
    )
    tasks, sides, distance = _held_karp(start, points, terminal_array, terminal is not None)
    decisions = tuple(
        ServiceDecision(
            channel=channels[int(task)],
            candidate_index=int(side),
            point=points[int(task), int(side)].copy(),
        )
        for task, side in zip(tasks, sides)
    )
    return RoutePlan(decisions, distance)
