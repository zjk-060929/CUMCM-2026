"""Offline-only paired ablation for Question 3 global rolling scheduling.

This scratch script deliberately imports the authoritative implementation without
editing it.  It never opens a socket and constructs only ``OfflineClient``
instances.  The variants test two ideas:

1. after obtaining a second bearing, return immediately so that the next
   measurement/clear task is selected by a fresh global DP;
2. after all 16 channels have been found early, optionally visit an unvisited
   certified shared-observation station when doing so shortens the current DP
   proxy route.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import geometry  # noqa: E402
from geometry import (  # noqa: E402
    GeometryError,
    guaranteed_shared_observation,
    localization_polygon,
    second_detection_candidates,
)
from offline_validation import (  # noqa: E402
    HiddenSource,
    NullLogger,
    OfflineClient,
    generate_sources,
)
from policy import (  # noqa: E402
    CHANNELS,
    MAX_SOURCE_COUNT,
    MIN_SOURCE_COUNT,
    PolicyError,
    RunSummary,
    SearchFirstPolicy,
)
from scheduler import optimal_candidate_route  # noqa: E402


class AuditClient(OfflineClient):
    """Offline client with an exact virtual-time decomposition."""

    def __init__(self, sources: list[HiddenSource], error_seed: int):
        super().__init__(sources, error_seed)
        self.movement_m = 0.0
        self.measure_fixed_s = 0.0
        self.switch_s = 0.0
        self.clear_fixed_s = 0.0

    def _move(self, x: float, y: float) -> np.ndarray:
        target = np.array([x, y], dtype=float)
        self.movement_m += float(np.linalg.norm(target - self._position))
        return super()._move(x, y)

    def measure(self, x: float, y: float, channel: int) -> dict:
        if channel != self._receiver_channel:
            self.switch_s += 1.0
        self.measure_fixed_s += 5.0
        return super().measure(x, y, channel)

    def clear(self, x: float, y: float, channel: int) -> dict:
        result = super().clear(x, y, channel)
        self.clear_fixed_s += 5.0 if result["clear_result"] == "success" else 3.0
        return result


class OneActionRollingPolicy(SearchFirstPolicy):
    """Acquire one second bearing and then hand control back to global DP."""

    name = "scratch_one_action_global_rolling"

    def _candidate_map(
        self, channels: set[int]
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        result = super()._candidate_map(channels)
        # A guaranteed candidate should never be no-signal in the stated model.
        # Still avoid selecting the same point again if an anomalous response was
        # returned.  Duplicating the remaining point preserves the scheduler API.
        for channel in channels:
            if len(self.observations[channel]) != 1:
                continue
            negatives = self.no_signal_points[channel]
            if not negatives:
                continue
            candidates = list(second_detection_candidates(self.observations[channel][0]))
            unused = [
                point
                for point in candidates
                if not any(float(np.linalg.norm(point - q)) <= 1.0e-7 for q in negatives)
            ]
            if not unused:
                raise PolicyError(f"频道 {channel} 的两个保证候选点均返回无信号")
            if len(unused) == 1:
                result[channel] = (unused[0].copy(), unused[0].copy())
        return result

    def _service_channel(
        self, channel: int, preferred_candidate_index: int | None = None
    ) -> None:
        observations = self.observations[channel]
        if len(observations) >= 2:
            # A localized task's next action remains its conservative clear action.
            return super()._service_channel(channel, preferred_candidate_index)

        candidates = second_detection_candidates(observations[0])
        if preferred_candidate_index not in (None, 0, 1):
            raise ValueError("第二检测点候选编号只能是0或1")
        if preferred_candidate_index is None:
            candidate_index = min(
                range(2),
                key=lambda index: float(
                    np.linalg.norm(candidates[index] - self.state.position)
                ),
            )
        else:
            candidate_index = preferred_candidate_index
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
                phase="second_observation_one_action",
            )
            return
        self._record_direction(channel, candidate, response)
        # Validate now, but deliberately do not move to the clear point.  This is
        # the key ablation: all currently known measurement and clear tasks are
        # re-ranked by the next global rolling plan.
        try:
            localization_polygon(observations, self.no_signal_points[channel])
        except GeometryError as error:
            raise PolicyError(f"频道 {channel} 的第二方位未形成定位区域: {error}") from error


class Post16SharedMixin:
    """Greedily add only DP-beneficial unvisited shared-observation stations."""

    post16_station_count: int
    post16_shared_count: int

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.post16_station_count = 0
        self.post16_shared_count = 0

    def _scan_coverage_stations(self) -> None:
        super()._scan_coverage_stations()
        if (
            len(self.discovered) == MAX_SOURCE_COUNT
            and self.stations_visited < len(self.coverage_points)
        ):
            self._collect_dp_beneficial_post16_stations()

    def _collect_dp_beneficial_post16_stations(self) -> None:
        remaining = set(range(self.stations_visited, len(self.coverage_points)))
        while remaining:
            pending = set(self.observations) - self.cleared
            one_bearing = {
                channel for channel in pending if len(self.observations[channel]) == 1
            }
            if not one_bearing:
                return
            baseline = optimal_candidate_route(
                self.state.position,
                self._candidate_map(pending),
                terminal=None,
            ).distance_m

            choices: list[tuple[float, int, set[int], float]] = []
            for station_index in sorted(remaining):
                station = self.coverage_points[station_index]
                covered = {
                    channel
                    for channel in one_bearing
                    if guaranteed_shared_observation(
                        self.observations[channel][0], station
                    )
                }
                if not covered:
                    continue
                reduced = pending - covered
                candidates = self._candidate_map(reduced)
                # Dummy task represents one physical station, regardless of how
                # many channel measurements are taken there.
                dummy = 100 + station_index
                candidates[dummy] = (station.copy(), station.copy())
                proxy = optimal_candidate_route(
                    self.state.position,
                    candidates,
                    terminal=None,
                ).distance_m
                choices.append((proxy, station_index, covered, baseline - proxy))
            if not choices:
                return
            proxy, station_index, covered, saving = min(choices, key=lambda item: item[0])
            # Fixed measurement time is unchanged: each covered channel needs one
            # second bearing with or without batching.  Require strict travel gain.
            if saving <= 1.0e-7:
                return
            station = self.coverage_points[station_index]
            before_shared = self.shared_observation_count
            candidates = sorted(covered)
            if self.state.receiver_channel in candidates:
                candidates.remove(self.state.receiver_channel)
                candidates.insert(0, self.state.receiver_channel)
            for channel in candidates:
                response = self._measure(station, channel)
                result = response["measure_result"]
                if result == "direction":
                    self._record_direction(channel, station, response)
                    self.shared_observation_count += 1
                elif result == "near":
                    self._clear_near(channel, station)
                else:
                    self._record_no_signal(
                        channel,
                        station,
                        phase="post16_shared_observation",
                    )
            self.post16_station_count += 1
            self.post16_shared_count += self.shared_observation_count - before_shared
            remaining.remove(station_index)


class Post16SharedPolicy(Post16SharedMixin, SearchFirstPolicy):
    name = "scratch_post16_shared"


class OneActionPost16Policy(Post16SharedMixin, OneActionRollingPolicy):
    name = "scratch_one_action_plus_post16"


@dataclass(frozen=True)
class Scenario:
    sources: list[HiddenSource]
    error_seed: int


def generate_scenarios(count: int, seed: int) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    return [
        Scenario(generate_sources(rng), int(rng.integers(0, 2**31 - 1)))
        for _ in range(count)
    ]


POLICIES = {
    "baseline": SearchFirstPolicy,
    "one_action": OneActionRollingPolicy,
    "post16": Post16SharedPolicy,
    "one_action_post16": OneActionPost16Policy,
}


def run_policy(policy_type: type[SearchFirstPolicy], scenario: Scenario) -> dict[str, Any]:
    client = AuditClient(copy.deepcopy(scenario.sources), scenario.error_seed)
    policy = policy_type(
        client,  # type: ignore[arg-type]
        NullLogger(),  # type: ignore[arg-type]
        {"remaining_real_duration_s": 1200, "max_virtual_duration_s": 360000},
        opportunistic_during_search=True,
        shared_observations=True,
        origin_information_scan=True,
    )
    summary = policy.run()
    if client.cleared_count != len(scenario.sources) or not summary.search_certified:
        raise RuntimeError("offline safety/completion assertion failed")
    components = (
        client.movement_m / 5.0
        + client.measure_fixed_s
        + client.switch_s
        + client.clear_fixed_s
    )
    if not math.isclose(components, summary.final_virtual_time_s, abs_tol=1.0e-6):
        raise RuntimeError("virtual-time decomposition mismatch")
    return {
        "time": summary.final_virtual_time_s,
        "sources": len(scenario.sources),
        "movement_m": client.movement_m,
        "measure_s": client.measure_fixed_s,
        "switch_s": client.switch_s,
        "clear_s": client.clear_fixed_s,
        "measures": summary.measure_count,
        "clears": summary.clear_attempt_count,
        "stations": summary.stations_visited,
        "post16_stations": int(getattr(policy, "post16_station_count", 0)),
        "post16_shared": int(getattr(policy, "post16_shared_count", 0)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = np.asarray([row["time"] for row in rows], dtype=float)
    return {
        "successes": len(rows),
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
        "p90_s": float(np.percentile(times, 90)),
        "max_s": float(np.max(times)),
        "mean_per_source_s": float(
            np.mean([row["time"] / row["sources"] for row in rows])
        ),
        "mean_movement_m": float(np.mean([row["movement_m"] for row in rows])),
        "mean_measure_s": float(np.mean([row["measure_s"] for row in rows])),
        "mean_switch_s": float(np.mean([row["switch_s"] for row in rows])),
        "mean_clear_s": float(np.mean([row["clear_s"] for row in rows])),
        "mean_measure_count": float(np.mean([row["measures"] for row in rows])),
        "mean_clear_count": float(np.mean([row["clears"] for row in rows])),
        "mean_post16_stations": float(
            np.mean([row["post16_stations"] for row in rows])
        ),
        "mean_post16_shared": float(
            np.mean([row["post16_shared"] for row in rows])
        ),
    }


def paired_summary(
    baseline: list[dict[str, Any]], variant: list[dict[str, Any]]
) -> dict[str, Any]:
    deltas = np.asarray(
        [new["time"] - old["time"] for old, new in zip(baseline, variant)],
        dtype=float,
    )
    return {
        "mean_delta_s": float(np.mean(deltas)),
        "median_delta_s": float(np.median(deltas)),
        "p90_delta_s": float(np.percentile(deltas, 90)),
        "min_delta_s": float(np.min(deltas)),
        "max_delta_s": float(np.max(deltas)),
        "wins": int(np.sum(deltas < -1.0e-7)),
        "ties": int(np.sum(np.abs(deltas) <= 1.0e-7)),
        "losses": int(np.sum(deltas > 1.0e-7)),
        "mean_reduction_percent": float(
            100.0
            * np.mean(
                [
                    (old["time"] - new["time"]) / old["time"]
                    for old, new in zip(baseline, variant)
                ]
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--forward", type=float, default=520.0)
    parser.add_argument("--lateral", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.scenarios <= 0:
        raise SystemExit("--scenarios must be positive")
    # Scratch-only parameter choice, identically applied to every policy.
    geometry.SECOND_FORWARD_M = arguments.forward
    geometry.SECOND_LATERAL_M = arguments.lateral

    scenarios = generate_scenarios(arguments.scenarios, arguments.seed)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in POLICIES}
    failures: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        for name, policy_type in POLICIES.items():
            try:
                rows[name].append(run_policy(policy_type, scenario))
            except Exception as error:  # collect all offline failures
                failures.append(
                    {
                        "scenario": index,
                        "policy": name,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise

    result = {
        "mode": "offline_no_network",
        "seed": arguments.seed,
        "scenarios": arguments.scenarios,
        "second_forward_m": arguments.forward,
        "second_lateral_m": arguments.lateral,
        "failures": failures,
        "summaries": {name: summarize(data) for name, data in rows.items()},
        "paired_vs_baseline": {
            name: paired_summary(rows["baseline"], data)
            for name, data in rows.items()
            if name != "baseline"
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
