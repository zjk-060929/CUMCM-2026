"""完全离线的随机场景回归测试；不会监听或连接任何网络端口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geometry import COVERAGE_LAYOUTS
from policy import SearchFirstPolicy


@dataclass(frozen=True)
class HiddenSource:
    channel: int
    point: np.ndarray
    reception_radius_m: float


class NullLogger:
    def write(self, event: str, **fields) -> None:
        del event, fields


class OfflineClient:
    """真值只用于生成接口反馈；策略对象无法读取本对象的私有源字典。"""

    worst_case_request_wait_s = 0.001

    def __init__(self, sources: list[HiddenSource], error_seed: int):
        self._sources = {source.channel: source for source in sources}
        self._cleared: set[int] = set()
        self._error_seed = error_seed
        self._position = np.zeros(2)
        self._receiver_channel = 1
        self._virtual_time_s = 0.0
        self.last_action_started_monotonic_s = time.monotonic()

    @property
    def cleared_count(self) -> int:
        return len(self._cleared)

    def _move(self, x: float, y: float) -> np.ndarray:
        target = np.array([x, y], dtype=float)
        self._virtual_time_s += float(np.linalg.norm(target - self._position)) / 5.0
        self._position = target
        return target

    def _bearing_error(self, channel: int, point: np.ndarray) -> float:
        # 直接从 0.01 度格点取值，保证落在 [-0.99, 0.99] 且有两位小数。
        key = (
            f"{self._error_seed}:{channel}:"
            f"{float(point[0]):.6f}:{float(point[1]):.6f}"
        ).encode("ascii")
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        return ((value % 199) - 99) / 100.0

    def measure(self, x: float, y: float, channel: int) -> dict:
        point = self._move(x, y)
        if channel != self._receiver_channel:
            self._virtual_time_s += 1.0
        self._receiver_channel = channel
        self._virtual_time_s += 5.0
        source = self._sources.get(channel)
        if source is None or channel in self._cleared:
            result = "no_signal"
            extra = {}
        else:
            distance = float(np.linalg.norm(source.point - point))
            if distance > source.reception_radius_m + 1.0e-9:
                result = "no_signal"
                extra = {}
            elif distance <= 5.0 + 1.0e-9:
                result = "near"
                extra = {}
            else:
                result = "direction"
                true_bearing = math.degrees(
                    math.atan2(
                        float(source.point[1] - point[1]),
                        float(source.point[0] - point[0]),
                    )
                )
                measured = (true_bearing + self._bearing_error(channel, point)) % 360.0
                extra = {"svd_deg": round(measured, 2) % 360.0}
        return {
            "accepted": True,
            "real_timestamp_ms": 0,
            "virtual_time_s": self._virtual_time_s,
            "measure_result": result,
            **extra,
        }

    def clear(self, x: float, y: float, channel: int) -> dict:
        point = self._move(x, y)
        source = self._sources.get(channel)
        success = (
            source is not None
            and channel not in self._cleared
            and float(np.linalg.norm(source.point - point)) <= 20.0 + 1.0e-9
        )
        self._virtual_time_s += 5.0 if success else 3.0
        if success:
            self._cleared.add(channel)
        return {
            "accepted": True,
            "real_timestamp_ms": 0,
            "virtual_time_s": self._virtual_time_s,
            "clear_result": "success" if success else "no_target_in_range",
        }


def generate_sources(rng: np.random.Generator) -> list[HiddenSource]:
    count = int(rng.integers(10, 17))
    channels = rng.choice(np.arange(1, 21), size=count, replace=False)
    sources = []
    for channel in channels:
        radius = 1800.0 * math.sqrt(float(rng.random()))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        sources.append(
            HiddenSource(
                channel=int(channel),
                point=radius * np.array([math.cos(angle), math.sin(angle)]),
                reception_radius_m=float(rng.uniform(1000.0, 1500.0)),
            )
        )
    return sources


def validate(
    scenarios: int,
    seed: int,
    *,
    opportunistic_during_search: bool = True,
    coverage_layout: str = "heptagon_minimax",
    shared_observations: bool = True,
    origin_information_scan: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    virtual_times = []
    measures = []
    clear_attempts = []
    source_counts = []
    average_times_per_source = []
    no_signal_counts = []
    shared_observation_counts = []
    piggyback_observation_counts = []
    reactive_refinement_counts = []
    certified_absent_counts = []
    failures = []
    for scenario_index in range(scenarios):
        sources = generate_sources(rng)
        client = OfflineClient(
            sources,
            error_seed=int(rng.integers(0, 2**31 - 1)),
        )
        try:
            policy = SearchFirstPolicy(
                client,  # type: ignore[arg-type]
                NullLogger(),  # type: ignore[arg-type]
                {
                    "remaining_real_duration_s": 1200,
                    "max_virtual_duration_s": 360000,
                },
                opportunistic_during_search=opportunistic_during_search,
                coverage_layout=coverage_layout,
                shared_observations=shared_observations,
                origin_information_scan=origin_information_scan,
            )
            summary = policy.run()
            if client.cleared_count != len(sources):
                raise RuntimeError(
                    f"仅清除 {client.cleared_count}/{len(sources)} 个干扰源"
                )
            if not summary.search_certified:
                raise RuntimeError("策略结束但没有形成完整频道排查证书")
            virtual_times.append(summary.final_virtual_time_s)
            measures.append(summary.measure_count)
            clear_attempts.append(summary.clear_attempt_count)
            source_counts.append(len(sources))
            average_times_per_source.append(
                summary.final_virtual_time_s / len(sources)
            )
            no_signal_counts.append(summary.no_signal_measure_count)
            shared_observation_counts.append(summary.shared_observation_count)
            piggyback_observation_counts.append(summary.piggyback_observation_count)
            reactive_refinement_counts.append(summary.reactive_refinement_count)
            certified_absent_counts.append(len(summary.certified_absent_channels))
        except Exception as error:  # 离线回归需收集全部失败场景
            failures.append(
                {
                    "scenario_index": scenario_index,
                    "source_count": len(sources),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    return {
        "mode": "offline_no_network",
        "strategy": SearchFirstPolicy.name,
        "opportunistic_during_search": opportunistic_during_search,
        "coverage_layout": coverage_layout,
        "shared_observations": shared_observations,
        "origin_information_scan": origin_information_scan,
        "seed": seed,
        "scenarios": scenarios,
        "successful_scenarios": scenarios - len(failures),
        "failures": failures,
        "mean_virtual_time_s": float(np.mean(virtual_times)) if virtual_times else None,
        "median_virtual_time_s": float(np.median(virtual_times)) if virtual_times else None,
        "p90_virtual_time_s": float(np.percentile(virtual_times, 90)) if virtual_times else None,
        "p95_virtual_time_s": float(np.percentile(virtual_times, 95)) if virtual_times else None,
        "max_virtual_time_s": float(np.max(virtual_times)) if virtual_times else None,
        "mean_time_per_source_s": (
            float(np.mean(average_times_per_source))
            if average_times_per_source
            else None
        ),
        "mean_source_count": float(np.mean(source_counts)) if source_counts else None,
        "mean_measure_count": float(np.mean(measures)) if measures else None,
        "mean_clear_attempt_count": (
            float(np.mean(clear_attempts)) if clear_attempts else None
        ),
        "max_clear_attempt_count": int(np.max(clear_attempts)) if clear_attempts else None,
        "mean_no_signal_measure_count": (
            float(np.mean(no_signal_counts)) if no_signal_counts else None
        ),
        "mean_shared_observation_count": (
            float(np.mean(shared_observation_counts))
            if shared_observation_counts
            else None
        ),
        "mean_piggyback_observation_count": (
            float(np.mean(piggyback_observation_counts))
            if piggyback_observation_counts
            else None
        ),
        "mean_reactive_refinement_count": (
            float(np.mean(reactive_refinement_counts))
            if reactive_refinement_counts
            else None
        ),
        "mean_certified_absent_channel_count": (
            float(np.mean(certified_absent_counts))
            if certified_absent_counts
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="问题3客户端完全离线随机回归")
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：把离线汇总写入 JSON 文件",
    )
    parser.add_argument(
        "--coverage-layout",
        choices=COVERAGE_LAYOUTS,
        default="heptagon_minimax",
        help="七点保证性布局；默认使用路程较短的正七边形环",
    )
    parser.add_argument(
        "--disable-shared-observations",
        action="store_true",
        help="离线消融比较：禁用共用停靠点的安全补测",
    )
    parser.add_argument(
        "--disable-opportunistic",
        action="store_true",
        help="仅比较用：禁用搜索中机会式清除，但保留尾段滚动动态规划",
    )
    arguments = parser.parse_args()
    if arguments.scenarios <= 0:
        raise SystemExit("--scenarios 必须为正整数")
    result = validate(
        arguments.scenarios,
        arguments.seed,
        opportunistic_during_search=not arguments.disable_opportunistic,
        coverage_layout=arguments.coverage_layout,
        shared_observations=not arguments.disable_shared_observations,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
