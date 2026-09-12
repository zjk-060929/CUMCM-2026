"""只验证第二问的固定闭式解析点。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from q1_localization import (
    LocalizationError,
    Measurement,
    build_bearing_halfplanes,
    point_inside,
    solve_localization,
)
from q2_theory import (
    ANGLE_ERROR_DEG,
    MAX_RECEPTION_RADIUS_M,
    MIN_RECEPTION_RADIUS_M,
    approximate_diameter,
    derive_solution,
)


HERE = Path(__file__).resolve().parent
DOG_SPEED_M_PER_S = 5.0
MEASUREMENT_TIME_S = 5.0
NEAR_RADIUS_M = 5.0


def _circle_from_three_points(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """返回三点外接圆；三点共线时返回 ``None``。"""

    matrix = 2.0 * np.array([second - first, third - first], dtype=float)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= 1.0e-10:
        return None
    right = np.array(
        [
            float(np.dot(second, second) - np.dot(first, first)),
            float(np.dot(third, third) - np.dot(first, first)),
        ]
    )
    center = np.linalg.solve(matrix, right)
    radius = float(np.linalg.norm(center - first))
    return center, radius


def minimum_enclosing_circle(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """求凸多边形顶点集的精确最小包围圆。

    两次测向的交集只有少量顶点，因此枚举由两点或三点确定的候选圆
    比随机算法更透明，并且兼容点、线段和凸多边形。
    """

    points = np.asarray(vertices, dtype=float)
    if len(points) == 0:
        raise ValueError("顶点集合不能为空")
    tolerance = 1.0e-8
    best_center = points[0].copy()
    best_radius = math.inf

    def consider(center: np.ndarray, radius: float) -> None:
        nonlocal best_center, best_radius
        distances = np.linalg.norm(points - center, axis=1)
        if radius < best_radius and bool(np.all(distances <= radius + tolerance)):
            best_center = center.copy()
            best_radius = radius

    for point in points:
        consider(point, 0.0)
    for first_index in range(len(points)):
        for second_index in range(first_index + 1, len(points)):
            center = (points[first_index] + points[second_index]) / 2.0
            radius = float(
                np.linalg.norm(points[first_index] - points[second_index]) / 2.0
            )
            consider(center, radius)
    for first_index in range(len(points)):
        for second_index in range(first_index + 1, len(points)):
            for third_index in range(second_index + 1, len(points)):
                circle = _circle_from_three_points(
                    points[first_index],
                    points[second_index],
                    points[third_index],
                )
                if circle is not None:
                    consider(*circle)

    if not math.isfinite(best_radius):
        raise RuntimeError("无法构造定位多边形的最小包围圆")
    return best_center, best_radius


@dataclass(frozen=True)
class Scenario:
    source_x_m: float
    source_y_m: float
    reception_radius_m: float
    error_field_seed: int

    @property
    def source(self) -> np.ndarray:
        return np.array([self.source_x_m, self.source_y_m], dtype=float)


def deterministic_location_error(field_seed: int, point: np.ndarray) -> float:
    """同一地点重复测量不变、不同地点变化的有界误差场。"""

    key = f"{field_seed}:{point[0]:.6f}:{point[1]:.6f}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / float(2**64 - 1)
    return ANGLE_ERROR_DEG * (2.0 * unit - 1.0)


def bearing_deg(origin: np.ndarray, target: np.ndarray) -> float:
    vector = target - origin
    return math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360.0


def _exact_result(
    second_point: np.ndarray,
    source: np.ndarray,
    first_measured_bearing_deg: float,
    second_error_deg: float,
):
    """按给定两次读数构造纯角度交会区域。"""

    true_second_bearing = bearing_deg(second_point, source)
    measured_second_bearing = (true_second_bearing + second_error_deg) % 360.0
    result = solve_localization(
        [
            Measurement(0.0, 0.0, first_measured_bearing_deg),
            Measurement(
                float(second_point[0]),
                float(second_point[1]),
                measured_second_bearing,
            ),
        ],
        ANGLE_ERROR_DEG,
    )
    _, mec_radius = minimum_enclosing_circle(result.vertices)
    return result, true_second_bearing, measured_second_bearing, mec_radius


def deterministic_checks() -> dict[str, object]:
    """返回四个可复算的确定性检验，不用随机抽样替代边界判断。"""

    solution = derive_solution()
    current = np.array([solution.forward_m, solution.lateral_abs_m])

    nominal_source = np.array([MAX_RECEPTION_RADIUS_M, 0.0])
    nominal_result, nominal_true_bearing, nominal_reading, nominal_mec = (
        _exact_result(current, nominal_source, 0.0, 0.0)
    )

    boundary_source = MAX_RECEPTION_RADIUS_M * np.array(
        [
            math.cos(math.radians(ANGLE_ERROR_DEG)),
            math.sin(math.radians(ANGLE_ERROR_DEG)),
        ]
    )
    boundary_result, boundary_true_bearing, boundary_reading, boundary_mec = (
        _exact_result(
            current,
            boundary_source,
            0.0,
            ANGLE_ERROR_DEG,
        )
    )

    near_source = np.array([10.0, 0.0])
    near_result, near_true_bearing, near_reading, near_mec = _exact_result(
        current,
        near_source,
        0.0,
        0.0,
    )

    forty_five = 1000.0 * np.array(
        [math.cos(math.radians(45.0)), math.sin(math.radians(45.0))]
    )
    forty_five_result, _, _, _ = _exact_result(
        forty_five,
        nominal_source,
        0.0,
        0.0,
    )

    return {
        "nominal_r1500_zero_error": {
            "exact_diameter_m": nominal_result.diameter,
            "first_order_diameter_m": solution.nominal_worst_diameter_m,
            "mec_radius_m": nominal_mec,
            "vertex_count": len(nominal_result.vertices),
            "second_true_bearing_deg": nominal_true_bearing,
            "second_reading_deg": nominal_reading,
            "linearization_relative_error": abs(
                nominal_result.diameter - solution.nominal_worst_diameter_m
            )
            / nominal_result.diameter,
        },
        "full_precision_error_boundary_counterexample": {
            "source_x_m": float(boundary_source[0]),
            "source_y_m": float(boundary_source[1]),
            "first_true_bearing_deg": ANGLE_ERROR_DEG,
            "first_error_deg": -ANGLE_ERROR_DEG,
            "first_reading_deg": 0.0,
            "second_distance_m": float(np.linalg.norm(boundary_source - current)),
            "second_true_bearing_deg": boundary_true_bearing,
            "second_error_deg": ANGLE_ERROR_DEG,
            "second_reading_deg": boundary_reading,
            "exact_diameter_m": boundary_result.diameter,
            "mec_radius_m": boundary_mec,
            "vertex_count": len(boundary_result.vertices),
            "interpretation": "合法边界场景值；证明名义一阶值不是全误差上界，但未证明这是全局最大值",
        },
        "near_source_triangle": {
            "source_x_m": 10.0,
            "source_y_m": 0.0,
            "second_true_bearing_deg": near_true_bearing,
            "second_reading_deg": near_reading,
            "exact_diameter_m": near_result.diameter,
            "first_order_diameter_m": approximate_diameter(
                10.0,
                solution.forward_m,
                solution.lateral_abs_m,
            ),
            "mec_radius_m": near_mec,
            "vertex_count": len(near_result.vertices),
        },
        "same_nominal_scene_candidate_check": {
            "source_x_m": 1500.0,
            "source_y_m": 0.0,
            "current_heading_offset_deg": math.degrees(
                math.atan2(solution.lateral_abs_m, solution.forward_m)
            ),
            "current_exact_diameter_m": nominal_result.diameter,
            "forty_five_heading_offset_deg": 45.0,
            "forty_five_exact_diameter_m": forty_five_result.diameter,
            "both_travel_m": 1000.0,
            "both_move_and_measurement_time_s": 205.0,
            "interpretation": "只支持当前35.10度方案在该名义场景下优于45度方案",
        },
    }


def generate_scenarios(count: int, seed: int) -> list[Scenario]:
    """在首次检测已收到信号的条件下生成独立验证场景。"""

    random = np.random.default_rng(seed)
    scenarios: list[Scenario] = []
    origin = np.array([0.0, 0.0])
    for _ in range(count):
        reception_radius = float(
            random.uniform(MIN_RECEPTION_RADIUS_M, MAX_RECEPTION_RADIUS_M)
        )
        radius = math.sqrt(
            NEAR_RADIUS_M**2
            + float(random.random())
            * (reception_radius**2 - NEAR_RADIUS_M**2)
        )
        field_seed = int(random.integers(0, 2**31 - 1))
        first_error = deterministic_location_error(field_seed, origin)
        # 旋转场景，使首次测得示向度恰为 0 度。
        true_angle = math.radians(-first_error)
        source = radius * np.array([math.cos(true_angle), math.sin(true_angle)])
        scenarios.append(
            Scenario(
                source_x_m=float(source[0]),
                source_y_m=float(source[1]),
                reception_radius_m=reception_radius,
                error_field_seed=field_seed,
            )
        )
    return scenarios


def acute_crossing_angle_deg(
    first_point: np.ndarray,
    second_point: np.ndarray,
    source: np.ndarray,
) -> float:
    first_vector = source - first_point
    second_vector = source - second_point
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    cosine = abs(float(np.dot(first_vector, second_vector))) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def quantile_record(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def run_validation(count: int, seed: int) -> dict[str, object]:
    solution = derive_solution()
    first_point = np.array([0.0, 0.0])
    second_point = np.array([solution.forward_m, solution.lateral_abs_m])
    scenarios = generate_scenarios(count, seed)
    records: list[dict[str, object]] = []

    for index, scenario in enumerate(scenarios):
        source = scenario.source
        second_distance = float(np.linalg.norm(source - second_point))
        reception_retained = second_distance <= scenario.reception_radius_m + 1.0e-7
        crossing_angle = acute_crossing_angle_deg(
            first_point, second_point, source
        )
        record: dict[str, object] = {
            "scenario": index,
            "source_distance_from_first_m": float(np.linalg.norm(source)),
            "source_distance_from_second_m": second_distance,
            "reception_radius_m": scenario.reception_radius_m,
            "reception_retained": reception_retained,
            "crossing_angle_deg": crossing_angle,
            "observation": "direction",
            "diameter_m": math.nan,
            "mec_radius_m": math.nan,
            "mec_center_x_m": math.nan,
            "mec_center_y_m": math.nan,
            "diameter_circle_covers": False,
            "true_source_inside_bearing_region": False,
        }

        if not reception_retained:
            record["observation"] = "no_signal"
            records.append(record)
            continue
        if second_distance <= NEAR_RADIUS_M:
            record.update(
                {
                    "observation": "near",
                    "diameter_m": 0.0,
                    "mec_radius_m": 0.0,
                    "mec_center_x_m": float(source[0]),
                    "mec_center_y_m": float(source[1]),
                    "diameter_circle_covers": True,
                    "true_source_inside_bearing_region": True,
                }
            )
            records.append(record)
            continue

        true_second_bearing = bearing_deg(second_point, source)
        second_error = deterministic_location_error(
            scenario.error_field_seed, second_point
        )
        measured_second_bearing = (true_second_bearing + second_error) % 360.0
        measurements = [
            Measurement(0.0, 0.0, 0.0),
            Measurement(
                float(second_point[0]),
                float(second_point[1]),
                measured_second_bearing,
            ),
        ]
        lines = build_bearing_halfplanes(measurements, ANGLE_ERROR_DEG)
        record["true_source_inside_bearing_region"] = all(
            point_inside(line, source, eps=1.0e-8) for line in lines
        )
        try:
            result = solve_localization(measurements, ANGLE_ERROR_DEG)
        except LocalizationError:
            record["observation"] = "invalid_intersection"
            records.append(record)
            continue
        mec_center, mec_radius = minimum_enclosing_circle(result.vertices)
        record["diameter_m"] = result.diameter
        record["mec_radius_m"] = mec_radius
        record["mec_center_x_m"] = float(mec_center[0])
        record["mec_center_y_m"] = float(mec_center[1])
        record["diameter_circle_covers"] = result.circle_covers_polygon
        records.append(record)

    columns = list(records[0])
    with (HERE / "simulation_samples.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    reception = np.array([bool(item["reception_retained"]) for item in records])
    valid_records = [
        item for item in records if math.isfinite(float(item["diameter_m"]))
    ]
    diameters = np.array([float(item["diameter_m"]) for item in valid_records])
    mec_radii = np.array([float(item["mec_radius_m"]) for item in valid_records])
    angles = np.array([float(item["crossing_angle_deg"]) for item in valid_records])
    covers = np.array(
        [bool(item["diameter_circle_covers"]) for item in valid_records]
    )
    inside = np.array(
        [bool(item["true_source_inside_bearing_region"]) for item in records]
    )
    standard_error = float(np.std(diameters, ddof=1) / math.sqrt(len(diameters)))

    nominal_source = np.array([MAX_RECEPTION_RADIUS_M, 0.0])
    nominal_second_bearing = bearing_deg(second_point, nominal_source)
    nominal_result = solve_localization(
        [
            Measurement(0.0, 0.0, 0.0),
            Measurement(
                float(second_point[0]),
                float(second_point[1]),
                nominal_second_bearing,
            ),
        ],
        ANGLE_ERROR_DEG,
    )
    _, nominal_mec_radius = minimum_enclosing_circle(nominal_result.vertices)
    nominal_relative_error = abs(
        nominal_result.diameter - solution.nominal_worst_diameter_m
    ) / nominal_result.diameter
    checks = deterministic_checks()
    summary = {
        "purpose": "离线验证固定闭式解析点；随机样本不参与选点，确定性检验不等同于全局穷举",
        "scenario_count": count,
        "seed": seed,
        "criterion": "解析充分安全域内，首次中心线名义源的一阶最坏径向定位直径最小",
        "local_candidate": {
            "a_m": solution.forward_m,
            "b_m": solution.lateral_abs_m,
            "travel_m": solution.travel_m,
        },
        "theoretical_nominal_worst_diameter_m": (
            solution.nominal_worst_diameter_m
        ),
        "exact_hpi_nominal_check": {
            "source_distance_m": MAX_RECEPTION_RADIUS_M,
            "diameter_m": nominal_result.diameter,
            "mec_radius_m": nominal_mec_radius,
            "linearization_relative_error": nominal_relative_error,
        },
        "deterministic_checks": checks,
        "move_and_second_measurement_time_s": (
            solution.travel_m / DOG_SPEED_M_PER_S + MEASUREMENT_TIME_S
        ),
        "reception_retained_rate": float(np.mean(reception)),
        "valid_localization_rate": len(valid_records) / count,
        "true_source_containment_rate": float(np.mean(inside)),
        "diameter_m": quantile_record(diameters),
        "mec_radius_m": quantile_record(mec_radii),
        "mean_diameter_95ci_m": [
            float(np.mean(diameters) - 1.96 * standard_error),
            float(np.mean(diameters) + 1.96 * standard_error),
        ],
        "crossing_angle_deg": quantile_record(angles),
        "diameter_circle_coverage_rate": float(np.mean(covers)),
        "diameter_at_most_40m_rate": float(np.mean(diameters <= 40.0)),
        "diameter_endpoint_circle_within_20m_and_covers_rate": float(
            np.mean(covers & (diameters <= 40.0))
        ),
        "mec_within_20m_rate": float(np.mean(mec_radii <= 20.0)),
        "observation_counts": {
            name: sum(item["observation"] == name for item in records)
            for name in ("direction", "near", "no_signal", "invalid_intersection")
        },
    }
    with (HERE / "simulation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="第二问解析点的纯离线验证")
    parser.add_argument("--scenarios", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="只打印确定性核验，不生成或覆盖随机样本文件",
    )
    arguments = parser.parse_args()
    if arguments.deterministic_only:
        print(json.dumps(deterministic_checks(), ensure_ascii=False, indent=2))
    else:
        run_validation(arguments.scenarios, arguments.seed)


if __name__ == "__main__":
    main()
