"""B 题定位几何：半平面交、旋转卡壳与直径圆覆盖判定。

坐标约定与题面一致：x 轴正向为正东，y 轴正向为正北，方位角从
x 轴正向逆时针增加。输入角度单位为度，内部统一转换为弧度。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


EPS = 1.0e-10


class LocalizationError(RuntimeError):
    """定位区域为空或无界时抛出的异常。"""


@dataclass(frozen=True)
class Measurement:
    """一次测向记录。"""

    x: float
    y: float
    bearing_deg: float

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass(frozen=True)
class DirectedLine:
    """有向直线；可行半平面规定在方向向量 d 的左侧。"""

    p: np.ndarray
    d: np.ndarray
    angle: float


@dataclass(frozen=True)
class DiameterResult:
    diameter: float
    point_a: np.ndarray
    point_b: np.ndarray


@dataclass(frozen=True)
class LocalizationResult:
    vertices: np.ndarray
    diameter: float
    point_a: np.ndarray
    point_b: np.ndarray
    circle_center: np.ndarray
    circle_radius: float
    circle_covers_polygon: bool
    vertex_distances_to_center: np.ndarray


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """二维叉积。"""

    return float(a[0] * b[1] - a[1] * b[0])


def make_line(point: np.ndarray, direction: np.ndarray) -> DirectedLine:
    """构造单位化的有向直线。"""

    p = np.asarray(point, dtype=float)
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    if norm <= EPS:
        raise ValueError("直线方向向量不能为零向量")
    d = d / norm
    angle = math.atan2(float(d[1]), float(d[0])) % (2.0 * math.pi)
    return DirectedLine(p=p, d=d, angle=angle)


def point_inside(line: DirectedLine, point: np.ndarray, eps: float = EPS) -> bool:
    """判断点是否位于有向直线左侧的闭半平面中。"""

    return cross2(line.d, np.asarray(point, dtype=float) - line.p) >= -eps


def line_intersection(
    first: DirectedLine,
    second: DirectedLine,
    eps: float = EPS,
) -> np.ndarray | None:
    """求两条非平行有向直线的交点；平行时返回 None。"""

    denominator = cross2(first.d, second.d)
    if abs(denominator) <= eps:
        return None
    parameter = cross2(second.p - first.p, second.d) / denominator
    return first.p + parameter * first.d


def build_bearing_halfplanes(
    measurements: Sequence[Measurement],
    error_deg: float = 1.0,
) -> list[DirectedLine]:
    """将每次带有正负 error_deg 误差的示向度转换为两个半平面。"""

    if not measurements:
        raise ValueError("至少需要一条测向记录")
    if not 0.0 < error_deg < 90.0:
        raise ValueError("角度误差应位于 (0, 90) 度内")

    lines: list[DirectedLine] = []
    for measurement in measurements:
        lower = math.radians(measurement.bearing_deg - error_deg)
        upper = math.radians(measurement.bearing_deg + error_deg)

        lower_direction = np.array([math.cos(lower), math.sin(lower)])
        upper_direction = np.array([math.cos(upper), math.sin(upper)])

        # 下边界以原方向定向，可行域位于其左侧。
        lines.append(make_line(measurement.point, lower_direction))
        # 上边界反向定向后，可行域同样位于其左侧。
        lines.append(make_line(measurement.point, -upper_direction))

    return lines


def _remove_same_direction_parallel_lines(
    lines: Sequence[DirectedLine],
    eps: float,
) -> list[DirectedLine]:
    """同方向平行边界中只保留限制最严格的一条。"""

    ordered = sorted(lines, key=lambda line: line.angle)
    unique: list[DirectedLine] = []

    for line in ordered:
        if unique:
            previous = unique[-1]
            parallel = abs(cross2(previous.d, line.d)) <= eps
            same_direction = float(np.dot(previous.d, line.d)) > 0.0
            if parallel and same_direction:
                # 若 line.p 位于 previous 的严格左侧，新半平面更严格。
                if cross2(previous.d, line.p - previous.p) > eps:
                    unique[-1] = line
                continue
        unique.append(line)

    return unique


def _required_intersection(
    first: DirectedLine,
    second: DirectedLine,
    eps: float,
) -> np.ndarray:
    point = line_intersection(first, second, eps)
    if point is None:
        raise LocalizationError("定位区域为空、无界，或含有无法闭合的平行边界")
    return point


def signed_polygon_area(vertices: np.ndarray) -> float:
    """多边形有向面积。"""

    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _has_nonzero_recession_direction(
    lines: Sequence[DirectedLine],
    eps: float,
) -> bool:
    """判断半平面交是否存在非零衰退方向，即是否无界。"""

    if not lines:
        return True
    candidates = [
        direction
        for line in lines
        for direction in (line.d, -line.d)
    ]
    candidates.extend(
        np.array(direction, dtype=float)
        for direction in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
    )
    return any(
        all(cross2(line.d, direction) >= -100.0 * eps for line in lines)
        for direction in candidates
    )


def _deduplicate_points(
    points: Sequence[np.ndarray],
    tolerance: float,
) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if all(float(np.linalg.norm(point - saved)) > tolerance for saved in unique):
            unique.append(point.copy())
    return unique


def _convex_hull(points: Sequence[np.ndarray], eps: float) -> np.ndarray:
    """Andrew 单调链凸包；点、线段和多边形均原样支持。"""

    ordered = sorted(
        (np.asarray(point, dtype=float) for point in points),
        key=lambda point: (float(point[0]), float(point[1])),
    )
    if len(ordered) <= 1:
        return np.asarray(ordered, dtype=float)

    def build_half(sequence: Sequence[np.ndarray]) -> list[np.ndarray]:
        half: list[np.ndarray] = []
        for point in sequence:
            while len(half) >= 2 and cross2(
                half[-1] - half[-2], point - half[-1]
            ) <= eps:
                half.pop()
            half.append(point)
        return half

    lower = build_half(ordered)
    upper = build_half(list(reversed(ordered)))
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def half_plane_intersection(
    lines: Sequence[DirectedLine],
    eps: float = EPS,
) -> np.ndarray:
    """求有界闭半平面交，返回点、线段或逆时针凸多边形。

    实现枚举每对非平行边界的交点，并保留满足全部约束的点，随后
    构造凸包。该写法显式保留退化为单点或线段的合法结果，时间复杂度
    为 ``O(m^3)``；本题每次只有少量测向边界，计算量可以忽略。
    """

    if len(lines) < 2:
        raise LocalizationError("半平面数量不足，定位区域无界")

    ordered = _remove_same_direction_parallel_lines(lines, eps)
    if _has_nonzero_recession_direction(ordered, eps):
        raise LocalizationError("测向信息不足：半平面交无界")

    feasible_intersections: list[np.ndarray] = []
    feasibility_tolerance = 100.0 * eps
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            point = line_intersection(first, second, eps)
            if point is None:
                continue
            if all(
                point_inside(line, point, feasibility_tolerance)
                for line in ordered
            ):
                feasible_intersections.append(point)

    points = _deduplicate_points(
        feasible_intersections,
        tolerance=feasibility_tolerance,
    )
    if not points:
        raise LocalizationError("半平面交为空")

    hull = _convex_hull(points, feasibility_tolerance)
    if len(hull) >= 3 and signed_polygon_area(hull) < 0.0:
        hull = hull[::-1].copy()
    return hull


def _distance_squared(first: np.ndarray, second: np.ndarray) -> float:
    difference = first - second
    return float(np.dot(difference, difference))


def _double_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return abs(cross2(b - a, c - a))


def rotating_calipers_diameter(
    vertices: np.ndarray,
    eps: float = EPS,
) -> DiameterResult:
    """使用旋转卡壳求逆时针凸多边形的直径及最远点对。"""

    polygon = np.asarray(vertices, dtype=float)
    count = len(polygon)
    if count == 0:
        raise ValueError("顶点集合不能为空")
    if count == 1:
        return DiameterResult(0.0, polygon[0].copy(), polygon[0].copy())
    if count == 2:
        return DiameterResult(
            math.sqrt(_distance_squared(polygon[0], polygon[1])),
            polygon[0].copy(),
            polygon[1].copy(),
        )

    if signed_polygon_area(polygon) < 0.0:
        polygon = polygon[::-1].copy()

    opposite = 1
    best_distance_squared = -1.0
    best_pair = (0, 1)

    def update(first_index: int, second_index: int) -> None:
        nonlocal best_distance_squared, best_pair
        value = _distance_squared(polygon[first_index], polygon[second_index])
        if value > best_distance_squared:
            best_distance_squared = value
            best_pair = (first_index, second_index)

    for index in range(count):
        next_index = (index + 1) % count
        advances = 0
        while advances < count:
            next_opposite = (opposite + 1) % count
            current_area = _double_triangle_area(
                polygon[index], polygon[next_index], polygon[opposite]
            )
            next_area = _double_triangle_area(
                polygon[index], polygon[next_index], polygon[next_opposite]
            )
            if next_area <= current_area + eps:
                break
            opposite = next_opposite
            advances += 1

        update(index, opposite)
        update(next_index, opposite)

        next_opposite = (opposite + 1) % count
        current_area = _double_triangle_area(
            polygon[index], polygon[next_index], polygon[opposite]
        )
        next_area = _double_triangle_area(
            polygon[index], polygon[next_index], polygon[next_opposite]
        )
        if abs(next_area - current_area) <= eps:
            update(index, next_opposite)
            update(next_index, next_opposite)

    first_index, second_index = best_pair
    return DiameterResult(
        diameter=math.sqrt(max(best_distance_squared, 0.0)),
        point_a=polygon[first_index].copy(),
        point_b=polygon[second_index].copy(),
    )


def diameter_circle_coverage(
    vertices: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    relative_tolerance: float = 1.0e-9,
) -> tuple[bool, np.ndarray, float, np.ndarray]:
    """判断以最远点对为直径的闭圆盘是否覆盖整个凸多边形。"""

    polygon = np.asarray(vertices, dtype=float)
    center = (np.asarray(point_a) + np.asarray(point_b)) / 2.0
    radius_squared = _distance_squared(point_a, point_b) / 4.0
    distances_squared = np.sum((polygon - center) ** 2, axis=1)
    tolerance = relative_tolerance * max(1.0, radius_squared)
    covered = bool(np.all(distances_squared <= radius_squared + tolerance))
    return covered, center, math.sqrt(radius_squared), np.sqrt(distances_squared)


def solve_localization(
    measurements: Sequence[Measurement],
    error_deg: float = 1.0,
) -> LocalizationResult:
    """求解第一问的完整计算流程。"""

    lines = build_bearing_halfplanes(measurements, error_deg)
    vertices = half_plane_intersection(lines)
    diameter = rotating_calipers_diameter(vertices)
    covered, center, radius, distances = diameter_circle_coverage(
        vertices, diameter.point_a, diameter.point_b
    )
    return LocalizationResult(
        vertices=vertices,
        diameter=diameter.diameter,
        point_a=diameter.point_a,
        point_b=diameter.point_b,
        circle_center=center,
        circle_radius=radius,
        circle_covers_polygon=covered,
        vertex_distances_to_center=distances,
    )


def _load_input(path: Path) -> tuple[list[Measurement], float]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    measurements = [
        Measurement(
            x=float(item["x"]),
            y=float(item["y"]),
            bearing_deg=float(item["bearing_deg"]),
        )
        for item in data["measurements"]
    ]
    return measurements, float(data.get("error_deg", 1.0))


def _result_as_dict(result: LocalizationResult) -> dict[str, object]:
    return {
        "vertices": np.round(result.vertices, 9).tolist(),
        "diameter_m": round(result.diameter, 9),
        "diameter_endpoints": {
            "A": np.round(result.point_a, 9).tolist(),
            "B": np.round(result.point_b, 9).tolist(),
        },
        "diameter_circle": {
            "center": np.round(result.circle_center, 9).tolist(),
            "radius_m": round(result.circle_radius, 9),
            "covers_polygon": result.circle_covers_polygon,
        },
        "vertex_distances_to_center_m": np.round(
            result.vertex_distances_to_center, 9
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B 题第一问：半平面交、旋转卡壳和直径圆覆盖判定"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="JSON 输入文件；省略时使用内置演示数据",
    )
    arguments = parser.parse_args()

    if arguments.input is None:
        measurements = [
            Measurement(0.0, 0.0, 0.0),
            Measurement(1000.0, 1000.0, 270.0),
        ]
        error_deg = 1.0
    else:
        measurements, error_deg = _load_input(arguments.input)

    try:
        result = solve_localization(measurements, error_deg)
    except LocalizationError as error:
        raise SystemExit(f"无法得到有界定位集合：{error}") from error

    print(json.dumps(_result_as_dict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
