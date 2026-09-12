"""B 题第一问：半平面交、旋转卡壳与直径圆覆盖判定。

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
ANGLE_EPS = 1.0e-14


class LocalizationError(RuntimeError):
    """无法返回有限定位区域；点和线段属于正常结果。"""

    status = "geometry_error"


class EmptyLocalizationError(LocalizationError):
    """全部测向约束不相容。"""

    status = "empty"


class UnboundedLocalizationError(LocalizationError):
    """角度交会区域非空，但无有限直径。"""

    status = "unbounded"


class NumericalGeometryError(LocalizationError):
    """浮点几何结果未通过约束复核，不误报为空集。"""

    status = "numerical_error"


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

    @property
    def region_type(self) -> str:
        return {1: "point", 2: "segment"}.get(len(self.vertices), "polygon")


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """二维叉积。"""

    return float(a[0] * b[1] - a[1] * b[0])


def make_line(point: np.ndarray, direction: np.ndarray) -> DirectedLine:
    """构造单位化的有向直线。"""

    p = np.asarray(point, dtype=float)
    d = np.asarray(direction, dtype=float)
    if p.shape != (2,) or d.shape != (2,) or not (np.isfinite(p).all() and np.isfinite(d).all()):
        raise ValueError("直线点和方向必须是含两个有限数值的向量")
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
    eps: float = ANGLE_EPS,
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
        if not all(math.isfinite(v) for v in (measurement.x, measurement.y, measurement.bearing_deg)):
            raise ValueError("检测点坐标和示向度必须是有限数值")
        bearing = measurement.bearing_deg % 360.0
        lower = math.radians(bearing - error_deg)
        upper = math.radians(bearing + error_deg)

        lower_direction = np.array([math.cos(lower), math.sin(lower)])
        upper_direction = np.array([math.cos(upper), math.sin(upper)])

        # 下边界以原方向定向，可行域位于其左侧。
        lines.append(make_line(measurement.point, lower_direction))
        # 上边界反向定向后，可行域同样位于其左侧。
        lines.append(make_line(measurement.point, -upper_direction))

    return lines


def _boundary_interval(
    boundary: DirectedLine,
    lines: Sequence[DirectedLine],
    eps: float,
) -> tuple[float, float] | None:
    """在 X=p+t*d 上，将全部半平面约束化为 t 的闭区间。"""

    lower, upper = -math.inf, math.inf
    for line in lines:
        offset = cross2(line.d, boundary.p - line.p)
        slope = cross2(line.d, boundary.d)
        # 方向向量均已单位化；角度容差与距离容差分开使用。
        if abs(slope) <= ANGLE_EPS:
            if offset < -eps:
                return None
            continue
        limit = -offset / slope
        if slope > 0.0:
            lower = max(lower, limit)
        else:
            upper = min(upper, limit)
        if lower > upper + eps:
            return None
    if lower > upper:
        # 在距离容差内相接的两个区间，合并为单点。
        lower = upper = lower / 2.0 + upper / 2.0
    return lower, upper


def signed_polygon_area(vertices: np.ndarray) -> float:
    """平移到局部原点后计算有向面积，降低大坐标相减损失。"""

    if len(vertices) < 3:
        return 0.0
    local = np.asarray(vertices, dtype=float) - vertices[0]
    x, y = local[:, 0], local[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _convex_hull(points: Sequence[np.ndarray], eps: float) -> np.ndarray:
    """单调链：返回逆时针极点，单点或线段分别返回 1 或 2 个点。"""

    ordered = sorted(points, key=lambda p: (float(p[0]), float(p[1])))
    unique: list[np.ndarray] = []
    for point in ordered:
        if not unique or np.linalg.norm(point - unique[-1]) > eps:
            unique.append(point)
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def chain(sequence: Sequence[np.ndarray]) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for point in sequence:
            while len(result) >= 2:
                edge = result[-1] - result[-2]
                # 叉积除以边长是到直线的有向距离，与 eps 单位一致。
                if cross2(edge, point - result[-2]) > eps * np.linalg.norm(edge):
                    break
                result.pop()
            result.append(point)
        return result

    lower = chain(unique)
    upper = chain(unique[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def half_plane_intersection(
    lines: Sequence[DirectedLine],
    eps: float = EPS,
) -> np.ndarray:
    """逐边界区间求交，O(h²)，h 为半平面数；返回多边形、线段或点。

    非空二维多面区域必有可行边界；边界参数区间无界则整个区域无界。
    有界区域的所有极点均为某个非空边界区间的端点，故取其凸包即可。
    不添加人为包围盒，也不裁剪目标圆域。空集、无界分别抛出专用异常。
    """

    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("距离容差必须是有限正数")
    if not lines:
        raise UnboundedLocalizationError("没有半平面约束：可行区域为整个平面，直径无穷大")

    # 使用局部坐标，保留原始边界；重复约束无需特殊去重。
    origin = np.asarray(lines[0].p, dtype=float).copy()
    local = [make_line(np.asarray(line.p) - origin, line.d) for line in lines]
    endpoints: list[np.ndarray] = []
    for boundary in local:
        interval = _boundary_interval(boundary, local, eps)
        if interval is None:
            continue
        lower, upper = interval
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise UnboundedLocalizationError("测向区域非空但无界：仅靠当前角度约束无法得到有限直径")
        endpoints.extend((boundary.p + lower * boundary.d, boundary.p + upper * boundary.d))

    if not endpoints:
        raise EmptyLocalizationError("测向约束不相容：定位区域为空，请检查坐标、角度或误差上界")
    vertices = _convex_hull(endpoints, eps)
    if not np.isfinite(vertices).all() or any(
        not point_inside(line, vertex, 100.0 * eps)
        for vertex in vertices
        for line in local
    ):
        raise NumericalGeometryError("边界交点未通过全部约束复核，请检查近乎平行约束与坐标尺度")
    return vertices + origin


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
        "status": result.region_type,
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
        "coverage_excess_m": round(
            float(np.max(result.vertex_distances_to_center)) - result.circle_radius, 9
        ),
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

    try:
        if arguments.input is None:
            measurements = [
                Measurement(0.0, 0.0, 0.0),
                Measurement(1000.0, 1000.0, 270.0),
            ]
            error_deg = 1.0
        else:
            measurements, error_deg = _load_input(arguments.input)
        result = solve_localization(measurements, error_deg)
    except LocalizationError as error:
        print(json.dumps({"status": error.status, "message": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"status": "invalid_input", "message": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error

    print(json.dumps(_result_as_dict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
