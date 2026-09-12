"""B题第一问：按方向排序，用双端队列构造上下包络的完整半平面交。

默认路径不使用人为包围盒、逐边界区间法或全顶点×全约束复核。
实数运算模型下总时间 O(m log m)、空间 O(m)。浮点误差约定见报告。
兼容旧版 Measurement / DirectedLine / solve_localization 有界结果接口。
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

EPS = 1.0e-10
ANGLE_EPS = 1.0e-14
REL_EPS = 64.0 * np.finfo(float).eps
BACKEND = "direction_sorted_upper_lower_deques"


class LocalizationError(RuntimeError):
    status = "geometry_error"


class EmptyLocalizationError(LocalizationError):
    status = "empty"


class UnboundedLocalizationError(LocalizationError):
    status = "unbounded"


class NumericalGeometryError(LocalizationError):
    status = "numerical_error"


@dataclass(frozen=True)
class Measurement:
    x: float
    y: float
    bearing_deg: float

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass(frozen=True)
class DirectedLine:
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


def cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def make_line(point, direction) -> DirectedLine:
    p, d = np.asarray(point, dtype=float), np.asarray(direction, dtype=float)
    if (
        p.shape != (2,)
        or d.shape != (2,)
        or not (np.isfinite(p).all() and np.isfinite(d).all())
    ):
        raise ValueError("直线点和方向必须是两个有限数值")
    norm = math.hypot(float(d[0]), float(d[1]))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("直线方向必须非零且可归一化")
    d = d / norm
    # 四轴附近小于角度分辨阈值的分量归零；其余近乎平行线不合并。
    d[np.abs(d) <= ANGLE_EPS] = 0.0
    d = d / math.hypot(float(d[0]), float(d[1]))
    return DirectedLine(
        p.copy(), d, math.atan2(float(d[1]), float(d[0])) % (2 * math.pi)
    )


def point_inside(line, point, eps=EPS) -> bool:
    return cross2(line.d, np.asarray(point) - line.p) >= -eps


def line_intersection(first, second, eps=0.0):
    determinant = cross2(first.d, second.d)
    if abs(determinant) <= eps:
        return None
    return first.p + cross2(second.p - first.p, second.d) / determinant * first.d


def build_bearing_halfplanes(measurements, error_deg=1.0):
    if not measurements:
        raise ValueError("至少需要一条测向记录")
    if not math.isfinite(error_deg) or not 0 < error_deg < 90:
        raise ValueError("误差半角必须在 (0,90) 度内")
    lines = []
    for item in measurements:
        if not all(math.isfinite(float(v)) for v in (item.x, item.y, item.bearing_deg)):
            raise ValueError("坐标与示向度必须是有限值")
        theta = item.bearing_deg % 360.0
        lo, hi = math.radians(theta - error_deg), math.radians(theta + error_deg)
        lines.append(make_line(item.point, [math.cos(lo), math.sin(lo)]))
        lines.append(make_line(item.point, [-math.cos(hi), -math.sin(hi)]))
    return lines


def halfplanes_from_coefficients(rows):
    """每行 [a,b,c] 表示 a*x+b*y>=c；零法向约束单独判断。"""
    lines = []
    for row in rows:
        if len(row) != 3 or not all(math.isfinite(float(v)) for v in row):
            raise ValueError("每条半平面必须为有限的 [a,b,c]")
        a, b, c = map(float, row)
        norm = math.hypot(a, b)
        if norm == 0:
            if c > 0:
                raise EmptyLocalizationError("0 >= 正数：恒假约束")
            continue
        a, b, c = a / norm, b / norm, c / norm
        lines.append(make_line([a * c, b * c], [b, -a]))
    return lines


@dataclass(frozen=True)
class _Function:
    slope: float
    intercept: float

    def at(self, x):
        return self.slope * x + self.intercept


@dataclass
class _Envelope:
    lines: list[_Function]
    starts: list[float]

    def index(self, x):
        return max(0, bisect.bisect_right(self.starts, x) - 1)

    def at(self, x):
        return self.lines[self.index(x)].at(x)


def _max_envelope(functions) -> _Envelope:
    """斜率排序等价于本组边界方向排序；队列弹出无有效区间的边界。"""
    ordered = sorted(functions, key=lambda f: (f.slope, f.intercept))
    unique = []
    for f in ordered:
        if unique and f.slope == unique[-1].slope:
            unique[-1] = f  # max包络保留截距较大者
        else:
            unique.append(f)
    active = deque()
    starts = deque()
    for f in unique:
        begin = -math.inf
        while active:
            previous = active[-1]
            begin = (previous.intercept - f.intercept) / (f.slope - previous.slope)
            if not math.isfinite(begin):
                raise NumericalGeometryError("包络交点溢出，坐标尺度或平行条件过差")
            if begin > starts[-1]:
                break
            active.pop()
            starts.pop()
        if not active:
            begin = -math.inf
        active.append(f)
        starts.append(begin)
    return _Envelope(list(active), list(starts))


def _feasible_x(lower, negative_upper, left, right, eps):
    """双指针合并包络；在每段解 L(x)+(-U(x))<=0，不重复扫描边界。"""
    i = lower.index(left) if math.isfinite(left) else 0
    j = negative_upper.index(left) if math.isfinite(left) else 0
    cursor = left
    feasible_left = math.inf
    feasible_right = -math.inf
    while True:
        a = lower.lines[i]
        b = negative_upper.lines[j]
        next_l = lower.starts[i + 1] if i + 1 < len(lower.lines) else math.inf
        next_u = (
            negative_upper.starts[j + 1]
            if j + 1 < len(negative_upper.lines)
            else math.inf
        )
        end = min(next_l, next_u, right)
        lo, hi = cursor, end
        slope = a.slope + b.slope
        intercept = a.intercept + b.intercept
        if slope == 0.0:
            valid = intercept <= eps
        else:
            cut = -intercept / slope
            if not math.isfinite(cut):
                raise NumericalGeometryError("上下包络相交位置溢出")
            if slope > 0:
                hi = min(hi, cut)
            else:
                lo = max(lo, cut)
            valid = lo <= hi + eps
        if valid:
            if lo > hi:
                lo = hi = lo / 2 + hi / 2
            feasible_left = min(feasible_left, lo)
            feasible_right = max(feasible_right, hi)
        if end == right or end == math.inf:
            break
        if next_l <= end:
            i += 1
        if next_u <= end:
            j += 1
        cursor = end
    if feasible_left == math.inf:
        raise EmptyLocalizationError("上下边界不存在共同可行横坐标")
    return feasible_left, feasible_right


def signed_polygon_area(vertices):
    if len(vertices) < 3:
        return 0.0
    local = np.asarray(vertices) - vertices[0]
    return 0.5 * float(
        np.sum(
            local[:, 0] * np.roll(local[:, 1], -1)
            - local[:, 1] * np.roll(local[:, 0], -1)
        )
    )


def _convex_hull(points, eps):
    ordered = sorted(points, key=lambda p: (float(p[0]), float(p[1])))
    unique = []
    for p in ordered:
        if not unique or np.linalg.norm(p - unique[-1]) > eps:
            unique.append(p)
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def chain(seq):
        result = []
        for p in seq:
            while len(result) >= 2:
                edge = result[-1] - result[-2]
                if cross2(edge, p - result[-2]) > eps * np.linalg.norm(edge):
                    break
                result.pop()
            result.append(p)
        return result

    return np.asarray(chain(unique)[:-1] + chain(unique[::-1])[:-1])


def half_plane_intersection(lines, eps=EPS):
    """O(m log m)完整求交：返回顶点，或抛出 empty/unbounded/numerical_error。"""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("距离容差必须为有限正数")
    if not lines:
        raise UnboundedLocalizationError("没有有效约束，整个平面可行")
    origin = np.asarray(lines[0].p, dtype=float).copy()
    local = [make_line(np.asarray(line.p) - origin, line.d) for line in lines]
    left, right = -math.inf, math.inf
    lower_functions = []
    negative_upper_functions = []
    for line in local:
        a, b = -float(line.d[1]), float(line.d[0])
        c = a * float(line.p[0]) + b * float(line.p[1])
        if b == 0.0:
            if a > 0:
                left = max(left, c / a)
            else:
                right = min(right, c / a)
        elif b > 0:
            lower_functions.append(_Function(-a / b, c / b))
        else:
            negative_upper_functions.append(_Function(a / b, -c / b))
    if left > right + eps:
        raise EmptyLocalizationError("竖直平行约束相互矛盾")
    if left > right:
        left = right = left / 2 + right / 2
    if not lower_functions or not negative_upper_functions:
        raise UnboundedLocalizationError("横坐标可行，且至少一个竖直方向可无限延伸")
    lower = _max_envelope(lower_functions)
    negative_upper = _max_envelope(negative_upper_functions)
    left, right = _feasible_x(lower, negative_upper, left, right, eps)
    if not (math.isfinite(left) and math.isfinite(right)):
        raise UnboundedLocalizationError("存在无限延伸的可行横坐标区间")
    xs_lower = [left] + [x for x in lower.starts[1:] if left < x < right] + [right]
    xs_upper = (
        [left] + [x for x in negative_upper.starts[1:] if left < x < right] + [right]
    )
    points = [np.array([x, lower.at(x)]) for x in xs_lower]
    points += [np.array([x, -negative_upper.at(x)]) for x in xs_upper]
    if not np.isfinite(points).all():
        raise NumericalGeometryError("有界结果包含非有限坐标")
    vertices = _convex_hull(points, eps)
    # 每个极点只查竖直界及两个包络。等价于全部原约束，O(k log m)。
    for x, y in vertices:
        lo, up = lower.at(x), -negative_upper.at(x)
        tol = 100 * eps + REL_EPS * max(1.0, abs(x), abs(y), abs(lo), abs(up))
        if x < left - tol or x > right + tol or y < lo - tol or y > up + tol:
            raise NumericalGeometryError("极点未通过包络复核")
    return vertices + origin


def rotating_calipers_diameter(vertices, eps=REL_EPS):
    polygon = np.asarray(vertices, dtype=float)
    n = len(polygon)
    if not n:
        raise ValueError("顶点集合不能为空")
    if n == 1:
        return DiameterResult(0.0, polygon[0].copy(), polygon[0].copy())
    if n == 2:
        return DiameterResult(
            float(np.linalg.norm(polygon[1] - polygon[0])),
            polygon[0].copy(),
            polygon[1].copy(),
        )
    if signed_polygon_area(polygon) < 0:
        polygon = polygon[::-1]
    opposite = 1
    best = -1.0
    pair = (0, 1)

    def update(i, j):
        nonlocal best, pair
        delta = polygon[i] - polygon[j]
        value = float(delta @ delta)
        if value > best:
            best = value
            pair = (i, j)

    for i in range(n):
        nxt = (i + 1) % n
        edge = polygon[nxt] - polygon[i]
        for _ in range(n):
            next_opposite = (opposite + 1) % n
            step = polygon[next_opposite] - polygon[opposite]
            increase = cross2(edge, step)
            tolerance = eps * float(np.linalg.norm(edge) * np.linalg.norm(step))
            if increase <= tolerance:
                break
            opposite = next_opposite
        update(i, opposite)
        update(nxt, opposite)
        other = (opposite + 1) % n
        step = polygon[other] - polygon[opposite]
        if abs(cross2(edge, step)) <= eps * float(
            np.linalg.norm(edge) * np.linalg.norm(step)
        ):
            update(i, other)
            update(nxt, other)
    return DiameterResult(
        math.sqrt(max(best, 0.0)), polygon[pair[0]].copy(), polygon[pair[1]].copy()
    )


def diameter_circle_coverage(vertices, point_a, point_b, relative_tolerance=1e-9):
    center = np.asarray(point_a) + (np.asarray(point_b) - point_a) / 2
    radius = float(np.linalg.norm(np.asarray(point_b) - point_a) / 2)
    distances = np.linalg.norm(np.asarray(vertices) - center, axis=1)
    tolerance = relative_tolerance * max(1.0, radius * radius)
    covered = bool(np.all(distances * distances <= radius * radius + tolerance))
    return covered, center, radius, distances


def solve_halfplanes(lines, eps=EPS):
    vertices = half_plane_intersection(lines, eps)
    diameter = rotating_calipers_diameter(vertices)
    covered, center, radius, distances = diameter_circle_coverage(
        vertices, diameter.point_a, diameter.point_b
    )
    return LocalizationResult(
        vertices,
        diameter.diameter,
        diameter.point_a,
        diameter.point_b,
        center,
        radius,
        covered,
        distances,
    )


def solve_localization(measurements, error_deg=1.0):
    return solve_halfplanes(build_bearing_halfplanes(measurements, error_deg))


def _result_as_dict(result):
    return {
        "status": result.region_type,
        "backend": BACKEND,
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
            float(max(result.vertex_distances_to_center)) - result.circle_radius, 9
        ),
    }


def solve_payload(data):
    if not isinstance(data, dict):
        raise ValueError("JSON顶层必须是对象")
    if ("measurements" in data) == ("halfplanes" in data):
        raise ValueError("只提供 measurements 或 halfplanes 中的一种")
    if "halfplanes" in data:
        return solve_halfplanes(halfplanes_from_coefficients(data["halfplanes"]))
    measurements = [
        Measurement(float(x["x"]), float(x["y"]), float(x["bearing_deg"]))
        for x in data["measurements"]
    ]
    return solve_localization(measurements, float(data.get("error_deg", 1.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    try:
        if args.input:
            data = json.loads(args.input.read_text(encoding="utf-8-sig"))
        else:
            data = {
                "measurements": [
                    {"x": 0, "y": 0, "bearing_deg": 0},
                    {"x": 1000, "y": 1000, "bearing_deg": 270},
                ]
            }
        answer = _result_as_dict(solve_payload(data))
    except LocalizationError as error:
        print(
            json.dumps(
                {"status": error.status, "message": str(error)}, ensure_ascii=False
            )
        )
        raise SystemExit(2)
    except (ValueError, TypeError, KeyError, OverflowError, OSError) as error:
        print(
            json.dumps(
                {"status": "invalid_input", "message": str(error)}, ensure_ascii=False
            )
        )
        raise SystemExit(2)
    print(json.dumps(answer, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
