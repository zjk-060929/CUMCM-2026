"""问题 3 演练策略使用的纯几何函数。

只处理可观测的测向结果，不包含、也不能访问干扰源真实坐标。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


EPS = 1.0e-10
REGION_RADIUS_M = 1800.0
MIN_RECEPTION_RADIUS_M = 1000.0
MAX_RECEPTION_RADIUS_M = 1500.0
BEARING_ERROR_DEG = 1.0
# 题面误差界限是闭区间。若真实误差恰为 +/-1 度，两条角带可能只在
# 一条线段或一个点上相交，纯浮点半平面交会把这个合法闭集误判为空。
# 这里仅把建模角带向外放宽 1e-5 度（1500 m 处不足 3e-4 m），使闭
# 边界在数值上成为内部点。放宽后的区域仍被后续 20 m 覆盖，因而只会
# 极微小地增加保守性，不会排除真实源或制造错误的清除保证。
BEARING_NUMERICAL_MARGIN_DEG = 1.0e-5
CLEAR_RADIUS_M = 20.0
# 圆盘用外切正1024边形转换为线性半平面。1800 m目标圆的最大径向
# 外扩仅约0.00848 m；近似方向保持保守，只会扩大而不会截掉真实可行域。
CIRCLE_APPROXIMATION_SIDES = 1024
COVERAGE_LAYOUT_HEPTAGON = "heptagon_minimax"
COVERAGE_LAYOUT_CENTER_HEX = "center_hex_1150"
COVERAGE_LAYOUTS = (COVERAGE_LAYOUT_HEPTAGON, COVERAGE_LAYOUT_CENTER_HEX)
CENTER_HEX_RADIUS_M = 1150.0

# 第二观测位移不是经验猜值。候选点必须同时满足下列解析安全约束：
#   L^2 = a^2+b^2 <= 1000^2,
#   L^2 <= 2*1000*(a*cos(1deg)-|b|*sin(1deg)),
#   atan(|b|/a) > 1deg.
# 在安全可行域内先作二维网格筛选，再用300组相同场景按均值、P90和
# 最大耗时的等权归一化分数复核（P95另作报告）；(a,b)=(535,105)m
# 的综合尾部风险最低。效率优选
# 来自离线仿真，但无论仿真分布如何，上述不等式都保证第二点能够接收。
SECOND_FORWARD_M = 535.0
SECOND_LATERAL_M = 105.0
SECOND_BASELINE_M = math.hypot(SECOND_FORWARD_M, SECOND_LATERAL_M)


class GeometryError(RuntimeError):
    """定位区域为空、无界或数值退化。"""


@dataclass(frozen=True)
class Observation:
    x: float
    y: float
    bearing_deg: float

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass(frozen=True)
class DirectedLine:
    point: np.ndarray
    direction: np.ndarray
    angle: float


def cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def make_line(point: np.ndarray, direction: np.ndarray) -> DirectedLine:
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        raise ValueError("直线方向向量不能为零")
    direction = direction / norm
    return DirectedLine(
        point=point,
        direction=direction,
        angle=math.atan2(float(direction[1]), float(direction[0]))
        % (2.0 * math.pi),
    )


def point_inside(line: DirectedLine, point: np.ndarray, eps: float = EPS) -> bool:
    return cross2(line.direction, np.asarray(point, dtype=float) - line.point) >= -eps


def line_intersection(
    first: DirectedLine, second: DirectedLine, eps: float = EPS
) -> np.ndarray | None:
    denominator = cross2(first.direction, second.direction)
    if abs(denominator) <= eps:
        return None
    parameter = cross2(second.point - first.point, second.direction) / denominator
    return first.point + parameter * first.direction


def bearing_halfplanes(
    observations: Sequence[Observation], error_deg: float = BEARING_ERROR_DEG
) -> list[DirectedLine]:
    if not observations:
        raise ValueError("至少需要一次示向度观测")
    lines: list[DirectedLine] = []
    robust_error_deg = float(error_deg) + BEARING_NUMERICAL_MARGIN_DEG
    for observation in observations:
        lower = math.radians(observation.bearing_deg - robust_error_deg)
        upper = math.radians(observation.bearing_deg + robust_error_deg)
        lower_direction = np.array([math.cos(lower), math.sin(lower)])
        upper_direction = np.array([math.cos(upper), math.sin(upper)])
        lines.append(make_line(observation.point, lower_direction))
        lines.append(make_line(observation.point, -upper_direction))
    return lines


def circle_outer_halfplanes(
    center: np.ndarray,
    radius: float,
    sides: int = CIRCLE_APPROXIMATION_SIDES,
) -> list[DirectedLine]:
    """用外切正多边形包住圆盘，保证不会排除真实源。"""

    center = np.asarray(center, dtype=float)
    result: list[DirectedLine] = []
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        normal = np.array([math.cos(angle), math.sin(angle)])
        tangent_point = center + radius * normal
        tangent_direction = np.array([-math.sin(angle), math.cos(angle)])
        result.append(make_line(tangent_point, tangent_direction))
    return result


def _remove_same_direction_parallel_lines(
    lines: Sequence[DirectedLine], eps: float
) -> list[DirectedLine]:
    ordered = sorted(lines, key=lambda line: line.angle)
    unique: list[DirectedLine] = []
    for line in ordered:
        if unique:
            previous = unique[-1]
            parallel = abs(cross2(previous.direction, line.direction)) <= eps
            same_direction = float(np.dot(previous.direction, line.direction)) > 0.0
            if parallel and same_direction:
                if cross2(previous.direction, line.point - previous.point) > eps:
                    unique[-1] = line
                continue
        unique.append(line)
    return unique


def _required_intersection(
    first: DirectedLine, second: DirectedLine, eps: float
) -> np.ndarray:
    point = line_intersection(first, second, eps)
    if point is None:
        raise GeometryError("定位边界出现无法闭合的平行线")
    return point


def signed_polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def half_plane_intersection(
    lines: Sequence[DirectedLine], eps: float = EPS
) -> np.ndarray:
    if len(lines) < 3:
        raise GeometryError("半平面数量不足")
    ordered = _remove_same_direction_parallel_lines(lines, eps)
    active: deque[DirectedLine] = deque()

    for line in ordered:
        while len(active) >= 2:
            tail = _required_intersection(active[-2], active[-1], eps)
            if point_inside(line, tail, eps):
                break
            active.pop()
        while len(active) >= 2:
            head = _required_intersection(active[0], active[1], eps)
            if point_inside(line, head, eps):
                break
            active.popleft()
        active.append(line)

    while len(active) >= 3:
        tail = _required_intersection(active[-2], active[-1], eps)
        if point_inside(active[0], tail, eps):
            break
        active.pop()
    while len(active) >= 3:
        head = _required_intersection(active[0], active[1], eps)
        if point_inside(active[-1], head, eps):
            break
        active.popleft()

    if len(active) < 3:
        raise GeometryError("测向信息没有形成有界定位区域")

    active_lines = list(active)
    polygon = np.asarray(
        [
            _required_intersection(
                active_lines[index], active_lines[(index + 1) % len(active_lines)], eps
            )
            for index in range(len(active_lines))
        ]
    )
    if any(
        not point_inside(line, vertex, 100.0 * eps)
        for vertex in polygon
        for line in ordered
    ):
        raise GeometryError("半平面没有形成有效闭合区域")

    filtered: list[np.ndarray] = []
    for vertex in polygon:
        if not filtered or np.linalg.norm(vertex - filtered[-1]) > 100.0 * eps:
            filtered.append(vertex)
    if len(filtered) > 1 and np.linalg.norm(filtered[0] - filtered[-1]) <= 100.0 * eps:
        filtered.pop()
    polygon = np.asarray(filtered, dtype=float)
    if len(polygon) < 3 or abs(signed_polygon_area(polygon)) <= eps:
        raise GeometryError("定位区域退化为点或线段")
    if signed_polygon_area(polygon) < 0:
        polygon = polygon[::-1].copy()
    return polygon


TARGET_REGION_LINES = circle_outer_halfplanes(
    np.zeros(2), REGION_RADIUS_M, CIRCLE_APPROXIMATION_SIDES
)


def observation_feasible_polygon(
    observations: Sequence[Observation],
    no_signal_points: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """返回正示向与无信号支配关系共同给出的保守凸可行域。"""

    if not observations:
        raise GeometryError("至少需要一次有效示向度")
    lines = list(TARGET_REGION_LINES)
    lines.extend(bearing_halfplanes(observations))
    for observation in observations:
        # 能收到信号可推出真实源到检测点不超过其接收半径，而该半径至多 1500 m。
        lines.extend(
            circle_outer_halfplanes(
                observation.point,
                MAX_RECEPTION_RADIUS_M,
                CIRCLE_APPROXIMATION_SIDES,
            )
        )
    # 同一频道的接收半径固定。若S处收到而Q处无信号，则真实源G满足
    # |G-Q| > R >= |G-S|，从而G必在“更靠近S而非Q”的垂直平分线半平面。
    # 该线性约束比仅使用Q周围1000米排除圆更强，且不依赖半径分布。
    for no_signal_point in no_signal_points:
        negative = np.asarray(no_signal_point, dtype=float)
        for observation in observations:
            positive = observation.point
            normal = negative - positive
            if float(np.linalg.norm(normal)) <= EPS:
                continue
            midpoint = 0.5 * (positive + negative)
            direction = np.array([-normal[1], normal[0]], dtype=float)
            lines.append(make_line(midpoint, direction))
    return half_plane_intersection(lines)


def localization_polygon(
    observations: Sequence[Observation],
    no_signal_points: Sequence[np.ndarray] = (),
) -> np.ndarray:
    if len(observations) < 2:
        raise GeometryError("至少需要两次有效示向度才能定位")
    return observation_feasible_polygon(observations, no_signal_points)


def minimax_coverage_radius() -> float:
    """七个等角覆盖点使最坏最近距离最小的解析半径。"""

    return REGION_RADIUS_M / (2.0 * math.cos(math.pi / 7.0))


def coverage_stations(
    layout: str = COVERAGE_LAYOUT_HEPTAGON,
) -> list[np.ndarray]:
    """返回一种七点保证性布局，列表顺序同时作为搜索骨架顺序。"""

    if layout == COVERAGE_LAYOUT_HEPTAGON:
        radius = minimax_coverage_radius()
        return [
            radius
            * np.array(
                [
                    math.cos(2.0 * math.pi * index / 7.0),
                    math.sin(2.0 * math.pi * index / 7.0),
                ]
            )
            for index in range(7)
        ]
    if layout == COVERAGE_LAYOUT_CENTER_HEX:
        return [
            np.zeros(2, dtype=float),
            *[
                CENTER_HEX_RADIUS_M
                * np.array(
                    [
                        math.cos(2.0 * math.pi * index / 6.0),
                        math.sin(2.0 * math.pi * index / 6.0),
                    ]
                )
                for index in range(6)
            ],
        ]
    raise ValueError(f"未知七点覆盖布局: {layout!r}")


def worst_coverage_distance(
    layout: str = COVERAGE_LAYOUT_HEPTAGON,
) -> float:
    if layout == COVERAGE_LAYOUT_HEPTAGON:
        radius = minimax_coverage_radius()
        boundary = math.sqrt(
            REGION_RADIUS_M**2
            + radius**2
            - 2.0 * REGION_RADIUS_M * radius * math.cos(math.pi / 7.0)
        )
        return max(radius, boundary)
    if layout == COVERAGE_LAYOUT_CENTER_HEX:
        # 内部六个等边三角形的最坏点为外接圆心，边界最坏点位于相邻
        # 两个外圈站点的角平分线上。
        interior = CENTER_HEX_RADIUS_M / math.sqrt(3.0)
        boundary = math.sqrt(
            REGION_RADIUS_M**2
            + CENTER_HEX_RADIUS_M**2
            - 2.0
            * REGION_RADIUS_M
            * CENTER_HEX_RADIUS_M
            * math.cos(math.pi / 6.0)
        )
        return max(interior, boundary)
    raise ValueError(f"未知七点覆盖布局: {layout!r}")


def coverage_route_length(layout: str = COVERAGE_LAYOUT_HEPTAGON) -> float:
    stations = coverage_stations(layout)
    start = np.zeros(2, dtype=float)
    total = 0.0
    for station in stations:
        total += float(np.linalg.norm(station - start))
        start = station
    return total


def guaranteed_shared_observation(
    first: Observation,
    candidate: np.ndarray,
) -> bool:
    """判断当前停靠点能否作为保证接收的第二观测点。

    设首次测点至候选点在测得方向及其法向上的分量为 ``a,b``，位移
    平方为 ``L²``。真实方向误差不超过 ``alpha``，故候选位移在真实
    源方向上的最小投影为 ``eta=a cos(alpha)-|b| sin(alpha)``。

    首次收到信号还给出实际接收半径 ``R >= max(1000,r)``，其中 ``r``
    是源到首次测点的距离。因此只要 ``L² <= 1000²`` 且
    ``L² <= 2*1000*eta``，即可分别对 ``r<=1000`` 与 ``r>=1000`` 证明
    候选点到源的距离不超过 ``R``。这比把未知 ``R`` 始终按1000米处理
    更紧，但仍是确定性保证，不依赖随机分布，也不声称两处误差独立。
    """

    point = np.asarray(candidate, dtype=float)
    displacement = point - first.point
    length_squared = float(np.dot(displacement, displacement))
    if length_squared <= EPS:
        return False
    measured_angle = math.radians(first.bearing_deg)
    forward_axis = np.array([math.cos(measured_angle), math.sin(measured_angle)])
    lateral_axis = np.array([-math.sin(measured_angle), math.cos(measured_angle)])
    forward = float(np.dot(displacement, forward_axis))
    lateral = float(np.dot(displacement, lateral_axis))
    error_angle = math.radians(BEARING_ERROR_DEG)
    minimum_projection = (
        forward * math.cos(error_angle) - abs(lateral) * math.sin(error_angle)
    )
    # 判定向安全侧收缩，而不是用 ``+ tolerance`` 放宽解析不等式。
    # 这样即使候选恰在理论边界附近，浮点舍入也不会把一个无法严格
    # 保证接收的点误判为可复用点。0.01 m^2 的收缩对535/105方案及
    # 998.925 m七点均无实际影响。
    tolerance = 1.0e-8 * MIN_RECEPTION_RADIUS_M**2
    if length_squared > MIN_RECEPTION_RADIUS_M**2 - tolerance:
        return False
    if length_squared > (
        2.0 * MIN_RECEPTION_RADIUS_M * minimum_projection - tolerance
    ):
        return False
    # 离开首次正负一度角带，防止两条观测方向退化为同一直线。
    separation = math.degrees(math.atan2(abs(lateral), forward))
    return separation > BEARING_ERROR_DEG + 1.0e-8


def observation_separation_deg(
    first: Observation,
    candidate: np.ndarray,
) -> float:
    """候选位移与首次测得方向之间的无符号夹角。"""

    point = np.asarray(candidate, dtype=float)
    displacement = point - first.point
    if float(np.linalg.norm(displacement)) <= EPS:
        return 0.0
    measured_angle = math.radians(first.bearing_deg)
    forward_axis = np.array([math.cos(measured_angle), math.sin(measured_angle)])
    lateral_axis = np.array([-math.sin(measured_angle), math.cos(measured_angle)])
    forward = float(np.dot(displacement, forward_axis))
    lateral = float(np.dot(displacement, lateral_axis))
    return math.degrees(math.atan2(abs(lateral), forward))


def second_detection_candidates(observation: Observation) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(observation.bearing_deg)
    forward = np.array([math.cos(angle), math.sin(angle)])
    lateral = np.array([-math.sin(angle), math.cos(angle)])
    base = observation.point + SECOND_FORWARD_M * forward
    return (
        base + SECOND_LATERAL_M * lateral,
        base - SECOND_LATERAL_M * lateral,
    )


def ordered_second_detection_candidates(
    current_position: np.ndarray, observation: Observation
) -> tuple[np.ndarray, np.ndarray]:
    candidates = second_detection_candidates(observation)
    return tuple(
        sorted(
            candidates,
            key=lambda point: float(np.linalg.norm(point - current_position)),
        )
    )  # type: ignore[return-value]


def _circle_two(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, float]:
    center = 0.5 * (first + second)
    return center, float(np.linalg.norm(first - center))


def _circle_three(
    first: np.ndarray, second: np.ndarray, third: np.ndarray
) -> tuple[np.ndarray, float] | None:
    ax, ay = first
    bx, by = second
    cx, cy = third
    determinant = 2.0 * (
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    )
    if abs(determinant) <= 1.0e-12:
        return None
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / determinant
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / determinant
    center = np.array([ux, uy])
    return center, float(np.linalg.norm(first - center))


def _contains(circle: tuple[np.ndarray, float], point: np.ndarray) -> bool:
    center, radius = circle
    return float(np.linalg.norm(point - center)) <= radius + 1.0e-7 * max(1.0, radius)


def minimum_enclosing_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    data = np.asarray(points, dtype=float)
    if len(data) == 0:
        raise ValueError("点集不能为空")
    # 固定顺序保证不同机器得到相同结果。
    order = data[np.random.default_rng(20260911 + len(data)).permutation(len(data))]
    circle: tuple[np.ndarray, float] | None = None
    for index, first in enumerate(order):
        if circle is not None and _contains(circle, first):
            continue
        circle = (first.copy(), 0.0)
        for second_index, second in enumerate(order[:index]):
            if _contains(circle, second):
                continue
            circle = _circle_two(first, second)
            for third in order[:second_index]:
                if _contains(circle, third):
                    continue
                triple = _circle_three(first, second, third)
                if triple is not None:
                    circle = triple
                else:
                    pairs = [
                        _circle_two(first, second),
                        _circle_two(first, third),
                        _circle_two(second, third),
                    ]
                    circle = min(
                        (
                            candidate
                            for candidate in pairs
                            if all(
                                _contains(candidate, point)
                                for point in (first, second, third)
                            )
                        ),
                        key=lambda candidate: candidate[1],
                    )
    assert circle is not None
    # _contains 的微小容差只应用于构造算法，不能进入最终清除判据；否则
    # 真实半径略大于 20 m 时可能仍返回 20 m 并错误地只清除圆心。固定
    # 圆心后用全部输入点的实际最大距离上修半径，恢复严格覆盖含义。
    center, _ = circle
    verified_radius = float(np.max(np.linalg.norm(data - center, axis=1)))
    return center, verified_radius


def _grid_axis_with_edges(
    minimum: float, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    width = maximum - minimum
    if width <= EPS:
        center = np.array([(minimum + maximum) / 2.0])
        return center, np.array([minimum, maximum])
    # 每格半对角线严格小于 20 m；1e-6 只抵消浮点边界误差。
    maximum_side = math.sqrt(2.0) * (CLEAR_RADIUS_M - 1.0e-6)
    count = max(1, int(math.ceil(width / maximum_side)))
    edges = np.linspace(minimum, maximum, count + 1)
    return (edges[:-1] + edges[1:]) / 2.0, edges


def _grid_axis(minimum: float, maximum: float) -> np.ndarray:
    return _grid_axis_with_edges(minimum, maximum)[0]


def _serpentine_routes(xs: np.ndarray, ys: np.ndarray) -> list[np.ndarray]:
    routes: list[np.ndarray] = []
    for reverse_y in (False, True):
        points: list[list[float]] = []
        ordered_y = ys[::-1] if reverse_y else ys
        for row, y in enumerate(ordered_y):
            ordered_x = xs[::-1] if row % 2 else xs
            points.extend([[float(x), float(y)] for x in ordered_x])
        route = np.asarray(points, dtype=float)
        routes.append(route)
        routes.append(route[::-1].copy())
    return routes


def _convex_polygon_intersects_rectangle(
    polygon: np.ndarray,
    minimum_x: float,
    maximum_x: float,
    minimum_y: float,
    maximum_y: float,
) -> bool:
    """用分离轴定理判断凸多边形是否与闭矩形相交。

    保守清除只需覆盖定位多边形，而不是覆盖它的整个包围盒。只有能够
    与定位多边形相交的网格单元才需要保留；边界接触按相交处理，避免
    因浮点误差漏掉真实目标。
    """

    data = np.asarray(polygon, dtype=float)
    rectangle = np.asarray(
        [
            [minimum_x, minimum_y],
            [maximum_x, minimum_y],
            [maximum_x, maximum_y],
            [minimum_x, maximum_y],
        ],
        dtype=float,
    )
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for edge in np.roll(data, -1, axis=0) - data:
        normal = np.array([-edge[1], edge[0]], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > EPS:
            axes.append(normal / norm)
    for axis in axes:
        polygon_projection = data @ axis
        rectangle_projection = rectangle @ axis
        tolerance = 100.0 * EPS * max(
            1.0,
            float(np.max(np.abs(polygon_projection))),
            float(np.max(np.abs(rectangle_projection))),
        )
        if (
            float(np.max(polygon_projection))
            < float(np.min(rectangle_projection)) - tolerance
            or float(np.max(rectangle_projection))
            < float(np.min(polygon_projection)) - tolerance
        ):
            return False
    return True


def route_length(start: np.ndarray, points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    total = float(np.linalg.norm(points[0] - start))
    if len(points) > 1:
        total += float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    return total


def _axis_aligned_clear_route(
    polygon: np.ndarray,
    current_position: np.ndarray,
    no_signal_points: Sequence[np.ndarray] = (),
) -> tuple[np.ndarray, float]:
    """返回轴对齐的最小包围圆优先、有限网格兜底清除路线。

    半径不超过 20 m 时只需访问包围圆圆心。否则用半对角线小于
    20 m 的网格覆盖包围盒；若某一整个网格单元已被一次无信号结果
    的 1000 m 排除圆覆盖，则可证明真实源不在该单元，安全跳过。
    """

    center, radius = minimum_enclosing_circle(polygon)
    if radius <= CLEAR_RADIUS_M + 1.0e-8:
        return np.asarray([center]), radius

    minimum = np.min(polygon, axis=0)
    maximum = np.max(polygon, axis=0)
    xs, x_edges = _grid_axis_with_edges(float(minimum[0]), float(maximum[0]))
    ys, y_edges = _grid_axis_with_edges(float(minimum[1]), float(maximum[1]))
    excluded = [np.asarray(point, dtype=float) for point in no_signal_points]

    allowed = np.ones((len(ys), len(xs)), dtype=bool)
    for row in range(len(ys)):
        for column in range(len(xs)):
            if not _convex_polygon_intersects_rectangle(
                polygon,
                float(x_edges[column]),
                float(x_edges[column + 1]),
                float(y_edges[row]),
                float(y_edges[row + 1]),
            ):
                allowed[row, column] = False
                continue
            corners = np.asarray(
                [
                    [x_edges[column], y_edges[row]],
                    [x_edges[column + 1], y_edges[row]],
                    [x_edges[column + 1], y_edges[row + 1]],
                    [x_edges[column], y_edges[row + 1]],
                ]
            )
            if any(
                float(np.max(np.linalg.norm(corners - point, axis=1)))
                < MIN_RECEPTION_RADIUS_M - 1.0e-7
                for point in excluded
            ):
                allowed[row, column] = False

    routes: list[np.ndarray] = []
    for reverse_y in (False, True):
        row_indices = list(range(len(ys)))
        if reverse_y:
            row_indices.reverse()
        points: list[list[float]] = []
        for route_row, row in enumerate(row_indices):
            columns = list(range(len(xs)))
            if route_row % 2:
                columns.reverse()
            for column in columns:
                if allowed[row, column]:
                    points.append([float(xs[column]), float(ys[row])])
        route = np.asarray(points, dtype=float).reshape((-1, 2))
        routes.append(route)
        routes.append(route[::-1].copy())

    if not any(len(route) for route in routes):
        return np.asarray([center]), radius
    route = min(routes, key=lambda points: route_length(center, points))
    route = np.asarray(
        [point for point in route if np.linalg.norm(point - center) > 1.0e-7],
        dtype=float,
    )
    if route.size == 0:
        return np.asarray([center]), radius
    return np.vstack([center, route]), radius


def _rotation_world_to_local(angle: float) -> np.ndarray:
    """返回把世界坐标逆时针旋转 ``-angle`` 的正交变换矩阵。"""

    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray([[cosine, sine], [-sine, cosine]], dtype=float)


def _clear_route_worst_time(start: np.ndarray, route: np.ndarray) -> float:
    """按前 ``n-1`` 次失败、最后一次成功计算清除路线时间上界。"""

    if len(route) == 0:
        return math.inf
    return route_length(np.asarray(start, dtype=float), route) / 5.0 + 3.0 * (
        len(route) - 1
    ) + 5.0


def _rotated_grid_routes(
    polygon: np.ndarray,
    center: np.ndarray,
    no_signal_points: Sequence[np.ndarray],
    angle: float,
) -> list[np.ndarray]:
    """在给定方向建立安全方格，返回四条世界坐标蛇形路线。"""

    transform = _rotation_world_to_local(angle)
    local_polygon = np.asarray(polygon, dtype=float) @ transform.T
    local_center = np.asarray(center, dtype=float) @ transform.T
    local_negatives = [
        np.asarray(point, dtype=float) @ transform.T for point in no_signal_points
    ]
    minimum = np.min(local_polygon, axis=0)
    maximum = np.max(local_polygon, axis=0)
    xs, x_edges = _grid_axis_with_edges(float(minimum[0]), float(maximum[0]))
    ys, y_edges = _grid_axis_with_edges(float(minimum[1]), float(maximum[1]))

    allowed = np.zeros((len(ys), len(xs)), dtype=bool)
    for row in range(len(ys)):
        for column in range(len(xs)):
            left = float(x_edges[column])
            right = float(x_edges[column + 1])
            bottom = float(y_edges[row])
            top = float(y_edges[row + 1])
            if not _convex_polygon_intersects_rectangle(
                local_polygon, left, right, bottom, top
            ):
                continue
            corners = np.asarray(
                [[left, bottom], [right, bottom], [right, top], [left, top]],
                dtype=float,
            )
            if any(
                float(np.max(np.linalg.norm(corners - point, axis=1)))
                < MIN_RECEPTION_RADIUS_M - 1.0e-7
                for point in local_negatives
            ):
                continue
            allowed[row, column] = True

    routes: list[np.ndarray] = []
    for reverse_y in (False, True):
        row_indices = list(range(len(ys)))
        if reverse_y:
            row_indices.reverse()
        points: list[list[float]] = []
        for route_row, row in enumerate(row_indices):
            columns = list(range(len(xs)))
            if route_row % 2:
                columns.reverse()
            for column in columns:
                if allowed[row, column]:
                    points.append([float(xs[column]), float(ys[row])])
        local_route = np.asarray(points, dtype=float).reshape((-1, 2))
        for ordered_route in (local_route, local_route[::-1].copy()):
            ordered_route = np.asarray(
                [
                    point
                    for point in ordered_route
                    if float(np.linalg.norm(point - local_center)) > 1.0e-7
                ],
                dtype=float,
            ).reshape((-1, 2))
            # 行向量从局部坐标恢复到世界坐标。圆心先访问，既利用最可能
            # 的一次清除，也覆盖被删除的圆心所在网格单元。
            world_route = ordered_route @ transform
            routes.append(np.vstack([center, world_route]))
    return routes


def conservative_clear_route(
    polygon: np.ndarray,
    current_position: np.ndarray,
    no_signal_points: Sequence[np.ndarray] = (),
) -> tuple[np.ndarray, float]:
    """返回方向自适应、具有有限覆盖保证的清除点序列。

    最小包围圆半径不超过20米时只访问圆心。否则保留原轴对齐路线，
    并按照定位多边形各边方向及其45度偏移建立旋转方格。每个方格的
    半对角线严格小于20米，只删除与可行域不相交或已被无信号证据
    整格排除的单元，因此所有候选均保持确定性覆盖保证。最终按移动、
    失败清除与成功清除的题设耗时精确选择代价最小者；由于原路线显式
    位于候选集内，模型化最坏路线代价不会增加。
    """

    baseline_route, radius = _axis_aligned_clear_route(
        polygon,
        current_position,
        no_signal_points=no_signal_points,
    )
    if radius <= CLEAR_RADIUS_M + 1.0e-8:
        return baseline_route, radius

    data = np.asarray(polygon, dtype=float)
    center, _ = minimum_enclosing_circle(data)
    angles = {0.0}
    for edge in np.roll(data, -1, axis=0) - data:
        if float(np.linalg.norm(edge)) <= EPS:
            continue
        angle = math.atan2(float(edge[1]), float(edge[0])) % (math.pi / 2.0)
        # 圆整后去重只影响重复计算，不改变几何安全性。
        angles.add(round(angle, 12))
        angles.add(round((angle + math.pi / 4.0) % (math.pi / 2.0), 12))

    routes = [baseline_route]
    for angle in sorted(angles):
        routes.extend(
            _rotated_grid_routes(data, center, no_signal_points, angle)
        )
    start = np.asarray(current_position, dtype=float)
    route = min(routes, key=lambda points: _clear_route_worst_time(start, points))
    return route, radius
