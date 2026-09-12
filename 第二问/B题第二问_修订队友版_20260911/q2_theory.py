"""B 题第二问：以最坏定位直径为指标的解析第二检测点策略。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


ANGLE_ERROR_DEG = 1.0
MIN_RECEPTION_RADIUS_M = 1000.0
MAX_RECEPTION_RADIUS_M = 1500.0


@dataclass(frozen=True)
class TheoreticalSolution:
    forward_m: float
    lateral_abs_m: float
    travel_m: float
    forward_ratio: float
    lateral_ratio: float
    nominal_worst_diameter_m: float
    safe_constraint_1_margin_m2: float
    safe_constraint_2_margin_m2: float


def safe_candidate(forward_m: float, lateral_m: float) -> bool:
    """检验保证第二次仍能收到全向源信号的解析充分条件。"""

    alpha = math.radians(ANGLE_ERROR_DEG)
    squared_length = forward_m**2 + lateral_m**2
    projection_lower_bound = (
        forward_m * math.cos(alpha) - abs(lateral_m) * math.sin(alpha)
    )
    return (
        squared_length <= MIN_RECEPTION_RADIUS_M**2 + 1.0e-7
        and squared_length
        <= 2.0 * MIN_RECEPTION_RADIUS_M * projection_lower_bound + 1.0e-7
    )


def approximate_diameter(
    source_distance_m: float,
    forward_m: float,
    lateral_abs_m: float,
) -> float:
    """两条一阶测向误差带交会后的定位平行四边形直径。"""

    if lateral_abs_m <= 0.0:
        return math.inf
    first_half_diagonal, second_half_diagonal = linearized_half_diagonals(
        source_distance_m, forward_m, lateral_abs_m
    )
    return 2.0 * max(first_half_diagonal, second_half_diagonal)


def linearized_half_diagonals(
    source_distance_m: float,
    forward_m: float,
    lateral_abs_m: float,
) -> tuple[float, float]:
    """返回一阶误差带交集的两条半对角线长度 ``h1,h2``。

    首次检测点置于原点，首次示向度中心线置于 x 轴，名义干扰源
    为 ``(r,0)``，第二检测点为 ``(a,b)``。该函数实现正文式（11）—（12）。
    """

    if source_distance_m < 0.0:
        raise ValueError("干扰源距离不能为负")
    if lateral_abs_m <= 0.0:
        return math.inf, math.inf
    alpha = math.radians(ANGLE_ERROR_DEG)
    tangent = math.tan(alpha)
    r = source_distance_m
    a = forward_m
    b = lateral_abs_m
    squared_length = a**2 + b**2
    first = tangent * math.sqrt(
        r**2 + (squared_length - a * r) ** 2 / b**2
    )
    second = tangent * math.sqrt(
        r**2 + (2.0 * r**2 - 3.0 * a * r + squared_length) ** 2 / b**2
    )
    return first, second


def boundary_safe_min_forward_ratio() -> float:
    """移动距离取 1000 m 时，解析安全圆弧允许的最小 ``a/L``。"""

    alpha = math.radians(ANGLE_ERROR_DEG)
    return math.cos(math.radians(60.0) - alpha)


def normalized_diameter_objective(forward_ratio: float) -> float:
    """在移动半径边界 L=1000 m 上，去掉常数后的直径平方目标。"""

    if not 0.0 <= forward_ratio < 1.0:
        return math.inf
    numerator = (13.0 / 4.0 - 3.0 * forward_ratio) * (
        10.0 - 6.0 * forward_ratio
    )
    return numerator / (1.0 - forward_ratio**2)


def derive_solution() -> TheoreticalSolution:
    """返回充分安全域与名义一阶最坏径向直径准则下的闭式解。"""

    length = MIN_RECEPTION_RADIUS_M
    forward_ratio = 9.0 / 11.0
    lateral_ratio = 2.0 * math.sqrt(10.0) / 11.0
    forward = length * forward_ratio
    lateral = length * lateral_ratio
    squared_length = forward**2 + lateral**2
    alpha = math.radians(ANGLE_ERROR_DEG)
    second_bound = 2.0 * MIN_RECEPTION_RADIUS_M * (
        forward * math.cos(alpha) - lateral * math.sin(alpha)
    )
    diameter = approximate_diameter(MAX_RECEPTION_RADIUS_M, forward, lateral)
    if not safe_candidate(forward, lateral):
        raise RuntimeError("闭式解没有通过解析安全域检验")
    return TheoreticalSolution(
        forward_m=forward,
        lateral_abs_m=lateral,
        travel_m=math.sqrt(squared_length),
        forward_ratio=forward_ratio,
        lateral_ratio=lateral_ratio,
        nominal_worst_diameter_m=diameter,
        safe_constraint_1_margin_m2=max(
            0.0, MIN_RECEPTION_RADIUS_M**2 - squared_length
        ),
        safe_constraint_2_margin_m2=second_bound - squared_length,
    )


def local_basis(bearing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """首次示向度方向 u 与其逆时针 90 度方向 v。"""

    angle = math.radians(bearing_deg)
    direction = np.array([math.cos(angle), math.sin(angle)])
    lateral = np.array([-math.sin(angle), math.cos(angle)])
    return direction, lateral


def second_detection_point(
    first_x_m: float,
    first_y_m: float,
    bearing_deg: float,
    side: int,
) -> np.ndarray:
    """把闭式局部解转换到实际坐标系；side=1 左侧，side=-1 右侧。"""

    if side not in (-1, 1):
        raise ValueError("side 必须为 1 或 -1")
    solution = derive_solution()
    direction, lateral = local_basis(bearing_deg)
    first = np.array([first_x_m, first_y_m], dtype=float)
    return (
        first
        + solution.forward_m * direction
        + side * solution.lateral_abs_m * lateral
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="第二问解析第二检测点计算")
    parser.add_argument("--x", type=float, default=0.0, help="首次检测点 x/m")
    parser.add_argument("--y", type=float, default=0.0, help="首次检测点 y/m")
    parser.add_argument("--bearing", type=float, default=0.0, help="首次示向度/度")
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="输出左侧、右侧或两个对称候选点",
    )
    arguments = parser.parse_args()
    sides = {"left": (1,), "right": (-1,), "both": (1, -1)}[arguments.side]
    solution = derive_solution()
    points = []
    for side in sides:
        point = second_detection_point(
            arguments.x, arguments.y, arguments.bearing, side
        )
        points.append(
            {
                "side": "left" if side == 1 else "right",
                "x_m": float(point[0]),
                "y_m": float(point[1]),
            }
        )
    print(
        json.dumps(
            {
                "criterion": "解析充分安全域内，首次中心线名义源的一阶最坏径向定位直径最小",
                "solution": asdict(solution),
                "second_detection_points": points,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
