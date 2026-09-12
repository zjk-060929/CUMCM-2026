"""生成 B 题第一问的论文插图，同时输出 PNG 和 SVG 两种格式。"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from q1_localization import Measurement, rotating_calipers_diameter, solve_localization


OUTPUT_DIR = Path(__file__).resolve().parent
BLUE = "#2F6690"
GREEN = "#3A9278"
RED = "#D1495B"
ORANGE = "#F4A261"
DARK = "#2B2D42"
PALE = "#F7F9FC"


def configure_chinese_font() -> None:
    """优先使用 Windows 微软雅黑，保证中文标题和标注正常显示。"""

    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = [
        windows_dir / "Fonts" / "msyh.ttc",
        windows_dir / "Fonts" / "simhei.ttf",
        windows_dir / "Fonts" / "simsun.ttc",
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(path)
            font_name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=360,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_DIR / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def wedge_points(
    measurement: Measurement,
    error_deg: float,
    length: float,
) -> np.ndarray:
    lower = math.radians(measurement.bearing_deg - error_deg)
    upper = math.radians(measurement.bearing_deg + error_deg)
    source = measurement.point
    return np.array(
        [
            source,
            source + length * np.array([math.cos(lower), math.sin(lower)]),
            source + length * np.array([math.cos(upper), math.sin(upper)]),
        ]
    )


def draw_wedge(
    axis: plt.Axes,
    measurement: Measurement,
    color: str,
    label: str,
    length: float = 1500.0,
    error_deg: float = 1.0,
) -> None:
    triangle = wedge_points(measurement, error_deg, length)
    axis.fill(
        triangle[:, 0],
        triangle[:, 1],
        color=color,
        alpha=0.14,
        linewidth=0,
    )
    axis.plot(
        triangle[[0, 1], 0],
        triangle[[0, 1], 1],
        color=color,
        linewidth=1.4,
    )
    axis.plot(
        triangle[[0, 2], 0],
        triangle[[0, 2], 1],
        color=color,
        linewidth=1.4,
    )
    center_angle = math.radians(measurement.bearing_deg)
    center_end = measurement.point + length * np.array(
        [math.cos(center_angle), math.sin(center_angle)]
    )
    axis.plot(
        [measurement.x, center_end[0]],
        [measurement.y, center_end[1]],
        linestyle="--",
        color=color,
        linewidth=1.0,
        alpha=0.9,
        label=label,
    )


def figure_half_plane_intersection() -> None:
    measurements = [
        Measurement(0.0, 0.0, 0.0),
        Measurement(1000.0, 1000.0, 270.0),
    ]
    result = solve_localization(measurements, error_deg=1.0)
    closed_polygon = np.vstack([result.vertices, result.vertices[0]])

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for axis in axes:
        draw_wedge(axis, measurements[0], BLUE, r"$S_1$ 测向中心线")
        draw_wedge(axis, measurements[1], GREEN, r"$S_2$ 测向中心线")
        axis.fill(
            closed_polygon[:, 0],
            closed_polygon[:, 1],
            color=RED,
            alpha=0.34,
            label="半平面交定位区域",
            zorder=5,
        )
        axis.plot(
            closed_polygon[:, 0],
            closed_polygon[:, 1],
            color=RED,
            linewidth=2.0,
            zorder=6,
        )
        axis.scatter(
            [measurement.x for measurement in measurements],
            [measurement.y for measurement in measurements],
            color=DARK,
            marker="s",
            s=40,
            zorder=8,
        )
        axis.scatter([1000.0], [0.0], marker="*", s=115, color=ORANGE, zorder=9)
        axis.grid(alpha=0.2, linewidth=0.7)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("东向坐标 $x$/m")
        axis.set_ylabel("北向坐标 $y$/m")

    axes[0].set_xlim(-100.0, 1120.0)
    axes[0].set_ylim(-150.0, 1100.0)
    axes[0].set_title("(a) 两次测向形成的误差扇形")
    axes[0].text(18.0, 35.0, r"$S_1$", color=DARK)
    axes[0].text(1015.0, 1015.0, r"$S_2$", color=DARK)
    axes[0].text(1018.0, -28.0, r"示例位置 $G$", color=ORANGE)
    axes[0].add_patch(
        Rectangle(
            (955.0, -45.0),
            90.0,
            90.0,
            fill=False,
            edgecolor=DARK,
            linewidth=1.1,
            linestyle=":",
        )
    )
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)

    axes[1].set_xlim(960.0, 1040.0)
    axes[1].set_ylim(-45.0, 45.0)
    axes[1].set_title("(b) 定位多边形局部放大图")
    for index, vertex in enumerate(result.vertices, start=1):
        axes[1].scatter(*vertex, color=RED, s=25, zorder=10)
        axes[1].text(
            vertex[0] + 1.2,
            vertex[1] + (1.5 if vertex[1] >= 0 else -3.5),
            rf"$V_{index}$",
            fontsize=9,
            color=DARK,
        )
    axes[1].text(
        962.0,
        38.0,
        r"示向度误差：$\pm1^\circ$",
        fontsize=10,
        color=DARK,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CBD5E1"),
    )

    figure.subplots_adjust(wspace=0.24)
    save_figure(figure, "fig1_half_plane_intersection")


def figure_rotating_calipers() -> None:
    vertices = np.array(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
            [4.0, -0.5],
            [5.0, 2.0],
            [3.5, 4.5],
            [1.0, 5.0],
            [-1.5, 3.0],
        ]
    )
    diameter = rotating_calipers_diameter(vertices)
    closed_polygon = np.vstack([vertices, vertices[0]])
    direction = diameter.point_b - diameter.point_a
    normal_line_direction = np.array([-direction[1], direction[0]])
    normal_line_direction /= np.linalg.norm(normal_line_direction)

    figure, axis = plt.subplots(figsize=(7.6, 6.4))
    axis.fill(
        closed_polygon[:, 0],
        closed_polygon[:, 1],
        facecolor="#DDEBF3",
        edgecolor=BLUE,
        linewidth=2.0,
        zorder=2,
    )
    axis.scatter(vertices[:, 0], vertices[:, 1], color=BLUE, s=38, zorder=4)

    for index, vertex in enumerate(vertices):
        axis.text(vertex[0] + 0.10, vertex[1] + 0.12, rf"$V_{index}$", fontsize=9)

    span = 8.0
    for point in (diameter.point_a, diameter.point_b):
        ends = np.vstack(
            [point - span * normal_line_direction, point + span * normal_line_direction]
        )
        axis.plot(
            ends[:, 0],
            ends[:, 1],
            color=ORANGE,
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )

    axis.annotate(
        "",
        xy=diameter.point_b,
        xytext=diameter.point_a,
        arrowprops=dict(arrowstyle="<->", color=RED, linewidth=2.5),
        zorder=6,
    )
    midpoint = (diameter.point_a + diameter.point_b) / 2.0
    axis.text(
        midpoint[0] + 0.15,
        midpoint[1] - 0.42,
        rf"$D={diameter.diameter:.2f}$",
        color=RED,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )
    axis.scatter(
        [diameter.point_a[0], diameter.point_b[0]],
        [diameter.point_a[1], diameter.point_b[1]],
        color=RED,
        s=72,
        zorder=7,
        label="最远对踵点对",
    )
    axis.text(
        -1.7,
        5.7,
        "两条平行支撑线随多边形边界同步旋转，\n最远点对必属于对踵点对集合。",
        fontsize=10,
        color=DARK,
        va="top",
    )
    axis.set_title("旋转卡壳求凸多边形直径")
    axis.set_xlabel("$x$")
    axis.set_ylabel("$y$")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-2.4, 6.0)
    axis.set_ylim(-1.8, 6.0)
    axis.grid(alpha=0.18)
    axis.legend(loc="lower right", frameon=True)
    save_figure(figure, "fig2_rotating_calipers")


def figure_diameter_circle_counterexample() -> None:
    side = 100.0
    height = math.sqrt(3.0) * side / 2.0
    point_a = np.array([0.0, 0.0])
    point_b = np.array([side, 0.0])
    point_e = np.array([side / 2.0, height])
    triangle = np.vstack([point_a, point_b, point_e, point_a])
    diameter_center = (point_a + point_b) / 2.0
    minimum_center = np.array([side / 2.0, height / 3.0])

    figure, axis = plt.subplots(figsize=(7.6, 6.8))
    axis.add_patch(
        Circle(
            diameter_center,
            side / 2.0,
            facecolor="#FCE8EA",
            edgecolor=RED,
            linewidth=2.0,
            alpha=0.75,
            label="以 $AB$ 为直径的圆",
        )
    )
    axis.add_patch(
        Circle(
            minimum_center,
            side / math.sqrt(3.0),
            fill=False,
            edgecolor=BLUE,
            linewidth=1.6,
            linestyle="--",
            label="三角形的最小覆盖圆",
        )
    )
    axis.fill(
        triangle[:, 0],
        triangle[:, 1],
        facecolor="#FFF3CD",
        edgecolor=DARK,
        linewidth=2.0,
        alpha=0.68,
        zorder=3,
    )
    axis.plot([0.0, side], [0.0, 0.0], color=RED, linewidth=3.0, zorder=5)
    axis.scatter(
        [point_a[0], point_b[0], point_e[0]],
        [point_a[1], point_b[1], point_e[1]],
        color=DARK,
        s=55,
        zorder=6,
    )
    axis.scatter(*diameter_center, color=RED, marker="x", s=75, zorder=7)
    axis.text(-5.0, 3.0, "$A$", fontsize=12)
    axis.text(side + 2.0, 3.0, "$B$", fontsize=12)
    axis.text(point_e[0] + 3.0, point_e[1] + 2.0, "$E$", fontsize=12)
    axis.text(diameter_center[0] + 2.0, -6.0, "$C$", fontsize=12, color=RED)
    axis.text(40.0, 4.0, "$D=AB$", fontsize=11, color="white", zorder=8)
    axis.annotate(
        "",
        xy=point_e,
        xytext=diameter_center,
        arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.8),
    )
    axis.text(
        56.0,
        48.0,
        r"$CE=\dfrac{\sqrt{3}}{2}D>\dfrac{D}{2}$",
        fontsize=12,
        color=DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1"),
    )
    axis.set_title("直径圆不能必然覆盖定位区域的反例")
    axis.set_xlabel("$x$/m")
    axis.set_ylabel("$y$/m")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-18.0, 118.0)
    axis.set_ylim(-58.0, 106.0)
    axis.grid(alpha=0.16)
    axis.legend(loc="lower center", frameon=True)
    save_figure(figure, "fig3_diameter_circle_counterexample")


def add_box(
    axis: plt.Axes,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = BLUE,
) -> None:
    patch = FancyBboxPatch(
        (center_x - width / 2.0, center_y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=axis.transAxes,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        center_x,
        center_y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color=DARK,
        transform=axis.transAxes,
        zorder=3,
    )


def add_vertical_arrow(axis: plt.Axes, start_y: float, end_y: float, x: float = 0.45) -> None:
    axis.annotate(
        "",
        xy=(x, end_y),
        xytext=(x, start_y),
        xycoords=axis.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=DARK, linewidth=1.25),
    )


def figure_algorithm_flowchart() -> None:
    figure, axis = plt.subplots(figsize=(8.2, 10.5))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    x = 0.45
    width = 0.52
    height = 0.064
    add_box(axis, x, 0.95, width, height, "输入检测点坐标与示向度", "#EAF2F8")
    add_box(axis, x, 0.84, width, height, "构造 $2n$ 个有向半平面约束", "#EAF2F8")
    add_box(axis, x, 0.73, width, height, "按方向角排序并删除同向冗余边界", "#EAF2F8")
    add_box(axis, x, 0.62, width, height, "双端队列执行半平面交", "#E8F5E9", GREEN)

    diamond_center = np.array([x, 0.50])
    diamond_width = 0.23
    diamond_height = 0.075
    diamond = Polygon(
        [
            diamond_center + [0.0, diamond_height],
            diamond_center + [diamond_width, 0.0],
            diamond_center + [0.0, -diamond_height],
            diamond_center + [-diamond_width, 0.0],
        ],
        closed=True,
        facecolor="#FFF3CD",
        edgecolor=ORANGE,
        linewidth=1.6,
        transform=axis.transAxes,
    )
    axis.add_patch(diamond)
    axis.text(
        x,
        0.50,
        "交集是否为\n有界凸多边形？",
        ha="center",
        va="center",
        fontsize=10,
        transform=axis.transAxes,
    )

    add_box(axis, 0.82, 0.50, 0.27, 0.085, "空集：检查数据\n无界：增加检测点", "#FCE8EA", RED)
    add_box(axis, x, 0.375, width, height, r"输出逆时针有序顶点 $V_0,\ldots,V_{m-1}$", "#E8F5E9", GREEN)
    add_box(axis, x, 0.265, width, height, "旋转卡壳枚举对踵点对，求 $D,A,B$", "#E8F5E9", GREEN)
    add_box(axis, x, 0.155, width, height, r"检验 $\|V_k-(A+B)/2\|\leq D/2$", "#FCE8EA", RED)
    add_box(axis, x, 0.055, width, height, "输出定位多边形、直径及覆盖结论", "#EAF2F8")

    add_vertical_arrow(axis, 0.915, 0.875, x)
    add_vertical_arrow(axis, 0.805, 0.765, x)
    add_vertical_arrow(axis, 0.695, 0.655, x)
    add_vertical_arrow(axis, 0.585, 0.568, x)
    add_vertical_arrow(axis, 0.425, 0.410, x)
    add_vertical_arrow(axis, 0.342, 0.298, x)
    add_vertical_arrow(axis, 0.232, 0.188, x)
    add_vertical_arrow(axis, 0.122, 0.088, x)

    axis.annotate(
        "",
        xy=(0.685, 0.50),
        xytext=(0.68, 0.50),
        xycoords=axis.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=DARK, linewidth=1.25),
    )
    axis.text(0.47, 0.414, "是", fontsize=9, transform=axis.transAxes)
    axis.text(0.70, 0.516, "否", fontsize=9, transform=axis.transAxes)
    axis.set_title("第一问求解流程", fontsize=15, pad=14, color=DARK)
    save_figure(figure, "fig4_algorithm_flowchart")


def main() -> None:
    configure_chinese_font()
    figure_half_plane_intersection()
    figure_rotating_calipers()
    figure_diameter_circle_counterexample()
    figure_algorithm_flowchart()
    print(f"已生成 4 幅 PNG 和 4 幅 SVG：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
