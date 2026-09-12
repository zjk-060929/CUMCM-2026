"""生成第二问最终正文所用的三组图像。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

from q1_localization import Measurement, solve_localization
from q2_theory import (
    ANGLE_ERROR_DEG,
    MAX_RECEPTION_RADIUS_M,
    MIN_RECEPTION_RADIUS_M,
    boundary_safe_min_forward_ratio,
    derive_solution,
    normalized_diameter_objective,
)
from q2_validation import bearing_deg, minimum_enclosing_circle


HERE = Path(__file__).resolve().parent


def configure_chinese_font() -> None:
    for filename in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        path = Path(r"C:\Windows\Fonts") / filename
        if path.exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=path
            ).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["svg.fonttype"] = "path"


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(HERE / f"{stem}.png", dpi=320, bbox_inches="tight")
    figure.savefig(HERE / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def figure_safe_region() -> None:
    solution = derive_solution()
    a_values = np.linspace(0.0, 1050.0, 650)
    b_values = np.linspace(-900.0, 900.0, 721)
    aa, bb = np.meshgrid(a_values, b_values)
    alpha = math.radians(ANGLE_ERROR_DEG)
    squared = aa**2 + bb**2
    projection = aa * math.cos(alpha) - np.abs(bb) * math.sin(alpha)
    safe = (squared <= MIN_RECEPTION_RADIUS_M**2) & (
        squared <= 2.0 * MIN_RECEPTION_RADIUS_M * projection
    )

    figure, axis = plt.subplots(figsize=(7.2, 6.6))
    axis.contourf(
        aa,
        bb,
        safe.astype(float),
        levels=[0.5, 1.5],
        colors=["#CDEDE4"],
        alpha=0.95,
    )
    axis.contour(
        aa,
        bb,
        safe.astype(float),
        levels=[0.5],
        colors=["#111111"],
        linewidths=1.8,
    )
    axis.contour(
        aa,
        bb,
        squared - MIN_RECEPTION_RADIUS_M**2,
        levels=[0.0],
        colors=["#666666"],
        linestyles="--",
        linewidths=1.2,
    )
    axis.scatter(
        [solution.forward_m, solution.forward_m],
        [solution.lateral_abs_m, -solution.lateral_abs_m],
        marker="*",
        s=260,
        color="#D1495B",
        edgecolor="black",
        linewidth=0.9,
        zorder=4,
        label="左右对称的闭式解析解",
    )
    axis.scatter([0.0], [0.0], marker="s", s=55, color="#222222", zorder=4)
    axis.annotate(
        "首次示向度方向 $u$",
        xy=(360.0, 0.0),
        xytext=(65.0, -65.0),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4},
        fontsize=11,
    )
    axis.text(
        solution.forward_m - 315.0,
        solution.lateral_abs_m + 55.0,
        r"$(818.18,\ 574.96)$",
        fontsize=10,
    )
    axis.text(
        solution.forward_m - 315.0,
        -solution.lateral_abs_m - 90.0,
        r"$(818.18,\ -574.96)$",
        fontsize=10,
    )
    axis.set_title("第二检测点解析安全候选域及闭式解")
    axis.set_xlabel("沿首次示向度方向位移 $a$/m")
    axis.set_ylabel("横向位移 $b$/m")
    axis.set_xlim(0.0, 1050.0)
    axis.set_ylim(-900.0, 900.0)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_figure(figure, "fig1_safe_region_and_optimum")


def figure_objective_and_geometry() -> None:
    solution = derive_solution()
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.1))

    ratios = np.linspace(boundary_safe_min_forward_ratio(), 0.985, 700)
    objective = np.array([normalized_diameter_objective(value) for value in ratios])
    axes[0].plot(ratios, objective, color="#2F6690", linewidth=2.2)
    optimum_ratio = solution.forward_ratio
    optimum_value = normalized_diameter_objective(optimum_ratio)
    axes[0].scatter(
        [optimum_ratio],
        [optimum_value],
        s=150,
        marker="*",
        color="#D1495B",
        edgecolor="black",
        zorder=4,
    )
    axes[0].axvline(optimum_ratio, color="#D1495B", linestyle="--", alpha=0.75)
    axes[0].annotate(
        "$x^*=9/11$",
        xy=(optimum_ratio, optimum_value),
        xytext=(0.63, optimum_value + 5.0),
        arrowprops={"arrowstyle": "->"},
        fontsize=11,
    )
    axes[0].set_title("(a) 最坏定位直径的无量纲目标函数")
    axes[0].set_xlabel("前向位移比例 $x=a/L$")
    axes[0].set_ylabel(r"$J_D^2/(4L^2\tan^2\alpha)$")
    axes[0].set_ylim(0.0, min(80.0, float(np.quantile(objective, 0.98))))
    axes[0].grid(alpha=0.22)

    second = np.array([solution.forward_m, solution.lateral_abs_m])
    source = np.array([MAX_RECEPTION_RADIUS_M, 0.0])
    second_bearing = bearing_deg(second, source)
    localization = solve_localization(
        [Measurement(0.0, 0.0, 0.0), Measurement(*second, second_bearing)],
        ANGLE_ERROR_DEG,
    )
    polygon = np.vstack([localization.vertices, localization.vertices[0]])
    mec_center, mec_radius = minimum_enclosing_circle(localization.vertices)
    mec_circle = plt.Circle(
        mec_center,
        mec_radius,
        fill=False,
        linestyle="--",
        linewidth=1.7,
        color="#E09F3E",
        label=rf"最小包围圆 $\rho={mec_radius:.2f}$ m",
    )
    axes[1].fill(
        polygon[:, 0],
        polygon[:, 1],
        color="#CDEDE4",
        alpha=0.9,
        label="精确半平面交定位区域",
    )
    axes[1].plot(polygon[:, 0], polygon[:, 1], color="#245C53", linewidth=1.8)
    axes[1].plot(
        [localization.point_a[0], localization.point_b[0]],
        [localization.point_a[1], localization.point_b[1]],
        color="#D1495B",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label=rf"旋转卡壳直径 $D={localization.diameter:.2f}$ m",
    )
    axes[1].add_patch(mec_circle)
    axes[1].scatter(
        [source[0]],
        [source[1]],
        marker="*",
        s=180,
        color="#E09F3E",
        edgecolor="black",
        zorder=4,
    )
    axes[1].text(source[0] + 4.0, source[1] + 3.5, "$G$", fontsize=11)
    axes[1].set_title("(b) $r=1500$ m 名义场景的精确定位区域")
    axes[1].set_xlabel("$x$/m")
    axes[1].set_ylabel("$y$/m")
    axes[1].set_xlim(1432.0, 1568.0)
    axes[1].set_ylim(-68.0, 68.0)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(alpha=0.18)
    axes[1].legend(loc="upper left", fontsize=8.5)

    figure.subplots_adjust(wspace=0.24)
    save_figure(figure, "fig2_diameter_objective")


def load_simulation_values() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with (HERE / "simulation_samples.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    diameters = np.array(
        [float(row["diameter_m"]) for row in rows if row["diameter_m"] != "nan"]
    )
    mec_radii = np.array(
        [float(row["mec_radius_m"]) for row in rows if row["diameter_m"] != "nan"]
    )
    with (HERE / "simulation_summary.json").open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    return diameters, mec_radii, summary


def figure_simulation() -> None:
    diameters, mec_radii, summary = load_simulation_values()
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    axes[0].hist(
        diameters,
        bins=52,
        color="#70A6A0",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.92,
    )
    mean = float(summary["diameter_m"]["mean"])
    p90 = float(summary["diameter_m"]["p90"])
    theoretical = float(summary["theoretical_nominal_worst_diameter_m"])
    axes[0].axvline(mean, color="#D1495B", linewidth=1.8, label=f"均值 {mean:.2f} m")
    axes[0].axvline(
        p90,
        color="#2F6690",
        linewidth=1.8,
        linestyle="--",
        label=f"90%分位 {p90:.2f} m",
    )
    axes[0].axvline(
        theoretical,
        color="#7A5195",
        linewidth=1.7,
        linestyle=":",
        label=f"理论名义最坏值 {theoretical:.2f} m",
    )
    scenario_count = int(summary["scenario_count"])
    axes[0].set_title(f"(a) {scenario_count} 个验证场景的定位直径分布")
    axes[0].set_xlabel("定位区域直径 $D$/m")
    axes[0].set_ylabel("场景数")
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].legend(fontsize=9)

    sorted_radii = np.sort(mec_radii)
    probabilities = np.arange(1, len(sorted_radii) + 1) / len(sorted_radii)
    axes[1].plot(
        sorted_radii,
        probabilities * 100.0,
        color="#2F6690",
        linewidth=2.1,
    )
    axes[1].axvline(20.0, color="#D1495B", linestyle="--", linewidth=1.5)
    within = float(np.mean(sorted_radii <= 20.0) * 100.0)
    axes[1].scatter([20.0], [within], color="#D1495B", s=65, zorder=4)
    axes[1].annotate(
        rf"$\rho\leq20$ m：{within:.1f}%",
        xy=(20.0, within),
        xytext=(29.0, max(8.0, within - 15.0)),
        arrowprops={"arrowstyle": "->"},
        fontsize=10,
    )
    axes[1].set_title("(b) 最小包围圆半径的经验累积分布")
    axes[1].set_xlabel(r"最小包围圆半径 $\rho$/m")
    axes[1].set_ylabel("累计比例/%")
    axes[1].set_xlim(0.0, max(68.0, float(np.max(sorted_radii)) * 1.03))
    axes[1].set_ylim(0.0, 101.0)
    axes[1].grid(alpha=0.2)

    figure.subplots_adjust(wspace=0.24)
    save_figure(figure, "fig3_simulation_validation")


def main() -> None:
    configure_chinese_font()
    figure_safe_region()
    figure_objective_and_geometry()
    figure_simulation()


if __name__ == "__main__":
    main()
