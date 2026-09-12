"""重建第一问配图、公式PNG/SVG及算例结果。只写本包指定输出。"""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib import font_manager
from q1_localization import solve_payload, _result_as_dict, LocalizationError

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FORM = HERE / "formulas"
BLUE = "#245F91"
ORANGE = "#C7652E"
GREEN = "#2D7A63"
RED = "#B54040"
GRAY = "#66727C"


def setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    selected = next(
        (
            f
            for f in [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "WenQuanYi Micro Hei",
                "Arial Unicode MS",
            ]
            if f in names
        ),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "svg.fonttype": "path",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    FIG.mkdir(exist_ok=True)
    FORM.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def axes(ax):
    ax.set_xlabel("x / 米")
    ax.set_ylabel("y / 米")
    ax.grid(alpha=0.16)


def demo(name):
    return solve_payload(json.loads((HERE / name).read_text(encoding="utf-8")))


def example_results():
    examples = {
        name: json.loads((HERE / name).read_text(encoding="utf-8"))
        for name in [
            "example_input.json",
            "counterexample_input.json",
            "point_input.json",
            "segment_input.json",
        ]
    }
    extra = {
        "empty_input.json": {"halfplanes": [[1, 0, 1], [-1, 0, 0]]},
        "unbounded_input.json": {"halfplanes": [[1, 0, 0], [0, 1, 0]]},
        "parallel_strip_input.json": {"halfplanes": [[1, 0, 0], [-1, 0, -1]]},
        "thin_triangle_input.json": {
            "halfplanes": [[1, 0, 0], [0, 1, 0], [-1e-9, -1, -1e-6]]
        },
    }
    (HERE / "examples").mkdir(exist_ok=True)
    for name, payload in extra.items():
        (HERE / "examples" / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        examples["examples/" + name] = payload
    result = {}
    for name, payload in examples.items():
        try:
            result[name] = _result_as_dict(solve_payload(payload))
        except LocalizationError as e:
            result[name] = {"status": e.status, "message": str(e)}
    (HERE / "example_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def bearing_figure():
    result = demo("counterexample_input.json")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, zoom in zip(axs, [False, True]):
        for p, b, c in [([0, 0], 0, BLUE), ([1000, 1000], 269, ORANGE)]:
            p = np.asarray(p)
            lo, hi = np.radians([b - 1, b + 1])
            reach = 1600
            triangle = np.array(
                [
                    p,
                    p + reach * np.array([np.cos(lo), np.sin(lo)]),
                    p + reach * np.array([np.cos(hi), np.sin(hi)]),
                ]
            )
            ax.add_patch(Polygon(triangle, color=c, alpha=0.12))
            for angle in [lo, hi]:
                q = p + reach * np.array([np.cos(angle), np.sin(angle)])
                ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=0.9)
            if not zoom:
                ax.scatter(*p, color=c, s=30)
                ax.annotate(
                    "$S_1$" if b == 0 else "$S_2$",
                    p,
                    xytext=(6, 8),
                    textcoords="offset points",
                )
        ax.add_patch(
            Polygon(result.vertices, facecolor=GREEN, alpha=0.4, edgecolor=GREEN)
        )
        ax.set_aspect("equal")
        axes(ax)
        if zoom:
            ax.set_xlim(955, 1010)
            ax.set_ylim(-27, 27)
            ax.set_title("角形交集局部放大")
        else:
            ax.set_xlim(-100, 1150)
            ax.set_ylim(-100, 1100)
            ax.set_title("两次测向及 ±1° 误差角形")
    save(fig, "fig1_bearing_intersection")


def envelope_figure():
    x = np.linspace(-0.3, 4.3, 500)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(x, 0 * x, color=BLUE, label="下包络 L(x)=0", lw=2.2)
    ax.plot(x, 3 + 0 * x, color=GRAY, lw=1.2, label="上界 y≤3")
    ax.plot(x, 5 - x, color=ORANGE, lw=1.2, label="上界 y≤5−x")
    ax.plot(x, 5 - 0.5 * x, color=GRAY, lw=1.2, ls="--", label="冗余上界 y≤5−0.5x")
    xx = np.linspace(0, 4, 401)
    upper = np.minimum(3, 5 - xx)
    ax.plot(xx, upper, color=ORANGE, lw=3, label="上包络 U(x)")
    ax.fill_between(xx, 0, upper, color=GREEN, alpha=0.16, label="L(x)≤y≤U(x)")
    ax.vlines([0, 4], [-0.5, -0.5], [6, 6], colors=GRAY, linestyles=":", lw=1)
    ax.annotate(
        "切换点 (2,3)",
        (2, 3),
        xytext=(2.2, 4),
        arrowprops={"arrowstyle": "-", "color": GRAY},
    )
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(-0.5, 6)
    axes(ax)
    ax.legend(ncol=2, fontsize=9, loc="upper right")
    save(fig, "fig2_upper_lower_envelopes")


def queue_figure():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    x = np.linspace(-3, 2, 150)
    for index, ax in enumerate(axs):
        ax.axhline(5, color=BLUE, label="y=5")
        ax.plot(x, 5 - x, color=GRAY, ls="--" if index == 2 else "-", label="x+y=5")
        if index >= 1:
            ax.plot(x, 3 - 2 * x, color=ORANGE, label="新边界 2x+y=3")
        if index < 2:
            ax.scatter([0], [5], c=RED)
            ax.annotate("(0,5)", (0, 5), xytext=(7, 5), textcoords="offset points")
        else:
            ax.plot(x, np.minimum(5, 3 - 2 * x), color=GREEN, lw=3)
            ax.scatter([-1], [5], color=GREEN)
        ax.set_xlim(-3, 2)
        ax.set_ylim(-2, 9)
        axes(ax)
        ax.set_title(["旧边界相交", "新约束排除旧拐角", "弹出冗余边界"][index])
        ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig3_deque_redundancy")


def cases_figure():
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.2), layout="constrained")
    titles = ["多边形", "线段", "单点", "空集", "无界条带", "无界射线"]
    for ax, title in zip(axs.flat, titles):
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_aspect("equal")
        ax.set_title(title)
        axes(ax)
    axs[0, 0].add_patch(
        Polygon([[0, 0], [3, 0], [2, 3], [0, 2]], facecolor=BLUE, alpha=0.25)
    )
    axs[0, 1].plot([0, 3], [0, 2], color=BLUE, lw=3, marker="o")
    axs[0, 2].scatter([1.5], [1.5], color=BLUE, s=50)
    axs[1, 0].axvspan(-1, 0, color=BLUE, alpha=0.15)
    axs[1, 0].axvspan(2, 4, color=ORANGE, alpha=0.15)
    axs[1, 0].text(-0.8, 2.5, "x≤0")
    axs[1, 0].text(2.2, 1, "x≥2")
    axs[1, 1].axvspan(0, 2, color=BLUE, alpha=0.15)
    axs[1, 1].axvline(0, color=BLUE)
    axs[1, 1].axvline(2, color=BLUE)
    axs[1, 1].annotate(
        "",
        xy=(1, 3.8),
        xytext=(1, -0.8),
        arrowprops={"arrowstyle": "<->", "color": BLUE},
    )
    axs[1, 2].annotate(
        "",
        xy=(3.6, 3.6),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": BLUE},
    )
    axs[1, 2].scatter([0], [0], color=BLUE)
    save(fig, "fig4_degenerate_and_unbounded")


def circle_comparison():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, name, label in zip(
        axs,
        ["example_input.json", "counterexample_input.json"],
        ["270° 示例：覆盖", "269° 示例：不覆盖"],
    ):
        r = demo(name)
        ax.add_patch(Polygon(r.vertices, facecolor=BLUE, alpha=0.15, edgecolor=BLUE))
        ax.add_patch(
            Circle(r.circle_center, r.circle_radius, fill=False, color=ORANGE, lw=2)
        )
        ax.plot(
            [r.point_a[0], r.point_b[0]],
            [r.point_a[1], r.point_b[1]],
            color=GREEN,
            lw=1.5,
            label="最远点对",
        )
        ax.scatter(*r.circle_center, color=GRAY, s=20)
        ax.scatter(r.vertices[:, 0], r.vertices[:, 1], color=BLUE, s=20)
        for i, p in enumerate(r.vertices):
            ax.annotate(
                f"V{i+1}", p, xytext=(4, 5), textcoords="offset points", fontsize=9
            )
        ax.set_title(label)
        ax.set_aspect("equal")
        axes(ax)
        margin = r.circle_radius * 1.35
        ax.set_xlim(r.circle_center[0] - margin, r.circle_center[0] + margin)
        ax.set_ylim(r.circle_center[1] - margin, r.circle_center[1] + margin)
        ax.text(
            0.02,
            0.02,
            f"D = {r.diameter:.6f} m\n最大超出 = {max(r.vertex_distances_to_center)-r.circle_radius:.6f} m",
            transform=ax.transAxes,
            fontsize=9,
        )
    save(fig, "fig5_diameter_circle_comparison")


def triangle_figure():
    p = np.array([[0, 0], [40, 0], [20, 20 * math.sqrt(3)]])
    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
    ax.add_patch(Polygon(p, facecolor=BLUE, alpha=0.12, edgecolor=BLUE))
    ax.scatter(p[:, 0], p[:, 1], color=BLUE)
    ax.add_patch(
        Circle([20, 0], 20, fill=False, color=ORANGE, lw=2, label="直径圆：半径20 m")
    )
    ax.add_patch(
        Circle(
            [20, 20 / math.sqrt(3)],
            40 / math.sqrt(3),
            fill=False,
            color=GREEN,
            lw=2,
            label="最小包围圆：半径23.094 m",
        )
    )
    ax.set_aspect("equal")
    ax.set_xlim(-8, 48)
    ax.set_ylim(-24, 42)
    axes(ax)
    ax.legend(loc="lower center", fontsize=9)
    save(fig, "fig6_equilateral_counterexample")


def benchmark_figure():
    data = json.loads(
        (HERE / "results/benchmark_summary.json").read_text(encoding="utf-8")
    )["records"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    styles = {
        "sorted_deques": ("本次完整排序队列", BLUE, "o"),
        "interval_reference": ("逐边界区间参考", ORANGE, "s"),
        "legacy_deque": ("旧队列参考", GRAY, "^"),
    }
    for ax, group, title in zip(
        axs, ["bearing", "all_active"], ["普通测向案例", "所有边界有效的正多边形"]
    ):
        for key, (label, color, marker) in styles.items():
            rows = [r for r in data if r["group"] == group and r["method"] == key]
            ax.plot(
                [r["halfplane_count"] for r in rows],
                [r["median_ms"] for r in rows],
                marker=marker,
                color=color,
                label=label,
                lw=1.6,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("半平面数量")
        ax.set_ylabel("计算耗时中位数 / 毫秒")
        ax.set_title(title)
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(
            FixedLocator(
                sorted({r["halfplane_count"] for r in data if r["group"] == group})
            )
        )
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    save(fig, "fig7_runtime_comparison")


def flow_figure():
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    labels = [
        "测向记录或一般线性约束\n检查有限值及误差范围",
        "构造左侧可行的半平面\n平移坐标，分出竖直界、下界、上界",
        "分别按方向排序并维护队列\n构造下包络 L 与负上包络 −U",
        "双指针合并包络\n求 L(x)≤U(x) 的可行横坐标",
        "分类：空集 / 无界 / 有界\n有界时生成端点与包络折点",
        "凸包保留极点\n点、线段直接处理；多边形旋转卡壳",
        "检查直径圆是否覆盖全部极点\n输出状态、直径、最远点对及覆盖结论",
    ]
    for i, label in enumerate(labels):
        y = 0.94 - i * 0.14
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=.55",
                "facecolor": "#EEF3F7",
                "edgecolor": BLUE,
                "linewidth": 1,
            },
        )
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(0.5, y - 0.095),
                xytext=(0.5, y - 0.045),
                arrowprops={"arrowstyle": "->", "color": GRAY},
            )
    save(fig, "fig8_algorithm_flow")


def calipers_figure():
    polygon = np.array([[0, 0], [6, 0], [8, 3], [5, 7], [1, 6], [-1, 3]], dtype=float)
    from q1_localization import rotating_calipers_diameter

    result = rotating_calipers_diameter(polygon)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for ax in axs:
        ax.add_patch(Polygon(polygon, facecolor=BLUE, alpha=0.13, edgecolor=BLUE))
        ax.scatter(polygon[:, 0], polygon[:, 1], color=BLUE, s=25)
        for i, p in enumerate(polygon):
            ax.annotate(
                f"V{i+1}", p, xytext=(5, 5), textcoords="offset points", fontsize=9
            )
        ax.set_xlim(-2, 9)
        ax.set_ylim(-1, 8)
        ax.set_aspect("equal")
        axes(ax)
    axs[0].axhline(0, color=ORANGE, ls="--")
    axs[0].axhline(7, color=ORANGE, ls="--")
    axs[0].plot([0, 5], [0, 7], color=GRAY, ls=":")
    axs[0].plot([6, 5], [0, 7], color=GRAY, ls=":")
    axs[0].annotate(
        "",
        xy=(8.5, 7),
        xytext=(8.5, 0),
        arrowprops={"arrowstyle": "<->", "color": ORANGE},
    )
    axs[0].set_title("固定一条边，寻找最远平行支撑点")
    axs[1].plot(
        [result.point_a[0], result.point_b[0]],
        [result.point_a[1], result.point_b[1]],
        color=GREEN,
        lw=2,
    )
    axs[1].set_title(f"旋转遍历后得到直径 D={result.diameter:.3f} m")
    save(fig, "fig9_rotating_calipers")


FORMULAS = {
    "eq01_angle": r"|\Delta_i(X)|\leq\alpha,\qquad\alpha=1^{\circ}",
    "eq02_halfplane": r"H_j=\{X:d_j\times(X-p_j)\geq0\},\qquad P=\bigcap_{j=1}^{m}H_j",
    "eq03_envelopes": r"L(x)=\max_i(l_i x+b_i),\qquad U(x)=\min_j(u_j x+c_j)",
    "eq04_switch": r"s=\frac{b_{\rm old}-b_{\rm new}}{l_{\rm new}-l_{\rm old}}",
    "eq05_feasibility": r"I=[x_{\min},x_{\max}]\cap\{x:L(x)-U(x)\leq0\}",
    "eq06_vertices": r"P=\operatorname{conv}\{V_1,\ldots,V_k\}",
    "eq07_diameter": r"D(P)=\max_{1\leq i\leq j\leq k}\|V_i-V_j\|_2",
    "eq08_circle": r"C=\frac{A+B}{2},\qquad R=\frac{D}{2},\qquad \max_i\|V_i-C\|_2\leq R",
    "eq09_complexity": r"T(m)=O(m\log m),\qquad S(m)=O(m)",
    "eq10_triangle": r"D=40,\qquad R_{\min}=\frac{40}{\sqrt{3}}\approx23.094>20",
}


def formulas():
    tex = [
        "\\documentclass{article}",
        "\\usepackage{amsmath,amssymb}",
        "\\begin{document}",
    ]
    for name, formula in FORMULAS.items():
        fig = plt.figure(figsize=(9, 0.72))
        fig.text(0.5, 0.5, "$" + formula + "$", ha="center", va="center", fontsize=21)
        fig.savefig(
            FORM / (name + ".png"),
            dpi=250,
            bbox_inches="tight",
            pad_inches=0.06,
            transparent=False,
            facecolor="white",
        )
        fig.savefig(
            FORM / (name + ".svg"),
            bbox_inches="tight",
            pad_inches=0.06,
            transparent=True,
        )
        plt.close(fig)
        tex.extend(["% " + name, "\\[", formula, "\\]"])
    tex.append("\\end{document}")
    (FORM / "formulas.tex").write_text("\n".join(tex), encoding="utf-8")
    (FORM / "formulas.json").write_text(
        json.dumps(FORMULAS, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    setup()
    example_results()
    bearing_figure()
    envelope_figure()
    queue_figure()
    cases_figure()
    circle_comparison()
    triangle_figure()
    benchmark_figure()
    flow_figure()
    calipers_figure()
    formulas()
    print(
        "Generated 9 figures x PNG/SVG and 10 formulas x PNG/SVG, plus example_results.json"
    )
