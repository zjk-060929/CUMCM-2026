from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "图片"
SOURCE_DIR = Path(
    r"C:\Users\lenovo\Documents\ChatGPT\mcm\output\B题第三问_Jammer无门禁代码_20260913"
)
LOG_DIR = SOURCE_DIR / "logs"
FORMAL_CASE = "S8NM-69AU-47PE-7JCP"
FORMAL_CASES = {
    "S8NM-69AU-47PE-7JCP",
    "U5H7-BJQ7-GHHS-TXV5",
    "854N-48FN-MYFC-UUQD",
}
REGION_RADIUS_M = 1800.0
RECEPTION_RADIUS_M = 1000.0
STATION_RADIUS_M = REGION_RADIUS_M / (2.0 * math.cos(math.pi / 7.0))


def configure_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        mpl.font_manager.fontManager.addfont(str(font_path))
        mpl.rcParams["font.family"] = "Microsoft YaHei"
    mpl.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def coverage_stations() -> np.ndarray:
    angles = 2.0 * math.pi * np.arange(7) / 7.0
    return np.column_stack(
        (STATION_RADIUS_M * np.cos(angles), STATION_RADIUS_M * np.sin(angles))
    )


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_log(case_code: str) -> Path:
    matches = sorted(LOG_DIR.glob(f"practice-p3-{case_code}-*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"找不到案例 {case_code} 的 JSONL 日志")
    return matches[-1]


def parse_actions(records: list[dict]) -> list[dict]:
    actions: list[dict] = []
    pending_request: dict | None = None
    previous_position = np.array([0.0, 0.0])
    previous_channel: int | None = None
    previous_virtual_time = 0.0

    for record in records:
        if record.get("event") == "request":
            pending_request = record
            continue
        if record.get("event") != "response" or pending_request is None:
            continue
        path = record.get("path")
        if path == "/enter":
            previous_virtual_time = float(record["response"].get("virtual_time_s", 0.0))
            pending_request = None
            continue
        if path not in {"/measure", "/clear"}:
            pending_request = None
            continue

        payload = pending_request["payload"]
        position = np.array(
            [float(payload["position"]["x"]), float(payload["position"]["y"])]
        )
        channel = int(payload["channel"])
        virtual_time = float(record["response"]["virtual_time_s"])
        result_key = "measure_result" if path == "/measure" else "clear_result"
        result = record["response"].get(result_key, "unknown")
        distance = float(np.linalg.norm(position - previous_position))
        switched = int(previous_channel is not None and channel != previous_channel)
        actions.append(
            {
                "path": path,
                "result": result,
                "channel": channel,
                "position": position,
                "previous_position": previous_position.copy(),
                "distance_m": distance,
                "switched": switched,
                "delta_virtual_time_s": virtual_time - previous_virtual_time,
                "virtual_time_s": virtual_time,
            }
        )
        previous_position = position
        previous_channel = channel
        previous_virtual_time = virtual_time
        pending_request = None
    return actions


def summary_timestamp(path: Path) -> str:
    match = re.search(r"-(\d{8})-(\d{6})-summary\.json$", path.name)
    return "" if match is None else "".join(match.groups())


def case_from_name(path: Path) -> str:
    match = re.search(r"([A-Z0-9]{4}(?:-[A-Z0-9]{4}){3})", path.name)
    if match is None:
        raise ValueError(f"摘要文件名缺少案例编码：{path.name}")
    return match.group(1)


def load_summaries() -> list[dict]:
    paths = sorted(LOG_DIR.glob("*-summary.json"), key=summary_timestamp)
    summaries: list[dict] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        case_code = case_from_name(path)
        source_count = len(data["cleared_channels"])
        data["case_code"] = case_code
        data["source_count"] = source_count
        data["failed_clear_count"] = (
            int(data["clear_attempt_count"]) - int(data["clear_success_count"])
        )
        data["time_per_source_s"] = float(data["final_virtual_time_s"]) / source_count
        summaries.append(data)
    return summaries


def add_domain(ax: plt.Axes, *, fill: bool = False) -> None:
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            REGION_RADIUS_M,
            facecolor="#f3f4f6" if fill else "none",
            edgecolor="#374151",
            linewidth=1.5,
            zorder=0,
        )
    )


def figure_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 11.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        color: str,
        fontsize: float = 10.5,
    ) -> None:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)

    def arrow(start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.25, **kwargs),
        )

    ax.text(
        0.5,
        0.975,
        "问题3探测—定位—清除流程",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )
    box(0.5, 0.905, 0.54, 0.06, "输入：当前位置、频道证据、剩余时间", "#edf6f9")
    box(0.5, 0.800, 0.54, 0.07, "原点零移动扫描20个频道\n记录有信号、无信号与方向", "#dceff1")
    box(0.5, 0.680, 0.54, 0.075, "七点覆盖保底 + 已知目标服务动作\n构成联合滚动规划问题", "#c9e6e6")
    box(0.5, 0.550, 0.54, 0.075, "执行当前计划的第一项\n移动 / 换频 / 测向 / 清除", "#b7dddd")
    box(0.5, 0.420, 0.54, 0.08, "按频道更新可行域\n1024边形 + 半平面交 + 无信号排除", "#a8d5ce")

    diamond = Polygon(
        [[0.5, 0.335], [0.69, 0.260], [0.5, 0.185], [0.31, 0.260]],
        closed=True,
        facecolor="#f1e8cf",
        edgecolor="#334155",
        linewidth=1.1,
    )
    ax.add_patch(diamond)
    ax.text(
        0.5,
        0.260,
        "所有目标已清除，且\n未发现频道取得覆盖证书？",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    box(0.5, 0.085, 0.48, 0.055, "输出清除序列、总时间与结束证书", "#8fc7bd")

    arrow((0.5, 0.872), (0.5, 0.838))
    arrow((0.5, 0.762), (0.5, 0.722))
    arrow((0.5, 0.641), (0.5, 0.592))
    arrow((0.5, 0.510), (0.5, 0.464))
    arrow((0.5, 0.377), (0.5, 0.337))
    arrow((0.5, 0.184), (0.5, 0.119))
    ax.text(0.53, 0.154, "是", color="#166534", fontsize=10)

    ax.annotate(
        "",
        xy=(0.785, 0.680),
        xytext=(0.69, 0.260),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#397a75",
            lw=1.25,
            connectionstyle="angle3,angleA=0,angleB=90",
        ),
    )
    ax.text(0.705, 0.283, "否：反馈后重规划", color="#2d6662", fontsize=10)

    box(0.105, 0.680, 0.18, 0.105, "发现保证\n七点覆盖\n1800 m目标圆域", "#eef3f4", 9.2)
    arrow((0.23, 0.680), (0.198, 0.680))
    box(0.895, 0.420, 0.18, 0.115, "清除保证\n包围圆优先\n否则20 m覆盖", "#eef3f4", 9.2)
    arrow((0.77, 0.420), (0.802, 0.420))

    ax.text(
        0.5,
        0.022,
        "每次只执行一步，利用新反馈重新排序；优化不改变七点发现保证和有限清除兜底。",
        ha="center",
        color="#475569",
        fontsize=9.5,
    )
    save_figure(fig, "图1_探测定位清除总体流程")


def figure_coverage() -> None:
    stations = coverage_stations()
    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    add_domain(ax, fill=True)
    for index, point in enumerate(stations, 1):
        ax.add_patch(
            Circle(
                point,
                RECEPTION_RADIUS_M,
                facecolor="#60a5fa",
                edgecolor="#2563eb",
                linewidth=0.8,
                alpha=0.075,
            )
        )
        ax.scatter(*point, s=52, color="#1d4ed8", zorder=5)
        offset = point / np.linalg.norm(point) * 95.0
        ax.text(*(point + offset), f"H{index}", ha="center", va="center", fontsize=9)

    route = np.vstack(([0.0, 0.0], stations))
    ax.plot(route[:, 0], route[:, 1], color="#f97316", linewidth=2.0, zorder=4)
    ax.scatter(0, 0, marker="*", s=170, color="#dc2626", edgecolor="white", zorder=6)
    ax.text(55, 55, "起点 O", color="#991b1b", fontsize=10)

    ax.annotate(
        "目标圆域 R=1800 m",
        xy=(0, REGION_RADIUS_M),
        xytext=(-1450, 1980),
        arrowprops=dict(arrowstyle="->", color="#374151"),
        fontsize=10,
    )
    ax.text(
        0,
        -2180,
        r"$\rho=1800/[2\cos(\pi/7)]=998.924638\ \mathrm{m}$；每个检测圆半径为1000 m",
        ha="center",
        fontsize=10.5,
    )
    ax.set_title("七点极小极大覆盖与保证性搜索骨架", fontweight="bold")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2200, 2200)
    ax.set_ylim(-2280, 2200)
    ax.grid(alpha=0.18, linewidth=0.6)
    save_figure(fig, "图2_七点覆盖与搜索骨架")


def clip_halfplane(polygon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    output: list[np.ndarray] = []
    for start, end in zip(polygon, np.vstack((polygon[1:], polygon[:1]))):
        fs = a * start[0] + b * start[1] + c
        fe = a * end[0] + b * end[1] + c
        inside_start = fs >= -1e-9
        inside_end = fe >= -1e-9
        if inside_start and inside_end:
            output.append(end)
        elif inside_start and not inside_end:
            t = fs / (fs - fe)
            output.append(start + t * (end - start))
        elif not inside_start and inside_end:
            t = fs / (fs - fe)
            output.append(start + t * (end - start))
            output.append(end)
    return np.asarray(output)


def bearing_constraints(station: np.ndarray, bearing: float, error: float) -> list[tuple]:
    lower = bearing - error
    upper = bearing + error
    dl = np.array([math.cos(lower), math.sin(lower)])
    du = np.array([math.cos(upper), math.sin(upper)])
    # cross(dl, x-station) >= 0；cross(du, x-station) <= 0
    return [
        (-dl[1], dl[0], dl[1] * station[0] - dl[0] * station[1]),
        (du[1], -du[0], -du[1] * station[0] + du[0] * station[1]),
    ]


def circle_from_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float] | None:
    ba = b - a
    ca = c - a
    determinant = 2.0 * (ba[0] * ca[1] - ba[1] * ca[0])
    if abs(determinant) < 1e-10:
        return None
    aa, bb, cc = np.dot(a, a), np.dot(b, b), np.dot(c, c)
    center = np.array(
        [
            (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1]))
            / determinant,
            (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0]))
            / determinant,
        ]
    )
    return center, float(np.linalg.norm(center - a))


def minimum_enclosing_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    candidates: list[tuple[np.ndarray, float]] = []
    for point in points:
        candidates.append((point, 0.0))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            center = (points[i] + points[j]) / 2.0
            candidates.append((center, float(np.linalg.norm(points[i] - center))))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                circle = circle_from_three(points[i], points[j], points[k])
                if circle is not None:
                    candidates.append(circle)
    feasible = [
        item
        for item in candidates
        if np.all(np.linalg.norm(points - item[0], axis=1) <= item[1] + 1e-7)
    ]
    return min(feasible, key=lambda item: item[1])


def figure_localization() -> None:
    target = np.array([700.0, 400.0])
    stations = [np.array([0.0, 0.0]), np.array([520.0, -360.0])]
    biases = [math.radians(0.45), math.radians(-0.35)]
    error = math.radians(1.0)
    angles = np.linspace(0.0, 2.0 * math.pi, 1024, endpoint=False)
    feasible = np.column_stack(
        (REGION_RADIUS_M * np.cos(angles), REGION_RADIUS_M * np.sin(angles))
    )
    bearings: list[float] = []
    for station, bias in zip(stations, biases):
        true_bearing = math.atan2(target[1] - station[1], target[0] - station[0])
        bearing = true_bearing + bias
        bearings.append(bearing)
        for a, b, c in bearing_constraints(station, bearing, error):
            feasible = clip_halfplane(feasible, a, b, c)

    center, radius = minimum_enclosing_circle(feasible)
    fig, ax = plt.subplots(figsize=(9.4, 7.7))
    colors = ["#3f7cac", "#8d6cab"]

    # 主图只显示两条方向带的交会区域。
    for index, (station, bearing, color) in enumerate(
        zip(stations, bearings, colors), 1
    ):
        for theta, style, width, alpha in [
            (bearing, "-", 1.45, 0.92),
            (bearing - error, "--", 1.05, 0.78),
            (bearing + error, "--", 1.05, 0.78),
        ]:
            end = station + 2400.0 * np.array([math.cos(theta), math.sin(theta)])
            ax.plot(
                [station[0], end[0]],
                [station[1], end[1]],
                color=color,
                linestyle=style,
                linewidth=width,
                alpha=alpha,
                label=f"S{index}测向中心线" if style == "-" else None,
                zorder=2,
            )

    ax.add_patch(
        Polygon(
            feasible,
            closed=True,
            facecolor="#f2c879",
            edgecolor="#b16a36",
            linewidth=2.2,
            alpha=0.88,
            zorder=4,
            label="半平面交可行域",
        )
    )
    ax.add_patch(
        Circle(
            center,
            radius,
            facecolor="none",
            edgecolor="#d65a5a",
            linestyle="-.",
            linewidth=1.8,
            zorder=5,
            label="最小包围圆",
        )
    )
    ax.scatter(
        *target,
        marker="*",
        s=210,
        color="#c94c4c",
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
        label="目标位置（示意）",
    )
    ax.scatter(*center, s=32, color="#9f3f4a", zorder=7)

    spacing = 20.0 * math.sqrt(2.0) * 0.96
    xs = np.arange(feasible[:, 0].min() - spacing, feasible[:, 0].max() + spacing, spacing)
    ys = np.arange(feasible[:, 1].min() - spacing, feasible[:, 1].max() + spacing, spacing)
    candidates = []
    for x_value in xs:
        for y_value in ys:
            if np.linalg.norm(np.array([x_value, y_value]) - center) <= radius + 25.0:
                candidates.append((x_value, y_value))
    if candidates:
        candidate_array = np.asarray(candidates)
        ax.scatter(
            candidate_array[:, 0],
            candidate_array[:, 1],
            s=30,
            facecolor="#5bb5a6",
            edgecolor="white",
            linewidth=0.7,
            label="20 m有限清除点",
            zorder=7,
        )

    ax.annotate(
        f"最小包围圆半径 {radius:.1f} m",
        xy=center,
        xytext=(center[0] + 72, center[1] - 72),
        arrowprops=dict(arrowstyle="->", color="#8f4f3a", lw=1.1),
        fontsize=10,
        color="#744033",
    )
    padding = max(105.0, radius * 2.6)
    x_low = float(feasible[:, 0].min() - padding)
    x_high = float(feasible[:, 0].max() + padding)
    y_low = float(feasible[:, 1].min() - padding)
    y_high = float(feasible[:, 1].max() + padding)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_title("局部放大：半平面交定位与20米有限清除", fontweight="bold")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.16)
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.94)

    # 缩略图保留观测点、目标圆域和放大区域之间的全局关系。
    overview = inset_axes(ax, width="29%", height="29%", loc="lower left", borderpad=1.35)
    add_domain(overview, fill=True)
    for station, bearing, color in zip(stations, bearings, colors):
        overview.scatter(*station, marker="^", s=28, color=color, zorder=5)
        end = station + 2300.0 * np.array([math.cos(bearing), math.sin(bearing)])
        overview.plot(
            [station[0], end[0]],
            [station[1], end[1]],
            color=color,
            linewidth=1.0,
            alpha=0.85,
        )
    overview.add_patch(
        Polygon(
            feasible,
            closed=True,
            facecolor="#f2c879",
            edgecolor="#b16a36",
            linewidth=1.0,
            zorder=5,
        )
    )
    overview.add_patch(
        Rectangle(
            (x_low, y_low),
            x_high - x_low,
            y_high - y_low,
            fill=False,
            edgecolor="#c94c4c",
            linewidth=1.2,
            linestyle="--",
            zorder=6,
        )
    )
    overview.set_xlim(-1900, 1900)
    overview.set_ylim(-1800, 1800)
    overview.set_aspect("equal", adjustable="box")
    overview.set_xticks([])
    overview.set_yticks([])
    overview.set_title("全局位置", fontsize=8.5, pad=2)

    save_figure(fig, "图3_交会区域局部放大与有限清除")


def figure_formal_trajectory() -> None:
    records = load_jsonl(find_log(FORMAL_CASE))
    actions = parse_actions(records)
    positions = np.vstack(([0.0, 0.0], [action["position"] for action in actions]))
    times = np.asarray([action["virtual_time_s"] for action in actions])
    segments = np.stack((positions[:-1], positions[1:]), axis=1)

    fig, ax = plt.subplots(figsize=(9.4, 8.3))
    add_domain(ax, fill=True)
    collection = LineCollection(segments, cmap="viridis", linewidth=1.35, alpha=0.9)
    collection.set_array(times)
    collection.set_clim(0.0, times.max())
    ax.add_collection(collection)

    measures = np.asarray([a["position"] for a in actions if a["path"] == "/measure"])
    failures = [a for a in actions if a["path"] == "/clear" and a["result"] != "success"]
    successes = [a for a in actions if a["path"] == "/clear" and a["result"] == "success"]
    if len(measures):
        ax.scatter(measures[:, 0], measures[:, 1], s=10, color="#2563eb", alpha=0.42, label="测量")
    if failures:
        points = np.asarray([a["position"] for a in failures])
        ax.scatter(points[:, 0], points[:, 1], marker="x", s=42, color="#dc2626", linewidth=1.5, label="清除失败")
    if successes:
        points = np.asarray([a["position"] for a in successes])
        ax.scatter(points[:, 0], points[:, 1], marker="*", s=105, color="#f59e0b", edgecolor="#78350f", linewidth=0.55, label="清除成功", zorder=7)
        for action in successes:
            point = action["position"]
            ax.text(point[0] + 25, point[1] + 25, str(action["channel"]), fontsize=8, color="#78350f")

    stations = coverage_stations()
    ax.scatter(stations[:, 0], stations[:, 1], marker="^", s=38, facecolor="white", edgecolor="#475569", label="七点保底站", zorder=6)
    ax.scatter(0, 0, marker="s", s=45, color="#111827", label="起点")
    colorbar = fig.colorbar(collection, ax=ax, pad=0.015, fraction=0.046)
    colorbar.set_label("到达该动作时的累计虚拟时间 / s")

    ax.set_title("正式测试第1次：机器狗轨迹与动作位置", fontweight="bold")
    ax.text(
        0.01,
        0.01,
        "案例 S8NM-69AU-47PE-7JCP\n10/10清除，T=3451.229 s，14次清除失败",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cbd5e1"),
    )
    bound = max(2050.0, float(np.abs(positions).max()) + 150.0)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(alpha=0.16)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    save_figure(fig, "图4_正式测试轨迹与动作分布")


def figure_run_comparison() -> None:
    summaries = load_summaries()
    labels = [
        item["case_code"].split("-")[0]
        + ("\n正式" if item["case_code"] in FORMAL_CASES else "")
        for item in summaries
    ]
    totals = np.asarray([float(item["final_virtual_time_s"]) for item in summaries])
    per_source = np.asarray([float(item["time_per_source_s"]) for item in summaries])
    sources = np.asarray([int(item["source_count"]) for item in summaries])
    failed = np.asarray([int(item["failed_clear_count"]) for item in summaries])
    colors = mpl.colormaps["YlGnBu"](np.linspace(0.30, 0.88, len(summaries)))

    fig, axes = plt.subplots(2, 1, figsize=(10.4, 8.2), sharex=True, gridspec_kw={"hspace": 0.12})
    x = np.arange(len(summaries))
    bars = axes[0].bar(x, totals, color=colors, edgecolor="#315a6b", linewidth=0.7, width=0.72)
    axes[0].axhline(totals.mean(), color="#536878", linestyle="--", linewidth=1.1, label=f"平均 {totals.mean():.1f} s")
    for bar, total, count in zip(bars, totals, sources):
        axes[0].text(bar.get_x() + bar.get_width() / 2, total + 65, f"{count}源", ha="center", fontsize=8.5)
    axes[0].set_ylabel("虚拟总时间 / s")
    axes[0].set_ylim(0, max(totals) * 1.17)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", alpha=0.18)

    bars2 = axes[1].bar(x, per_source, color=colors, edgecolor="#315a6b", linewidth=0.7, width=0.72)
    axes[1].axhline(per_source.mean(), color="#536878", linestyle="--", linewidth=1.1, label=f"逐局平均 {per_source.mean():.1f} s/源")
    for bar, value, fail_count in zip(bars2, per_source, failed):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 7, f"失败{fail_count}", ha="center", fontsize=8.2, color="#74454f")
    axes[1].set_ylabel("平均每源时间 / s")
    axes[1].set_xlabel("测试案例编码前四位")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(per_source) * 1.20)
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", alpha=0.18)

    fig.suptitle("七次模拟器实测结果", fontsize=16, fontweight="bold", y=0.98)
    save_figure(fig, "图5_七次模拟器实测结果对比")


def figure_formal_time_breakdown() -> None:
    actions = parse_actions(load_jsonl(find_log(FORMAL_CASE)))
    total = actions[-1]["virtual_time_s"]
    distance = sum(action["distance_m"] for action in actions)
    movement = distance / 5.0
    switching = float(sum(action["switched"] for action in actions))
    action_time = total - movement - switching
    components = np.asarray([movement, switching, action_time], dtype=float)
    labels = ["移动", "换频", "测量与清除"]
    colors = ["#3f7cac", "#72b7a8", "#e6b566"]

    measure_count = sum(action["path"] == "/measure" for action in actions)
    clear_success = sum(action["path"] == "/clear" and action["result"] == "success" for action in actions)
    clear_failure = sum(action["path"] == "/clear" and action["result"] != "success" for action in actions)
    no_signal = sum(action["path"] == "/measure" and action["result"] == "no_signal" for action in actions)

    fig, (ax_pie, ax_info) = plt.subplots(
        1,
        2,
        figsize=(10.6, 5.4),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.84, wspace=0.04)
    wedges, _ = ax_pie.pie(
        components,
        startangle=92,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 2.0},
    )
    ax_pie.text(
        0,
        0.055,
        f"{total:.1f}",
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color="#243746",
    )
    ax_pie.text(
        0,
        -0.12,
        "总虚拟时间 / s",
        ha="center",
        va="center",
        color="#60717d",
    )
    ax_pie.set_aspect("equal")

    legend_labels = [
        f"{label}    {value:.1f} s（{value / total:.1%}）"
        for label, value in zip(labels, components)
    ]
    ax_info.axis("off")
    ax_info.legend(
        wedges,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.92),
        frameon=False,
        fontsize=11.5,
        handlelength=1.3,
        handletextpad=0.8,
        labelspacing=1.15,
    )
    ax_info.text(
        0.03,
        0.48,
        "运行统计",
        transform=ax_info.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color="#243746",
    )
    stats = [
        ("移动距离", f"{distance:.1f} m"),
        ("测量次数", f"{measure_count}"),
        ("无信号测量", f"{no_signal}"),
        ("清除成功", f"{clear_success}"),
        ("清除失败", f"{clear_failure}"),
    ]
    for index, (name, value) in enumerate(stats):
        y = 0.40 - index * 0.083
        ax_info.text(
            0.03,
            y,
            name,
            transform=ax_info.transAxes,
            color="#60717d",
            fontsize=10.5,
        )
        ax_info.text(
            0.72,
            y,
            value,
            transform=ax_info.transAxes,
            ha="right",
            color="#243746",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_info.plot(
            [0.03, 0.72],
            [y - 0.025, y - 0.025],
            transform=ax_info.transAxes,
            color="#dce4e8",
            linewidth=0.8,
        )

    fig.suptitle("正式测试第1次虚拟时间构成", fontsize=16, fontweight="bold", y=0.95)
    save_figure(fig, "图7_正式测试虚拟时间构成饼图")


def main() -> None:
    configure_style()
    figure_workflow()
    figure_coverage()
    figure_localization()
    figure_formal_trajectory()
    figure_run_comparison()
    figure_formal_time_breakdown()
    print(f"已生成论文图片：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
