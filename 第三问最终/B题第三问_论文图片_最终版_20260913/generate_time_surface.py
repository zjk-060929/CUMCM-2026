from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "图片"
DATA_FILE = Path(
    r"C:\Users\lenovo\Documents\ChatGPT\mcm\output\B题第三问_队友分享包_20260911"
    r"\空间相关误差调整_20260913\smooth_sensitivity_current_exact2_100.json"
)
ERROR_MODELS = {"smooth100": 100, "smooth300": 300, "smooth600": 600}
SOURCE_COUNTS = np.arange(10, 17)
COMFORT_CMAP = LinearSegmentedColormap.from_list(
    "comfortable_gradient",
    ["#355C7D", "#6FA8C9", "#A8DADC", "#F2E8C9", "#F4B183", "#D96C75"],
)


def configure_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        mpl.font_manager.fontManager.addfont(str(font_path))
        mpl.rcParams["font.family"] = "Microsoft YaHei"
    mpl.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "figure.dpi": 130,
            "savefig.dpi": 360,
            "svg.fonttype": "none",
            "axes.linewidth": 0.9,
        }
    )


def grouped_means() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    scales = np.array(sorted(ERROR_MODELS.values()), dtype=float)
    means = np.zeros((len(scales), len(SOURCE_COUNTS)), dtype=float)
    sample_sizes = np.zeros_like(means, dtype=int)

    for row_index, scale in enumerate(scales.astype(int)):
        model = f"smooth{scale}"
        rows = payload["rows"][model]["exact2"]
        for column_index, source_count in enumerate(SOURCE_COUNTS):
            values = [
                float(row["per_source_s"])
                for row in rows
                if row["success"] and int(row["source_count"]) == int(source_count)
            ]
            if not values:
                raise RuntimeError(f"缺少 N={source_count}, L={scale} m 的仿真结果")
            means[row_index, column_index] = float(np.mean(values))
            sample_sizes[row_index, column_index] = len(values)
    return SOURCE_COUNTS.astype(float), scales, means, sample_sizes


def bilinear_surface(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, nx: int = 240, ny: int = 180
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xd = np.linspace(x.min(), x.max(), nx)
    yd = np.linspace(y.min(), y.max(), ny)
    along_x = np.vstack([np.interp(xd, x, row) for row in z])
    zd = np.column_stack(
        [np.interp(yd, y, along_x[:, column]) for column in range(along_x.shape[1])]
    )
    return *np.meshgrid(xd, yd), zd


def common_levels(z: np.ndarray) -> np.ndarray:
    lower = 10.0 * math.floor(float(z.min()) / 10.0)
    upper = 10.0 * math.ceil(float(z.max()) / 10.0)
    return np.linspace(lower, upper, 17)


def save_data_csv(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, sample_sizes: np.ndarray
) -> None:
    path = HERE / "平均单源用时_21组仿真均值.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["干扰源数量", "空间相关尺度_m", "平均单源用时_s", "样本数"])
        for row_index, scale in enumerate(y):
            for column_index, source_count in enumerate(x):
                writer.writerow(
                    [
                        int(source_count),
                        int(scale),
                        f"{z[row_index, column_index]:.6f}",
                        int(sample_sizes[row_index, column_index]),
                    ]
                )


def make_contour(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, levels: np.ndarray
) -> None:
    xd, yd, zd = bilinear_surface(x, y, z)
    smooth_levels = np.linspace(float(levels.min()), float(levels.max()), 256)

    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    fig.subplots_adjust(left=0.10, right=0.84, bottom=0.14, top=0.86)
    filled = ax.contourf(
        xd, yd, zd, levels=smooth_levels, cmap=COMFORT_CMAP, extend="both"
    )
    ax.set_title("平均单源任务用时的二维等高图", pad=18)
    ax.set_xlabel("干扰源数量 N")
    ax.set_ylabel("空间相关尺度 L / m")
    ax.set_xticks(x.astype(int))
    ax.set_yticks(y.astype(int))
    colorbar = fig.colorbar(filled, ax=ax, pad=0.07, fraction=0.055)
    colorbar.set_label("平均单源用时 / (s/个)", labelpad=10)
    colorbar.outline.set_linewidth(0.7)
    stem = "图9_源数量与空间相关尺度_平均单源用时二维等高图"
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_surface(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, levels: np.ndarray
) -> None:
    xd, yd, zd = bilinear_surface(x, y, z)
    lower = float(levels.min())

    fig = plt.figure(figsize=(11.0, 6.9))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.08, top=0.91)
    surface = ax.plot_surface(
        xd,
        yd,
        zd,
        cmap=COMFORT_CMAP,
        vmin=levels.min(),
        vmax=levels.max(),
        rcount=120,
        ccount=160,
        linewidth=0,
        antialiased=True,
        alpha=0.94,
    )
    ax.set_title("平均单源任务用时的三维响应曲面", loc="left", pad=17)
    ax.set_xlabel("干扰源数量 N", labelpad=10)
    ax.set_ylabel("空间相关尺度 L / m", labelpad=10)
    ax.set_zlabel("平均单源用时 / (s/个)", labelpad=10)
    ax.set_xticks(x.astype(int))
    ax.set_yticks(y.astype(int))
    ax.set_zlim(lower, float(levels.max()))
    ax.view_init(elev=27, azim=-60)
    ax.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    ax.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    ax.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    colorbar = fig.colorbar(surface, ax=ax, pad=0.08, fraction=0.04, shrink=0.72)
    colorbar.set_label("平均单源用时 / (s/个)", labelpad=9)
    colorbar.outline.set_linewidth(0.7)
    stem = "图10_源数量与空间相关尺度_平均单源用时三维曲面"
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y, z, sample_sizes = grouped_means()
    levels = common_levels(z)
    save_data_csv(x, y, z, sample_sizes)
    make_contour(x, y, z, levels)
    make_surface(x, y, z, levels)


if __name__ == "__main__":
    main()
