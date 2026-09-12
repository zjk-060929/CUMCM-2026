"""相同输入比较三份完整定位实现；耗时是本机现实计算时间，非检测虚拟时间。"""

from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import q1_localization as current

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "reference" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_solve(mod, lines):
    vertices = mod.half_plane_intersection(lines)
    diameter = mod.rotating_calipers_diameter(vertices)
    mod.diameter_circle_coverage(vertices, diameter.point_a, diameter.point_b)
    return diameter.diameter


def run():
    mods = {
        "sorted_deques": current,
        "interval_reference": load("interval_reference"),
        "legacy_deque": load("legacy_deque"),
    }
    rng = np.random.default_rng(20260912)
    records = []
    for group, counts in [
        ("bearing", [2, 5, 10, 20, 50, 100]),
        ("all_active", [20, 80, 200, 400]),
    ]:
        for count in counts:
            scenarios = []
            for case in range(4 if group == "bearing" else 1):
                if group == "bearing":
                    target = rng.uniform(-400, 400, 2)
                    angles = (
                        np.array([0.0, math.pi / 2])
                        if count == 2
                        else np.arange(count) * 2 * math.pi / count
                    ) + rng.uniform(0, 2 * math.pi)
                    stations = target + np.column_stack(
                        (np.cos(angles), np.sin(angles))
                    ) * rng.uniform(200, 1000, (count, 1))
                    bearings = np.degrees(
                        np.arctan2(
                            target[1] - stations[:, 1], target[0] - stations[:, 0]
                        )
                    ) + rng.uniform(-0.8, 0.8, count)
                    scenarios.append((stations, bearings))
                else:
                    angles = np.arange(count) * 2 * math.pi / count + 0.031
                    scenarios.append(
                        [
                            (
                                np.array([math.cos(a), math.sin(a)]) * 100,
                                np.array([-math.sin(a), math.cos(a)]),
                            )
                            for a in angles
                        ]
                    )
            jobs = {}
            for key, mod in mods.items():
                if group == "bearing":
                    batches = [
                        [
                            mod.Measurement(float(p[0]), float(p[1]), float(b))
                            for p, b in zip(s, bearings)
                        ]
                        for s, bearings in scenarios
                    ]
                    jobs[key] = (
                        lambda data, mod=mod: mod.solve_localization(data).diameter,
                        batches,
                    )
                else:
                    batches = [
                        [mod.make_line(p, d) for p, d in scene] for scene in scenarios
                    ]
                    jobs[key] = (
                        lambda data, mod=mod: reference_solve(mod, data),
                        batches,
                    )
            reference = [jobs["sorted_deques"][0](x) for x in jobs["sorted_deques"][1]]
            times = {k: [] for k in mods}
            errors = {k: 0.0 for k in mods}
            repeats = 9 if count <= 50 else (5 if count <= 100 else 3)
            for rep in range(repeats):
                for key in (list(mods) if rep % 2 == 0 else list(mods)[::-1]):
                    func, batches = jobs[key]
                    start = time.perf_counter()
                    answers = [func(x) for x in batches]
                    times[key].append(
                        (time.perf_counter() - start) * 1000 / len(batches)
                    )
                    errors[key] = max(
                        errors[key], max(abs(a - b) for a, b in zip(answers, reference))
                    )
            for key in mods:
                records.append(
                    {
                        "group": group,
                        "input_count": count,
                        "halfplane_count": 2 * count if group == "bearing" else count,
                        "method": key,
                        "median_ms": statistics.median(times[key]),
                        "min_ms": min(times[key]),
                        "case_count": len(scenarios),
                        "repetitions_per_case": repeats,
                        "max_diameter_difference_m": errors[key],
                    }
                )
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    with (output / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": 20260912,
        "scope": "同机交替计时；不含输入对象创建、HTTP、文件输出；bearing含约束构造、求交、直径、覆盖；all_active从半平面对象开始。旧版仅作普通有界输入速度参考。",
        "records": records,
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
