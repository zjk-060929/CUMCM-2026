"""使用 SciPy/HiGHS 线性规划独立判别一般半平面；不用于在线求解。"""

from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from q1_localization import (
    LocalizationError,
    halfplanes_from_coefficients,
    solve_halfplanes,
)


def classify_lp(rows):
    matrix = np.asarray(rows, dtype=float)
    options = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
    result = linprog(
        [0, 0],
        A_ub=-matrix[:, :2],
        b_ub=-matrix[:, 2],
        bounds=[(None, None)] * 2,
        method="highs",
        options=options,
    )
    if result.status == 2:
        return "empty", None
    if not result.success:
        raise RuntimeError(result.message)
    for c in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
        r = linprog(
            c,
            A_ub=-matrix[:, :2],
            b_ub=-matrix[:, 2],
            bounds=[(None, None)] * 2,
            method="highs",
            options=options,
        )
        if r.status == 3:
            return "unbounded", None
        if not r.success:
            raise RuntimeError(r.message)
    points = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        if abs(np.linalg.det(matrix[[i, j], :2])) < 1e-12:
            continue
        point = np.linalg.solve(matrix[[i, j], :2], matrix[[i, j], 2])
        if np.all(matrix[:, :2] @ point >= matrix[:, 2] - 1e-7):
            points.append(point)
    if not points:
        raise RuntimeError("有界LP缺少枚举极点")
    diameter = max(
        (float(np.linalg.norm(a - b)) for a in points for b in points), default=0.0
    )
    return "bounded", diameter


def run(count, seed):
    rng = np.random.default_rng(seed)
    counts = {"bounded": 0, "empty": 0, "unbounded": 0}
    max_error = 0.0
    failures = []
    for case in range(count):
        n = int(rng.integers(2, 15))
        normals = rng.integers(-8, 9, (n, 2))
        normals[np.all(normals == 0, axis=1)] = [1, 0]
        offsets = rng.integers(-25, 26, n)
        rows = np.column_stack((normals, offsets)).tolist()
        if case % 3 == 0:
            rows += [[1, 0, -30], [-1, 0, -30], [0, 1, -30], [0, -1, -30]]
        expected, diameter = classify_lp(rows)
        counts[expected] += 1
        try:
            result = solve_halfplanes(halfplanes_from_coefficients(rows))
            actual = "bounded"
        except LocalizationError as e:
            actual = e.status
        if actual != expected:
            failures.append(
                {"case": case, "expected": expected, "actual": actual, "rows": rows}
            )
            continue
        if actual == "bounded":
            error = abs(result.diameter - diameter)
            max_error = max(max_error, error)
            if error > 1e-6:
                failures.append({"case": case, "diameter_error_m": error, "rows": rows})
    return {
        "seed": seed,
        "case_count": count,
        "reference": "SciPy HiGHS LP classification + independent 2x2 vertex enumeration",
        "class_counts": counts,
        "max_diameter_difference_m": max_error,
        "failure_count": len(failures),
        "failures": failures,
        "scope": "整数一般半平面；不包含官方模拟器，不证明任意浮点输入的精确性",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    summary = run(args.cases, args.seed)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "independent_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if summary["failure_count"] else 0)
