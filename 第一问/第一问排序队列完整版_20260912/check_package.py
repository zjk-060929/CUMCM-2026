"""只读核验交接包文件大小与SHA256；不修改结果或清单。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "文件清单_SHA256.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    if not MANIFEST.is_file():
        raise FileNotFoundError("缺少文件清单_SHA256.csv，请先完整解压原交接包")
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    failures = []
    expected = set()
    for row in rows:
        relative = row["path"]
        path = (HERE / relative).resolve()
        if not path.is_relative_to(HERE):
            failures.append({"path": relative, "reason": "清单路径超出包目录"})
            continue
        if relative in expected:
            failures.append({"path": relative, "reason": "清单出现重复路径"})
        expected.add(relative)
        if not path.is_file():
            failures.append({"path": relative, "reason": "缺少文件"})
        elif path.stat().st_size != int(row["bytes"]):
            failures.append({"path": relative, "reason": "文件大小不一致"})
        elif sha256(path) != row["sha256"]:
            failures.append({"path": relative, "reason": "SHA256不一致"})
    actual = {
        path.relative_to(HERE).as_posix()
        for path in HERE.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    return {
        "passed": not failures,
        "checked_file_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "additional_files": sorted(actual - expected),
        "note": "清单不校验自身；Python缓存不计。额外文件单独列出。重建材料会改变原交付哈希。",
    }


if __name__ == "__main__":
    try:
        result = verify()
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"passed": False, "message": str(error)}, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
