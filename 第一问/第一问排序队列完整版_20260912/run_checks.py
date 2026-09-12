"""运行全部单元/回归检查并保存可引用记录；不连接任何模拟器。"""

import io
import json
import platform
import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(here), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = here / "results"
    output.mkdir(exist_ok=True)
    (output / "test_results.txt").write_text(stream.getvalue(), encoding="utf-8")
    summary = {
        "test_count": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
        "python": platform.python_version(),
        "scope": "21项旧版回归+31项新增检查；旧回归内部包含100组卡壳和300组测向独立对照，不另计测试项数。",
    }
    (output / "test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(stream.getvalue())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
