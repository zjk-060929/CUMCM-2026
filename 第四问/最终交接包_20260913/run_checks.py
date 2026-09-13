"""Run the 14 focused geometry and control tests; no complete mission runs."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    out = ROOT/'runs'/('tests_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v',
                             'test_integration', 'test_action', 'test_reliable'],
                            cwd=ROOT/'code/integrated_q4_20260913', capture_output=True, text=True,
                            encoding='utf-8', errors='replace', creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    output = result.stdout+result.stderr
    (out/'tests.txt').write_text(output, encoding='utf-8')
    print(output)
    print('Log:', out/'tests.txt')
    raise SystemExit(result.returncode)
