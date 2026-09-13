"""Run one offline case with the unchanged frozen Q4 algorithm."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    args = list(sys.argv[1:])
    if '--output' in args:
        i = args.index('--output') + 1
        args[i] = str(Path(args[i]).resolve())
    elif not any(a.startswith('--output=') for a in args):
        args += ['--output', str(ROOT/'runs'/('demo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))]
    else:
        args = ['--output='+str(Path(a.split('=',1)[1]).resolve()) if a.startswith('--output=') else a for a in args]
    raise SystemExit(subprocess.call([sys.executable, '-B', str(ROOT/'code/run_integrated_q4.py'), *args],
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
