"""Verify delivered files, or rerun in a separate timestamped output directory."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess,sys

BASE=Path(__file__).resolve().parent

def verify():
    manifest=json.loads((BASE/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=BASE/item['path']
        if not path.is_file() or hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=item['sha256']:
            raise SystemExit('Missing or modified delivered file: '+item['path'])
    print('Verified all '+str(len(manifest['files']))+' delivered files.',flush=True)

def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--verify',action='store_true');group.add_argument('--quick',action='store_true');group.add_argument('--all',action='store_true')
    args=parser.parse_args();verify()
    if not(args.quick or args.all):return
    target=BASE/'_复现输出'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    target.mkdir(parents=True)
    for file in BASE.glob('*.py'):
        if file.name!='run_reproduce.py':shutil.copy2(file,target/file.name)
    shutil.copy2(BASE/'requirements.txt',target/'requirements.txt')
    for name in ['00_先读我.md','01_运行与文件说明.md']:
        shutil.copy2(BASE/name,target/name)
    steps=['solve.py','refine.py','validate_geometry.py','statistics_run.py','final_checks.py','make_report.py'] if args.all else ['validate_geometry.py']
    print('Reproduction output: '+str(target),flush=True)
    for step in steps:
        print('Running '+step,flush=True)
        subprocess.run([sys.executable,str(target/step)],cwd=target,check=True)
    print('Completed: '+str(target),flush=True)

if __name__=='__main__':main()
