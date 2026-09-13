"""Explicitly rerun a finite subset of the archived common-scene benchmark.

Default: one held-out case, integrated policy. --full selects every case in
the specified stage. New output always goes to a fresh runs directory.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT/'code/integrated_q4_20260913'
ALLOWED = ['fifth','eighth','boundary','reliable','integrated','expanded','free','no_sensing']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['smoke','development','heldout'], default='heldout')
    parser.add_argument('--variants', nargs='+', choices=ALLOWED, default=['integrated'])
    parser.add_argument('--case-limit', type=int, default=1)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.case_limit<1:parser.error('--case-limit must be positive')
    if len(set(args.variants))!=len(args.variants):parser.error('duplicate variants')
    manifest=json.loads((ENGINE/'results'/args.stage/'manifest.json').read_text(encoding='utf-8'))
    jobs=manifest['cases'] if args.full else manifest['cases'][:args.case_limit]
    out=Path(args.output).resolve() if args.output else ROOT/'runs'/('replay_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    jobfile=out/'jobs.json';jobfile.write_text(json.dumps(jobs),encoding='utf-8')
    print(f'Offline complete missions to run: {len(jobs)*len(args.variants)}',flush=True)
    comparisons=[]
    for variant in args.variants:
        result=subprocess.run([sys.executable,'-B',str(ENGINE/'experiment.py'),'--worker',variant,
                               '--jobs',str(jobfile),'--out',str(out)],cwd=ENGINE,
                              capture_output=True,text=True,encoding='utf-8',errors='replace',
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (out/(variant+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        print(result.stdout,end='',flush=True)
        if result.returncode:
            print(result.stderr)
            raise SystemExit(result.returncode)
        archived=ENGINE/'results'/args.stage/(variant+'.jsonl')
        previous={(r['seed'],r['mode'],r['length_scale_m']):r for r in map(json.loads,archived.read_text(encoding='utf-8').splitlines())} if archived.exists() else {}
        for row in map(json.loads,(out/(variant+'.jsonl')).read_text(encoding='utf-8').splitlines()):
            assert row['completed'] and row['clear_ratio']==1, row
            key=row['seed'],row['mode'],row['length_scale_m']
            record=dict(variant=variant,case=key,completed=True,matched_archived=None)
            if key in previous:
                old=previous[key]
                for name in ['measures','switches','cleared','clear_failures','fallback_attempts','stations_visited']:
                    assert row[name]==old[name],(variant,key,name,row[name],old[name])
                for name in ['virtual_time_s','distance_m','time_per_source_s']:
                    assert abs(row[name]-old[name])<1e-5,(variant,key,name)
                assert row['case_sha256']==old['case_sha256'] and row['error_samples']==old['error_samples']
                record['matched_archived']=True
            comparisons.append(record)
    (out/'replay_check.json').write_text(json.dumps(dict(stage=args.stage,complete_missions=len(comparisons),
                                                       comparisons=comparisons),ensure_ascii=False,indent=2),encoding='utf-8')
    print('New output:',out)

if __name__=='__main__':main()
