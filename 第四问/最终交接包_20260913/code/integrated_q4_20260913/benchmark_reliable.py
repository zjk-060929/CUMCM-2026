"""Second, new validation set after the first validation found a real failure."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from run_reliable import run_one
from benchmark_action import summary


def dispatch(job):return run_one(*job)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stress',action='store_true');ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'runs.jsonl').exists():ap.error('Use a fresh output directory')
    frozen=json.loads(Path('results/reliable_frozen_config.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['source_sha256'].items())
    cases=([(s,m,300.) for m in ('edge_outward','cluster','minimum_range') for s in range(334000,334004)] if args.stress
           else [(s,'mixed',scale) for s in range(333000,333021) for scale in (100.,300.,600.)])
    jobs=[(*case,v) for case in cases for v in ('previous','reliable')];rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool,(out/'runs.jsonl').open('w',encoding='utf-8') as f:
        for row in pool.map(dispatch,jobs):
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
            print(row['seed'],row['mode'],row['scale_m'],row['variant'],round(row['time_per_source_s'],2),row['completed'],flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['source_sha256'].items())
    (out/'summary.json').write_text(json.dumps(dict(new_runs=len(rows),groups=summary(rows),wall_s=time.perf_counter()-started),indent=2),encoding='utf-8')


if __name__=='__main__':main()
