"""Bounded paired experiments, isolated from other Q4 working branches."""
import argparse
import hashlib
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from simulator import make_case
from spatial_error_q4 import SpatialErrorSimulator
from predictive_strategy_q4 import PredictiveClient,PredictivePolicy,PREDICTIVE_CONFIG
from action_strategy import ActionPolicy

VARIANTS={
    'previous':None,
    'action':dict(opportunities=False,cover_risk=False),
    'opportunity':dict(cover_risk=False),
    'risk':dict(),
    'wide':dict(depth=5,width=12,branch=8),
    'deep':dict(depth=7,width=24,branch=10),
    'travel':dict(depth=5,width=12,branch=8,delay_weight=.02),
    'adaptive':dict(depth=5,width=12,branch=8,delay_weight=.02,adaptive_budget=True),
    'adaptive08':dict(depth=5,width=12,branch=8,delay_weight=.08,adaptive_budget=True),
    'cautious':dict(depth=5,width=12,branch=8,delay_weight=.02,adaptive_budget=True,conservative_reception=True),
    'opportunistic_unknown':dict(depth=5,width=12,branch=8,delay_weight=.02,adaptive_budget=True,unknown_probes=True),
}


def run_one(seed,mode,scale,variant,log=None):
    sim=SpatialErrorSimulator(make_case(seed,mode),seed,scale)
    client=PredictiveClient(sim,log_path=log)
    policy=PredictivePolicy(client,**PREDICTIVE_CONFIG) if variant=='previous' else ActionPolicy(client,**VARIANTS[variant])
    try:
        result=policy.run();evaluation=sim.evaluation();evaluation.pop('sources')
        row=dict(seed=seed,mode=mode,scale_m=scale,variant=variant,**{**result,**evaluation})
        times=list(client.first_discovery.values())
        row['mean_first_discovery_s']=statistics.mean(times) if times else None
        row['last_first_discovery_s']=max(times) if times else None
        rebuilt=row['distance_m']/5+5*row['measures']+row['switches']+5*row['cleared']+3*row['clear_failures']
        assert abs(rebuilt-row['virtual_time_s'])<1e-5
        if variant!='previous':assert row['decisions']==row['measures']+row['cleared']+row['clear_failures']
        return row
    finally:client.close()


def summary(rows):
    result={}
    for mode in sorted({r['mode'] for r in rows}):
        result[mode]={}
        for variant in sorted({r['variant'] for r in rows}):
            part=[r for r in rows if r['mode']==mode and r['variant']==variant]
            if not part:continue
            item=dict(runs=len(part),completed=sum(r['completed'] and r['clear_ratio']==1 for r in part))
            for k in ('time_per_source_s','virtual_time_s','distance_m','measures','switches','clear_failures','fallback_attempts','runtime_s','mean_first_discovery_s','last_first_discovery_s','stations_visited'):
                values=sorted(r[k] for r in part if r[k] is not None)
                item[k]=dict(mean=statistics.mean(values),median=statistics.median(values),p90=values[int(.9*(len(values)-1))],max=max(values))
            result[mode][variant]=item
    return result


def dispatch(job):return run_one(*job)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variants',nargs='+',default=['previous','risk','wide'],choices=VARIANTS)
    ap.add_argument('--seed',type=int,default=321000);ap.add_argument('--count',type=int,default=6)
    ap.add_argument('--modes',nargs='+',default=['mixed','edge_outward','cluster','minimum_range'])
    ap.add_argument('--scales',nargs='+',type=float,default=[300.]);ap.add_argument('--output',required=True)
    ap.add_argument('--trace',action='store_true')
    ap.add_argument('--workers',type=int,default=1)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'runs.jsonl').exists():ap.error('Use a fresh output directory')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}
    (out/'config.json').write_text(json.dumps(dict(arguments=vars(args),variants=VARIANTS,source_sha256=hashes),indent=2),encoding='utf-8')
    rows=[];started=time.perf_counter()
    jobs=[(seed,mode,scale,variant,out/f'{variant}_{mode}_{seed}_{scale}.jsonl' if args.trace else None)
          for mode in args.modes for seed in range(args.seed,args.seed+args.count)
          for scale in args.scales for variant in args.variants]
    pool=ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else None
    results=pool.map(dispatch,jobs) if pool else map(dispatch,jobs)
    with (out/'runs.jsonl').open('w',encoding='utf-8') as f:
        for row in results:
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
            print(row['seed'],row['mode'],row['scale_m'],row['variant'],round(row['time_per_source_s'],2),row['completed'],round(row['runtime_s'],2),'elapsed',round(time.perf_counter()-started,1),flush=True)
    if pool:pool.shutdown()
    (out/'summary.json').write_text(json.dumps(dict(new_runs=len(rows),groups=summary(rows),wall_s=time.perf_counter()-started),indent=2),encoding='utf-8')


if __name__=='__main__':main()
