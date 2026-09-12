"""Paired self-generated benchmarks, independent of the official case generator."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time
from client import Client
from simulator import LocalSimulator,make_case
from strategy import Policy

VARIANTS={'square_optical':('square',False),'square_active':('square',True),
          'triangle_optical':('triangle',False),'triangle_active':('triangle',True)}

def percentile(values,p):
    x=sorted(values); i=(len(x)-1)*p; k=int(i)
    return x[k]+(x[min(k+1,len(x)-1)]-x[k])*(i-k)

def summary(rows):
    keys=['virtual_time_s','time_per_source_s','distance_m','measures','switches','clear_failures',
          'runtime_s','fallback_channels','fallback_attempts','no_signal_localization','stations_visited']
    out={'runs':len(rows),'all_cleared_runs':sum(r['clear_ratio']==1 and r['completed'] for r in rows),
         'total_sources':sum(r['source_count'] for r in rows),'total_cleared':sum(r['cleared'] for r in rows)}
    for key in keys:
        v=[r[key] for r in rows]
        out[key]={'mean':statistics.mean(v),'median':statistics.median(v),'p90':percentile(v,.9),'max':max(v)}
    return out

def run_one(seed,mode,error,variant,log_dir=None):
    sim=LocalSimulator(make_case(seed,mode),seed,error,capture=False)
    c=Client(sim,log_path=log_dir/'actions.jsonl' if log_dir else None)
    try:
        grid,active=VARIANTS[variant]
        result=Policy(c,grid,active).run()
        evaluation=sim.evaluation()
        truth=evaluation.pop('sources')
        row={'case_id':f'SELF-Q4-{mode}-{error}-{seed}','seed':seed,'mode':mode,'error_mode':error,'variant':variant,
             **result,**evaluation}
        reconstructed=row['distance_m']/5+row['measures']*5+row['switches']+row['cleared']*5+row['clear_failures']*3
        if abs(row['virtual_time_s']-reconstructed)>1e-5: raise AssertionError('Timing accounting mismatch')
        if not row['completed'] or row['clear_ratio']!=1: raise AssertionError(f'Incomplete case: {row}')
        if log_dir:
            (log_dir/'truth_after_exit.json').write_text(json.dumps(truth,indent=2))
            (log_dir/'summary.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
        return row
    finally: c.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--random-cases',type=int,default=500)
    ap.add_argument('--stress-per-group',type=int,default=50)
    ap.add_argument('--output',default='results')
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    rows=[]; start=time.perf_counter()
    for variant in VARIANTS:
        for seed in range(a.random_cases): rows.append(run_one(seed,'mixed','fixed_hash',variant))
        print(f'{variant}: {a.random_cases} paired cases complete',flush=True)
    stress_groups=[('edge_outward','plus_one'),('edge_outward','minus_one'),('minimum_range','smooth'),
                   ('minimum_range','checker'),('cluster','plus_one'),('near_origin','fixed_hash'),
                   ('all_directional','checker'),('all_omni','smooth')]
    stress=[]
    for mode,error in stress_groups:
        for seed in range(10000,10000+a.stress_per_group):
            stress.append(run_one(seed,mode,error,'triangle_active'))
        print(f'stress {mode}/{error} complete',flush=True)
    output_summary={'test_kind':'SELF_BUILT_OFFLINE','random_cases':a.random_cases,
                    'random_seed_range':[0,a.random_cases-1],'stress_seed_range':[10000,9999+a.stress_per_group],
                    'variants':{v:summary([r for r in rows if r['variant']==v]) for v in VARIANTS},
                    'stress':{m+'/'+e:summary([r for r in stress if r['mode']==m and r['error_mode']==e]) for m,e in stress_groups}}
    primary=[r for r in rows if r['variant']=='triangle_active']
    comparisons={}
    for v in VARIANTS:
        if v=='triangle_active': continue
        other=[r for r in rows if r['variant']==v]
        delta=[b['virtual_time_s']-p['virtual_time_s'] for b,p in zip(other,primary)]
        mean=statistics.mean(delta); se=statistics.stdev(delta)/math.sqrt(len(delta)) if len(delta)>1 else 0
        comparisons[v]={'mean_saved_seconds':mean,'normal_approx_95pct_CI':[mean-1.96*se,mean+1.96*se],
                        'mean_time_reduction_pct':100*mean/statistics.mean(r['virtual_time_s'] for r in other),
                        'primary_faster_cases':sum(x>0 for x in delta)}
    output_summary['paired_comparisons']=comparisons
    output_summary['wall_runtime_s']=time.perf_counter()-start
    (out/'benchmark_summary.json').write_text(json.dumps(output_summary,ensure_ascii=False,indent=2))
    for name,records in [('paired_cases.csv',rows),('stress_cases.csv',stress)]:
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    worst=max(primary,key=lambda r:r['virtual_time_s'])
    sample_seeds=sorted({42,worst['seed'],next(r['seed'] for r in primary if r['source_count']==16)})
    for seed in sample_seeds:
        directory=out/f'sample_seed_{seed}'; directory.mkdir(exist_ok=True)
        run_one(seed,'mixed','fixed_hash','triangle_active',directory)
    print(json.dumps({'random_runs':len(rows),'stress_runs':len(stress),'wall_runtime_s':output_summary['wall_runtime_s'],
                      'worst_primary_seed':worst['seed']},indent=2))

if __name__=='__main__': main()
