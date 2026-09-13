"""Finite development ablations and independent spatial-error validation."""
import argparse,csv,hashlib,json,statistics,time
from pathlib import Path
from datetime import datetime

from benchmark_metaheuristic_q4 import SOURCES as BASE_SOURCES
from simulator import make_case
from spatial_error_q4 import SpatialErrorSimulator,SpatialBeamPolicy
from predictive_strategy_q4 import PredictiveClient,PredictivePolicy

VARIANTS={'previous':None,'discovery':dict(discovery=True,sensing=False,failure=False),
          'sensing':dict(discovery=False,sensing=True,failure=False),
          'failure':dict(discovery=False,sensing=False,failure=True),
          'all':dict(discovery=True,sensing=True,failure=True),
          'calibrated':dict(recovery_scale=1.5),
          'guarded':dict(recovery_scale=1.5,clear_probability_floor=.55),
          'balanced':dict(recovery_scale=2.,clear_probability_floor=.55,delay_weight=.03)}
SOURCES=BASE_SOURCES+['spatial_error_q4.py','bayesian_belief.py','predictive_models_q4.py',
                      'predictive_strategy_q4.py','benchmark_predictive_q4.py','run_predictive_q4.py']
ROOT=Path('results/predictive_costs_20260913')

def cases(split):
    if split=='development':
        return [(s,'mixed',scale) for scale in (100.,600.) for s in range(281000,281012)]+[(s,m,300.) for m in ('edge_outward','cluster','minimum_range') for s in range(282000,282004)]
    return [(s,'mixed',scale) for scale in (100.,300.,600.) for s in range(291000,291070)]+[(s,m,300.) for m in ('edge_outward','cluster','minimum_range') for s in range(292000,292008)]

def run_one(case,variant,out=None):
    seed,mode,scale=case
    sim=SpatialErrorSimulator(make_case(seed,mode),seed,scale)
    client=PredictiveClient(sim,log_path=out/'actions.jsonl' if out else None)
    policy=SpatialBeamPolicy(client) if variant=='previous' else PredictivePolicy(client,**VARIANTS[variant])
    try:
        result=policy.run();evaluation=sim.evaluation();truth=evaluation.pop('sources')
        row=dict(seed=seed,mode=mode,scale_m=scale,variant=variant,**{**result,**evaluation})
        times=list(client.first_discovery.values())
        row['mean_first_discovery_s']=statistics.mean(times) if times else None
        row['last_first_discovery_s']=max(times) if times else None
        forecast_pairs=[]
        for snapshot in getattr(policy,'discovery_forecasts',[]):
            delays=[client.first_discovery[ch]-snapshot['time_s'] for ch in snapshot['unknown_channels'] if ch in client.first_discovery]
            if delays:
                forecast_pairs.append((snapshot['mean_remaining_discovery_delay_s'],statistics.mean(delays)))
        row['discovery_forecast_validation']=dict(n=len(forecast_pairs),
            predicted_sum=sum(p for p,a in forecast_pairs),actual_sum=sum(a for p,a in forecast_pairs),
            absolute_error=sum(abs(p-a) for p,a in forecast_pairs))
        rebuilt=row['distance_m']/5+5*row['measures']+row['switches']+5*row['cleared']+3*row['clear_failures']
        assert abs(rebuilt-row['virtual_time_s'])<1e-5
        if out:
            (out/'truth_after_exit.json').write_text(json.dumps(truth,indent=2),encoding='utf-8')
            (out/'summary.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        return row
    finally:client.close()

def summarize(rows):
    groups={}
    variants=list(dict.fromkeys(r['variant'] for r in rows))
    for group in ('mixed','edge_outward','cluster','minimum_range'):
        part=[r for r in rows if r['mode']==group]
        baseline={(r['seed'],r['scale_m']):r for r in part if r['variant']=='previous'}
        summaries={}
        for v in variants:
            chosen=[r for r in part if r['variant']==v]
            if not chosen:continue
            summary=dict(runs=len(chosen),completed=sum(r['completed'] and r['clear_ratio']==1 for r in chosen))
            for key in ('time_per_source_s','virtual_time_s','distance_m','measures','clear_failures',
                        'fallback_attempts','runtime_s','mean_first_discovery_s','last_first_discovery_s','stations_visited'):
                values=sorted(r[key] for r in chosen)
                summary[key]=dict(mean=statistics.mean(values),median=statistics.median(values),p90=values[int(.9*(len(values)-1))],max=max(values))
            if baseline and v!='previous':
                delta=[baseline[r['seed'],r['scale_m']]['time_per_source_s']-r['time_per_source_s'] for r in chosen]
                # Rescale replication: confidence uses layout-level mean savings.
                layout={s:statistics.mean(d for r,d in zip(chosen,delta) if r['seed']==s) for s in {r['seed'] for r in chosen}}
                se=statistics.stdev(layout.values())/len(layout)**.5 if len(layout)>1 else 0.
                summary['paired']=dict(reduction_pct=100*sum(delta)/sum(baseline[r['seed'],r['scale_m']]['time_per_source_s'] for r in chosen),
                    mean_saved_per_source_s=statistics.mean(delta),faster=sum(d>1e-6 for d in delta),slower=sum(d< -1e-6 for d in delta),ties=sum(abs(d)<=1e-6 for d in delta),
                    independent_layouts=len(layout),layout_mean_normal_95pct_ci=[statistics.mean(layout.values())-1.96*se,statistics.mean(layout.values())+1.96*se])
            summaries[v]=summary
        groups[group]=summaries
    return groups

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=['development','heldout'],default='development')
    ap.add_argument('--variants',nargs='+',choices=VARIANTS,required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--baseline')
    args=ap.parse_args();out=Path(args.output)
    if out.exists() and any(out.iterdir()):ap.error('Use a new empty directory')
    out.mkdir(parents=True,exist_ok=True)
    if args.split=='heldout':
        frozen=json.loads((ROOT/'frozen_config.json').read_text(encoding='utf-8'))
        assert args.variants==['previous',frozen['selected_variant']]
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['source_sha256'].items())
    started=time.perf_counter();rows=[]
    with (out/'runs.jsonl').open('w',encoding='utf-8') as stream:
        for v in args.variants:
            for i,case in enumerate(cases(args.split)):
                log=out/f'trace_{v}_{case[1]}_{int(case[2])}_{case[0]}' if args.split=='development' and case[0] in (281000,282000) else None
                if log:log.mkdir()
                row=run_one(case,v,log);rows.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if (i+1)%6==0:print(v,i+1,'/',len(cases(args.split)),'elapsed',round(time.perf_counter()-started,1),flush=True)
            print(v,'mixed',statistics.mean(r['time_per_source_s'] for r in rows if r['variant']==v and r['mode']=='mixed'),flush=True)
    reused=[json.loads(x) for x in Path(args.baseline).read_text(encoding='utf-8-sig').splitlines()] if args.baseline else []
    assert all(r['variant']=='previous' for r in reused)
    report=dict(test_kind='SELF_BUILT_SPATIALLY_SMOOTH',split=args.split,new_runs=len(rows),reused_rows=len(reused),
                variants={v:VARIANTS[v] for v in args.variants},groups=summarize(reused+rows),wall_runtime_s=time.perf_counter()-started,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES})
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'runs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({g:{v:round(x.get('paired',{}).get('reduction_pct',0),3) for v,x in variants.items()} for g,variants in report['groups'].items()},indent=2))

if __name__=='__main__':main()
