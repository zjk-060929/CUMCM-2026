"""Finite paired research benchmark: baseline, heuristic families and ablations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

from benchmark_refined_q4 import aggregate,compare,STRESS
from client import Client
from simulator import LocalSimulator,make_case
from planned_strategy import PlannedPolicy,PLANNED_CONFIG
from metaheuristic_strategy import MetaheuristicPolicy

VARIANTS = {
    'previous':None,
    'vnd_time':dict(method='vnd'),
    'ils_time':dict(method='ils'),
    'alns_geometry':dict(method='alns',uncertainty_weight=0.),
    'alns_time':dict(method='alns'),
    'beam_time':dict(method='beam'),
    'alns_local':dict(method='alns',local_sweep=True),
    'beam_info':dict(method='beam',discovery_model=True),
    'alns_info':dict(method='alns',discovery_model=True),
    'beam_info_low':dict(method='beam',discovery_model=True,uncertainty_weight=.1),
    'beam_info_high':dict(method='beam',discovery_model=True,uncertainty_weight=1.2),
}
SOURCES = ['geometry.py','strategy.py','optimized_geometry.py','optimized_strategy.py',
           'refined_geometry.py','refined_strategy.py','planned_strategy.py',
           'heuristic_search.py','metaheuristic_strategy.py','simulator.py','client.py',
           'benchmark_metaheuristic_q4.py','run_metaheuristic_q4.py']


def run_one(seed,mode,error,variant,out=None):
    sim = LocalSimulator(make_case(seed,mode),seed,error)
    client = Client(sim,log_path=out/'actions.jsonl' if out else None)
    policy = PlannedPolicy(client,**PLANNED_CONFIG) if variant == 'previous' else MetaheuristicPolicy(client,**VARIANTS[variant])
    try:
        result = policy.run()
        evaluation = sim.evaluation()
        truth = evaluation.pop('sources')
        row = {'seed':seed,'mode':mode,'error_mode':error,'variant':variant,**result,**evaluation}
        rebuilt = row['distance_m']/5+5*row['measures']+row['switches']+5*row['cleared']+3*row['clear_failures']
        assert abs(rebuilt-row['virtual_time_s']) < 1e-5
        row['time_per_source_s'] = row['virtual_time_s']/row['source_count']
        if out:
            (out/'truth_after_exit.json').write_text(json.dumps(truth,indent=2),encoding='utf-8')
            (out/'summary.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        return row
    finally:
        client.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=['development','heldout'],default='development')
    ap.add_argument('--variants',nargs='+',choices=VARIANTS,required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--baseline')
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        ap.error('Use a new, empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    if args.split == 'heldout':
        frozen = Path('results/metaheuristics_20260913/frozen_config.json')
        if not frozen.exists():
            ap.error('Freeze configuration before held-out evaluation')
        config = json.loads(frozen.read_text(encoding='utf-8'))
        assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in config['source_sha256'].items())
        assert args.variants == ['previous',config['selected_variant']]
    rs,nr,ss,ns = (131000,24,132000,4) if args.split == 'development' else (211000,196,212000,20)
    cases = [(s,'mixed','fixed_hash') for s in range(rs,rs+nr)]
    cases += [(s,m,e) for m,e in STRESS for s in range(ss,ss+ns)]
    rows,started = [],time.perf_counter()
    with (out/'runs.jsonl').open('w',encoding='utf-8') as stream:
        for variant in args.variants:
            for i,(seed,mode,error) in enumerate(cases):
                trace = out/f'trace_{variant}_{mode}_{error}_{seed}' if args.split=='development' and seed in (rs,ss) else None
                if trace:
                    trace.mkdir()
                row = run_one(seed,mode,error,variant,trace)
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                if (i+1) % 10 == 0:
                    print(f'{variant} {i+1}/{len(cases)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
            ordinary = [r for r in rows if r['variant']==variant and r['mode']=='mixed']
            print(json.dumps({'variant':variant,'mixed_mean_s':statistics.mean(r['time_per_source_s'] for r in ordinary),
                              'completed':sum(r['completed'] for r in rows if r['variant']==variant)}),flush=True)
    reused = [json.loads(x) for x in Path(args.baseline).read_text(encoding='utf-8').splitlines()] if args.baseline else []
    assert not reused or all(r['variant']=='previous' for r in reused)
    combined = reused+rows
    variants = list(dict.fromkeys(r['variant'] for r in combined))
    groups = {}
    for mode,error in [('mixed','fixed_hash')]+STRESS:
        part = [r for r in combined if r['mode']==mode and r['error_mode']==error]
        groups[mode+'/'+error] = dict(summary={v:aggregate([r for r in part if r['variant']==v]) for v in variants},
                                      paired={v:compare(part,v) for v in variants if v!='previous'} if 'previous' in variants else {})
    report = dict(test_kind='SELF_BUILT_OFFLINE',split=args.split,new_runs=len(rows),reused_rows=len(reused),
                  variants={v:VARIANTS[v] for v in variants},random_seed_range=[rs,rs+nr-1],
                  stress_seed_range=[ss,ss+ns-1],groups=groups,wall_runtime_s=time.perf_counter()-started,
                  source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in SOURCES})
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'runs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({g:{v:round(x['per_source_reduction_pct'],3) for v,x in a['paired'].items()} for g,a in groups.items()},indent=2))


if __name__ == '__main__':
    main()
