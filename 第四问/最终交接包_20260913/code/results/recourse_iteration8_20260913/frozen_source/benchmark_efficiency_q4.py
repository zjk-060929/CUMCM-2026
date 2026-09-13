"""Iteration 3: action-level timing and finite paired offline comparisons."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from benchmark_refined_q4 import aggregate, compare, STRESS
from client import Client
from geometry import dist, polygon_distance
from planned_strategy import PlannedPolicy, PLANNED_CONFIG
from simulator import LocalSimulator, Source, make_case

ROOT = Path('results/efficiency_iteration3_20260913')
VARIANTS = {'previous':None, 'tight':dict(tight=True), 'local':dict(local=True),
            'tight_local':dict(tight=True, local=True),
            'stable_local':dict(tight=True, local=True, stable=True),
            'reuse':dict(tight=True, local=True, stable=True, reuse=True),
            'endpoint':dict(local=True, stable=True, endpoint=True)}


def make_benchmark_case(seed, mode):
    if mode != 'rotated_cluster':
        return make_case(seed, mode)
    angle = random.Random(seed+70513).uniform(0, 2*math.pi)
    co, si = math.cos(angle), math.sin(angle)
    return [Source(s.channel, co*s.x-si*s.y, si*s.x+co*s.y, s.radius,
                   None if s.orientation is None else (s.orientation+math.degrees(angle)) % 360)
            for s in make_case(seed, 'cluster')]


class ProfiledClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None
        self.profile = defaultdict(lambda:defaultdict(float))

    def action(self, path, position=None, channel=None):
        policy = self.policy
        phase, impossible = 'other', False
        if position is not None and policy is not None:
            if getattr(policy, 'reuse_busy', False):
                phase = 'replacement_scan'
            elif policy.in_station_scan:
                phase = 'station_unknown' if channel not in policy.discovered else 'station_known'
            elif getattr(policy, 'local_busy', False):
                phase = 'local_bundle'
            elif policy.reactive_busy:
                phase = 'reactive'
            elif policy.piggyback_busy:
                phase = 'piggyback'
            else:
                phase = 'post_service' if policy.coverage_complete else 'early_service'
            if path == '/clear' and channel in policy.tracks:
                impossible = polygon_distance(position, policy.tracks[channel].polygon) > 20.00001
        movement = dist(self.position, position)/5 if position is not None else 0.
        switch = float(path == '/measure' and channel != self.channel)
        reply = super().action(path, position, channel)
        if position is not None:
            p = self.profile[phase]
            p['movement_s'] += movement
            if path == '/measure':
                p['measurements'] += 1
                p['measurement_s'] += 5
                p['switch_s'] += switch
                p['negative_measurements'] += reply['measure_result'] == 'no_signal'
            else:
                success = reply['clear_result'] == 'success'
                p['clear_successes' if success else 'clear_failures'] += 1
                p['clear_s'] += 5 if success else 3
                p['provably_useless_clears'] += impossible
            p['total_s'] = sum(p.get(k, 0) for k in ('movement_s','measurement_s','switch_s','clear_s'))
            self.log('action_profile', phase=phase, path=path, movement_s=movement,
                     impossible_under_current_polygon=impossible)
        return reply


def run_one(seed, mode, error, variant, folder=None):
    sim = LocalSimulator(make_benchmark_case(seed, mode), seed, error)
    if folder:
        folder.mkdir(parents=True)
    client = ProfiledClient(sim, log_path=folder/'actions.jsonl' if folder else None)
    if variant == 'previous':
        policy = PlannedPolicy(client, **PLANNED_CONFIG)
    else:
        from efficient_strategy import EfficientPolicy
        policy = EfficientPolicy(client, **VARIANTS[variant])
    client.policy = policy
    try:
        result = policy.run()
        evaluation = sim.evaluation()
        truth = evaluation.pop('sources')
        row = {'seed':seed,'mode':mode,'error_mode':error,'variant':variant,**result,**evaluation}
        row['time_per_source_s'] = row['virtual_time_s']/row['source_count']
        row['action_profile'] = {k:dict(v) for k,v in client.profile.items()}
        reconstructed = sum(v['total_s'] for v in client.profile.values())
        assert abs(reconstructed-row['virtual_time_s']) < 1e-5
        station_path = [policy.station_points[i] for i in policy.visited_ids]
        direct = sum(dist(a,b) for a,b in zip([(0.,0.)]+station_path, station_path))
        row['direct_station_path_m'] = direct
        row['extra_service_path_m'] = row['distance_m']-direct
        if folder:
            (folder/'truth_after_exit.json').write_text(json.dumps(truth,indent=2),encoding='utf-8')
            (folder/'summary.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        return row
    finally:
        client.close()


def summarize(rows):
    answer = aggregate(rows)
    for key in ['direct_station_path_m','extra_service_path_m','local_bursts','local_discoveries',
                'local_clears','pruned_fallback_points','dynamic_station_skips']:
        a = sorted(r.get(key,0) for r in rows)
        answer[key] = dict(mean=statistics.mean(a),median=statistics.median(a),max=max(a))
    phases = sorted({p for r in rows for p in r['action_profile']})
    answer['profile_means'] = {}
    for phase in phases:
        keys = {k for r in rows for k in r['action_profile'].get(phase,{})}
        answer['profile_means'][phase] = {k:statistics.mean(r['action_profile'].get(phase,{}).get(k,0) for r in rows) for k in keys}
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=['development','heldout'],default='development')
    ap.add_argument('--variants',nargs='+',choices=VARIANTS,required=True)
    ap.add_argument('--baseline')
    ap.add_argument('--output',required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        ap.error('Use an empty new output folder')
    if args.split == 'heldout' and not (ROOT/'frozen_config.json').exists():
        ap.error('Freeze the configuration before held-out runs')
    out.mkdir(parents=True,exist_ok=True)
    rs,nr,ss,ns = (201000,28,202000,4) if args.split=='development' else (211000,126,212000,20)
    stress = STRESS+([('rotated_cluster','plus_one')] if args.split=='heldout' else [])
    cases = [(s,'mixed','fixed_hash') for s in range(rs,rs+nr)]
    cases += [(s,m,e) for m,e in stress for s in range(ss,ss+ns)]
    rows,started = [],time.perf_counter()
    with (out/'runs.jsonl').open('w',encoding='utf-8') as stream:
        for variant in args.variants:
            for index,(seed,mode,error) in enumerate(cases):
                trace = out/f'trace_{variant}_{mode}_{error}_{seed}' if args.split=='development' and seed in (rs,ss) else None
                row = run_one(seed,mode,error,variant,trace)
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                if (index+1)%28==0:
                    print(f'{variant}: {index+1}/{len(cases)}, {time.perf_counter()-started:.1f}s',flush=True)
            ordinary = [r for r in rows if r['variant']==variant and r['mode']=='mixed']
            print(json.dumps({'variant':variant,'mixed_per_source_s':statistics.mean(r['time_per_source_s'] for r in ordinary)}),flush=True)
    extra = [json.loads(s) for s in Path(args.baseline).read_text(encoding='utf-8').splitlines()] if args.baseline else []
    assert not extra or all(r['variant']=='previous' for r in extra)
    combined = extra+rows
    variants = list(dict.fromkeys(r['variant'] for r in combined))
    groups = {}
    for m,e in [('mixed','fixed_hash')]+stress:
        group = [r for r in combined if r['mode']==m and r['error_mode']==e]
        groups[m+'/'+e] = dict(summary={v:summarize([r for r in group if r['variant']==v]) for v in variants},
                             paired={v:compare(group,v) for v in variants if v!='previous'} if 'previous' in variants else {})
    names = ['geometry.py','strategy.py','optimized_geometry.py','optimized_strategy.py','refined_geometry.py',
             'refined_strategy.py','planned_strategy.py','efficient_strategy.py','simulator.py','client.py']
    report = dict(test_kind='SELF_BUILT_OFFLINE',split=args.split,new_runs=len(rows),reused_baseline_rows=len(extra),
                  variants={v:VARIANTS[v] for v in variants},random_seed_range=[rs,rs+nr-1],stress_seed_range=[ss,ss+ns-1],
                  source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names if Path(n).exists()},
                  groups=groups,wall_runtime_s=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'runs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps({g:{v:round(a['per_source_reduction_pct'],3) for v,a in d['paired'].items()} for g,d in groups.items()},indent=2))


if __name__=='__main__':
    main()
