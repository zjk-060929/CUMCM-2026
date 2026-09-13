"""Bounded iteration-4 development and frozen spatial-correlation validation."""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

from adaptive_service_strategy import AdaptiveServicePolicy, VARIANTS
from benchmark_correlation_q4 import AuditClient
from benchmark_efficiency_q4 import make_benchmark_case
from correlated_simulator import CorrelatedSimulator
from correlation_aware_strategy import CorrelationAwarePolicy
from geometry import dist, polygon_distance


ROOT = Path('results/adaptive_service_iteration4_20260913')
NAMES = ['adaptive_service_strategy.py', 'benchmark_adaptive_service_q4.py', 'correlation_aware_strategy.py',
         'correlated_simulator.py', 'efficient_strategy.py', 'planned_strategy.py', 'refined_strategy.py',
         'optimized_strategy.py', 'refined_geometry.py', 'optimized_geometry.py', 'strategy.py', 'geometry.py',
         'simulator.py', 'client.py', 'metaheuristic_strategy.py', 'heuristic_search.py',
         'benchmark_correlation_q4.py', 'benchmark_efficiency_q4.py']


def hashes():
    return {n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in NAMES}


def cases(split):
    if split == 'development':
        specs, scales = [('mixed',251000,14), ('edge_outward',252000,2),
                         ('rotated_cluster',253000,2), ('minimum_range',254000,2)], (100.,300.)
    else:
        specs, scales = [('mixed',261000,35), ('edge_outward',262000,5),
                         ('rotated_cluster',263000,5), ('minimum_range',264000,5)], (100.,300.,900.)
    return [(s,m,l) for m,start,n in specs for s in range(start,start+n) for l in scales]


def run_one(seed, mode, scale, variant, trace=None):
    sim = CorrelatedSimulator(make_benchmark_case(seed, mode), seed, scale)
    client = AuditClient(sim, log_path=trace)
    policy = (CorrelationAwarePolicy(client) if variant == 'current' else
              AdaptiveServicePolicy(client, **VARIANTS[variant]))
    client.policy = policy
    try:
        result = policy.run()
        evaluation = sim.evaluation()
        truth = {s['channel']:(s['x'],s['y']) for s in evaluation.pop('sources')}
        snapshots = client.snapshots+[(ch,t.polygon) for ch,t in policy.tracks.items()]
        violations = sum(polygon_distance(truth[ch],p) > 1e-4 for ch,p in snapshots)
        assert violations == 0
        assert client.nearby_bearings == 0
        if result['completed'] and len(policy.discovered) < 16:
            assert sorted(policy.visited_ids) == list(range(21))
        profile = {k:dict(v) for k,v in client.profile.items()}
        assert abs(sum(p['total_s'] for p in profile.values())-evaluation['virtual_time_s']) < 1e-5
        station_path = [policy.station_points[i] for i in policy.visited_ids]
        station_distance = sum(dist(a,b) for a,b in zip([(0.,0.)]+station_path,station_path))
        return dict(seed=seed,mode=mode,length_scale_m=scale,variant=variant,
                    **{**result,**evaluation}, action_profile=profile,
                    nearby_bearings_le30m=client.nearby_bearings,
                    containment_violations=violations, direct_station_path_m=station_distance,
                    extra_service_path_m=evaluation['distance_m']-station_distance)
    finally:
        client.close()


def describe(rows):
    names = ['virtual_time_s','time_per_source_s','distance_m','measures','switches','clear_failures',
             'stations_visited','endpoint_probes','endpoint_successes','correlated_measure_skips',
             'direct_station_path_m','extra_service_path_m','runtime_s','fallback_attempts']
    d = {k:statistics.mean(r.get(k,0) for r in rows) for k in names}
    a = sorted(r['time_per_source_s'] for r in rows)
    d.update(runs=len(rows), complete=sum(r['completed'] for r in rows),
             sources=sum(r['source_count'] for r in rows), median=statistics.median(a),
             p90=a[min(len(a)-1,int(.9*(len(a)-1)))], maximum=max(a),
             le400=sum(x<=400 for x in a),
             profile={k:statistics.mean(sum(p.get(k,0) for p in r['action_profile'].values()) for r in rows)
                      for k in ('movement_s','measurement_s','switch_s','clear_s')})
    return d


def paired(rows, before, after):
    a = {(r['seed'],r['length_scale_m']):r for r in rows if r['variant']==before}
    b = {(r['seed'],r['length_scale_m']):r for r in rows if r['variant']==after}
    diffs = {k:a[k]['time_per_source_s']-b[k]['time_per_source_s'] for k in a}
    # Average correlated scale replicates within each source layout before CI.
    clusters = [statistics.mean(d for (s,_),d in diffs.items() if s==seed) for seed in sorted({k[0] for k in a})]
    mean = statistics.mean(clusters)
    margin = 1.96*statistics.stdev(clusters)/len(clusters)**.5 if len(clusters)>1 else None
    return dict(mean_saved_per_source_s=statistics.mean(diffs.values()),
                reduction_pct=100*statistics.mean(diffs.values())/statistics.mean(r['time_per_source_s'] for r in a.values()),
                independent_source_layouts=len(clusters), layout_mean_normal_95ci=None if margin is None else [mean-margin,mean+margin],
                faster=sum(d>1e-6 for d in diffs.values()), slower=sum(d < -1e-6 for d in diffs.values()),
                ties=sum(abs(d)<=1e-6 for d in diffs.values()),
                worst_regressions=sorted([dict(seed=s,scale=l,extra_total_s=b[(s,l)]['virtual_time_s']-a[(s,l)]['virtual_time_s'],
                                             before_total_s=a[(s,l)]['virtual_time_s'], after_total_s=b[(s,l)]['virtual_time_s'],
                                             before_stations=a[(s,l)]['stations_visited'], after_stations=b[(s,l)]['stations_visited'])
                                          for s,l in a],key=lambda r:r['extra_total_s'],reverse=True)[:3])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', choices=['development','heldout'], required=True)
    ap.add_argument('--variants', nargs='+', choices=list(VARIANTS), required=True)
    args=ap.parse_args()
    out=ROOT/args.split
    if out.exists():
        raise SystemExit('Refusing to overwrite or rerun completed output')
    scenes=cases(args.split)
    frozen=ROOT/'frozen_config.json'
    if args.split=='heldout':
        config=json.loads(frozen.read_text(encoding='utf-8'))
        assert config['source_sha256']==hashes()
        assert args.variants==config['heldout_variants']
    out.mkdir(parents=True)
    (out/'pre_run_manifest.json').write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
             variants={v:VARIANTS[v] for v in args.variants}, cases=scenes, source_sha256=hashes()),indent=2),encoding='utf-8')
    rows,started=[],time.perf_counter()
    with (out/'runs.jsonl').open('w',encoding='utf-8') as stream:
        for variant in args.variants:
            for j,(seed,mode,scale) in enumerate(scenes):
                trace=out/f'trace_{variant}_{mode}_{seed}_{int(scale)}.jsonl' if args.split=='development' and j in (0,28,32,36) else None
                row=run_one(seed,mode,scale,variant,trace)
                rows.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if (j+1)%20==0:
                    print(f'{variant}: {j+1}/{len(scenes)}, batch={len(rows)}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
    groups={}
    for mode in ('mixed','edge_outward','rotated_cluster','minimum_range'):
        for scale in (None,)+tuple(sorted({l for _,_,l in scenes})):
            subset=[r for r in rows if r['mode']==mode and (scale is None or r['length_scale_m']==scale)]
            key=mode+'/'+('all_scales' if scale is None else str(scale))
            groups[key]=dict(summary={v:describe([r for r in subset if r['variant']==v]) for v in args.variants},
                  vs_current={v:paired(subset,'current',v) for v in args.variants if v!='current'} if 'current' in args.variants else {},
                  vs_reference={v:paired(subset,'reference',v) for v in args.variants if v!='reference'} if 'reference' in args.variants else {})
    report=dict(test_kind='SELF_BUILT_CORRELATED_OFFLINE', split=args.split,runs=len(rows),
                complete=sum(r['completed'] for r in rows),groups=groups,source_sha256=hashes(),
                wall_runtime_s=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'runs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps({g:{v:round(s['time_per_source_s'],3) for v,s in d['summary'].items()}
                      for g,d in groups.items() if g.endswith('all_scales')},indent=2))


if __name__=='__main__':
    main()
