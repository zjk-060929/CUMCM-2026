"""Finite paired tests, with development selection preceding new held-out layouts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

from benchmark_adaptive_service_q4 import NAMES, describe, paired
from benchmark_correlation_q4 import AuditClient
from benchmark_efficiency_q4 import make_benchmark_case
from correlated_simulator import CorrelatedSimulator
from geometry import dist, polygon_distance
from precision_search_strategy import PrecisionSearchPolicy, VARIANTS

ROOT = Path('results/precision_search_20260913')
FILES = list(dict.fromkeys([*NAMES,'guarded_route_strategy.py','mission_cost_strategy.py',
                           'precision_search_strategy.py','benchmark_precision_search_q4.py']))


def hashes():
    return {n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in FILES}


def cases(split):
    if split=='development':
        groups=[('mixed',861000,7),('edge_outward',862000,1),
                ('rotated_cluster',863000,1),('minimum_range',864000,1)]
        scales=(100.,900.)
    elif split=='heldout':
        groups=[('mixed',871000,14),('edge_outward',872000,2),
                ('rotated_cluster',873000,2),('minimum_range',874000,2)]
        scales=(100.,300.,900.)
    else:
        return [(291001,'mixed',900.)]
    return [(s,m,l) for m,start,n in groups for s in range(start,start+n) for l in scales]


def run_one(seed, mode, scale, variant, trace=None):
    sim=CorrelatedSimulator(make_benchmark_case(seed,mode),seed,scale)
    client=AuditClient(sim,log_path=trace)
    policy=PrecisionSearchPolicy(client,**VARIANTS[variant])
    client.policy=policy
    try:
        result=policy.run()
        evaluation=sim.evaluation()
        truth={s['channel']:(s['x'],s['y']) for s in evaluation.pop('sources')}
        # Deduplicate unchanged snapshots; audit AFTER exit, never in policy.
        unique={(ch,tuple(p)) for ch,p in client.snapshots}
        unique.update((ch,tuple(t.polygon)) for ch,t in policy.tracks.items())
        violations=sum(polygon_distance(truth[ch],list(p))>1e-4 for ch,p in unique)
        assert violations==client.nearby_bearings==0
        if result['completed'] and len(policy.discovered)<16:
            assert sorted(policy.visited_ids)==list(range(21))
        profile={k:dict(v) for k,v in client.profile.items()}
        assert abs(sum(p['total_s'] for p in profile.values())-evaluation['virtual_time_s'])<1e-5
        points=[policy.station_points[i] for i in policy.visited_ids]
        station_distance=sum(dist(a,b) for a,b in zip([(0.,0.)]+points,points))
        if policy.station_limit is not None:
            assert station_distance<=policy.station_limit+1e-4
        return dict(seed=seed,mode=mode,length_scale_m=scale,variant=variant,
                    **{**result,**evaluation},action_profile=profile,
                    containment_violations=violations,nearby_bearings_le30m=client.nearby_bearings,
                    direct_station_path_m=station_distance,
                    extra_service_path_m=evaluation['distance_m']-station_distance)
    finally:
        client.close()


def summary(rows):
    report={}
    variants=list(dict.fromkeys(r['variant'] for r in rows))
    for mode in ['all','mixed','edge_outward','rotated_cluster','minimum_range']:
        group=[r for r in rows if mode=='all' or r['mode']==mode]
        if not group:continue
        values={}
        for v in variants:
            a=[r for r in group if r['variant']==v]
            d=describe(a)
            d.update(fallback_runs=sum(r['fallback_channels']>0 for r in a),
                     fallback_channels=sum(r['fallback_channels'] for r in a),
                     fallback_attempts_total=sum(r['fallback_attempts'] for r in a),
                     mean_plan_s=statistics.mean(r['schedule_planning_s'] for r in a),
                     max_replan_s=max(r['max_replan_runtime_s'] for r in a),
                     mean_cost_evaluations=statistics.mean(r['meta_evaluations'] for r in a))
            values[v]=d
        comparisons={f'{a}->{b}':paired(group,a,b) for a in variants[:2] for b in variants if a!=b}
        report[mode]=dict(summary=values,paired=comparisons)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=['development','heldout','replay'],required=True)
    ap.add_argument('--variants',nargs='+',choices=list(VARIANTS),required=True)
    a=ap.parse_args()
    out=ROOT/a.split
    if out.exists():raise SystemExit('Refusing to overwrite existing output')
    if a.split=='heldout':
        frozen=json.loads((ROOT/'frozen_config.json').read_text(encoding='utf-8'))
        assert a.variants==frozen['heldout_variants'] and hashes()==frozen['source_sha256']
    out.mkdir(parents=True)
    scenes=cases(a.split)
    manifest=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  test_kind='SELF_BUILT_SMOOTH_CORRELATED_OFFLINE',cases=scenes,
                  variants={v:VARIANTS[v] for v in a.variants},source_sha256=hashes(),
                  whole_mission_runs=len(scenes)*len(a.variants))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for n in FILES:shutil.copy2(n,snapshot/n)
    rows=[];started=time.perf_counter()
    with (out/'runs.jsonl').open('w',encoding='utf-8') as f:
        for v in a.variants:
            for j,(s,m,l) in enumerate(scenes):
                trace=out/f'trace_{v}_{s}_{int(l)}.jsonl' if a.split=='replay' else None
                row=run_one(s,m,l,v,trace)
                rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                if (j+1)%5==0 or j+1==len(scenes):
                    print(f'{v}: {j+1}/{len(scenes)}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
    report=dict(runs=len(rows),completed=sum(r['completed'] for r in rows),
                wall_runtime_s=time.perf_counter()-started,groups=summary(rows))
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({g:{v:round(s['time_per_source_s'],3) for v,s in d['summary'].items()}
                      for g,d in report['groups'].items()},indent=2))


if __name__=='__main__':main()
