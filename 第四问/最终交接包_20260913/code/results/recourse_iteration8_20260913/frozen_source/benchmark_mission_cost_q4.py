"""Iteration-6 finite development and frozen paired validation."""
import argparse
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

from adaptive_service_strategy import AdaptiveServicePolicy
from benchmark_adaptive_service_q4 import NAMES as PRIOR_NAMES, describe, paired
from benchmark_correlation_q4 import AuditClient
from benchmark_efficiency_q4 import make_benchmark_case
from correlated_simulator import CorrelatedSimulator
from geometry import dist,polygon_distance
from guarded_route_strategy import GuardedRoutePolicy
from mission_cost_strategy import MissionCostPolicy,VARIANTS


ROOT=Path('results/mission_cost_iteration6_20260913')
NAMES=list(dict.fromkeys(['mission_cost_strategy.py','benchmark_mission_cost_q4.py','test_mission_cost_q4.py','run_mission_cost_q4.py','guarded_route_strategy.py',*PRIOR_NAMES]))


def hashes():return {n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in NAMES}


def cases(split):
    if split=='development':
        specs=[('mixed',301000,14),('edge_outward',302000,2),('rotated_cluster',303000,2),('minimum_range',304000,2)]
        scales=(100.,300.)
    else:
        specs=[('mixed',311000,35),('edge_outward',312000,5),('rotated_cluster',313000,5),('minimum_range',314000,5)]
        scales=(100.,300.,900.)
    return [(s,m,l) for m,start,n in specs for s in range(start,start+n) for l in scales]


def run_one(seed,mode,scale,variant,trace=None):
    sim=CorrelatedSimulator(make_benchmark_case(seed,mode),seed,scale)
    client=AuditClient(sim,log_path=trace)
    if variant=='previous':policy=GuardedRoutePolicy(client,station_margin=500.,skip_gain=.08)
    elif variant=='reference':policy=AdaptiveServicePolicy(client,order='information',endpoint_gate='off',local=False)
    else:policy=MissionCostPolicy(client,**VARIANTS[variant])
    client.policy=policy
    try:
        result=policy.run()
        evaluation=sim.evaluation()
        truth={s['channel']:(s['x'],s['y']) for s in evaluation.pop('sources')}
        snapshots=client.snapshots+[(ch,t.polygon) for ch,t in policy.tracks.items()]
        violations=sum(polygon_distance(truth[ch],p)>1e-4 for ch,p in snapshots)
        assert violations==client.nearby_bearings==0
        if result['completed'] and len(policy.discovered)<16:assert sorted(policy.visited_ids)==list(range(21))
        profile={k:dict(v) for k,v in client.profile.items()}
        assert abs(sum(p['total_s'] for p in profile.values())-evaluation['virtual_time_s'])<1e-5
        points=[policy.station_points[i] for i in policy.visited_ids]
        station_distance=sum(dist(a,b) for a,b in zip([(0.,0.)]+points,points))
        limit=result.get('station_limit_m')
        if limit is not None:assert station_distance<=limit+1e-4
        return dict(seed=seed,mode=mode,length_scale_m=scale,variant=variant,**{**result,**evaluation},
                    action_profile=profile,nearby_bearings_le30m=client.nearby_bearings,
                    containment_violations=violations,direct_station_path_m=station_distance,
                    extra_service_path_m=evaluation['distance_m']-station_distance)
    finally:client.close()


def describe_extra(rows):
    a=describe(rows)
    for k in ('rejected_route_candidates','low_gain_measure_skips','maximum_accepted_station_potential_m'):
        a[k]=statistics.mean(r.get(k,0) for r in rows)
    a['maximum_station_path_m']=max(r['direct_station_path_m'] for r in rows)
    return a


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=['development','heldout'],required=True)
    ap.add_argument('--variants',nargs='+',choices=list(VARIANTS),required=True)
    args=ap.parse_args()
    out=ROOT/args.split
    if out.exists():raise SystemExit('Refusing to overwrite/repeat output')
    if args.split=='heldout':
        frozen=json.loads((ROOT/'frozen_config.json').read_text())
        assert hashes()==frozen['source_sha256'] and args.variants==frozen['heldout_variants']
    scenes=cases(args.split)
    out.mkdir(parents=True)
    (out/'pre_run_manifest.json').write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
             variants={v:VARIANTS[v] for v in args.variants},cases=scenes,source_sha256=hashes()),indent=2),encoding='utf-8')
    rows,started=[],time.perf_counter()
    with (out/'runs.jsonl').open('w',encoding='utf-8') as stream:
        for variant in args.variants:
            for j,(seed,mode,scale) in enumerate(scenes):
                trace=out/f'trace_{variant}_{mode}_{seed}_{int(scale)}.jsonl' if args.split=='development' and j in (0,28,32,36) else None
                row=run_one(seed,mode,scale,variant,trace)
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                if (j+1)%20==0:print(f'{variant}: {j+1}/{len(scenes)}, batch={len(rows)}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
    groups={}
    for mode in ('mixed','edge_outward','rotated_cluster','minimum_range'):
        for scale in (None,)+tuple(sorted({l for _,_,l in scenes})):
            a=[r for r in rows if r['mode']==mode and (scale is None or r['length_scale_m']==scale)]
            key=mode+'/'+('all_scales' if scale is None else str(scale))
            groups[key]=dict(summary={v:describe_extra([r for r in a if r['variant']==v]) for v in args.variants},
                       vs_previous={v:paired(a,'previous',v) for v in args.variants if v!='previous'} if 'previous' in args.variants else {},
                       vs_reference={v:paired(a,'reference',v) for v in args.variants if v!='reference'} if 'reference' in args.variants else {})
    report=dict(test_kind='SELF_BUILT_CORRELATED_OFFLINE',split=args.split,runs=len(rows),
                complete=sum(r['completed'] for r in rows),groups=groups,source_sha256=hashes(),wall_runtime_s=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'runs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps({g:{v:round(s['time_per_source_s'],3) for v,s in d['summary'].items()}
                     for g,d in groups.items() if g.endswith('all_scales')},indent=2))


if __name__=='__main__':main()
