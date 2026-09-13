"""Same-layout historical comparison and bounded boundary-rescue development."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import time
import math

from benchmark_adaptive_service_q4 import describe,paired
from benchmark_precision_search_q4 import FILES as PRECISION_FILES
from benchmark_efficiency_q4 import make_benchmark_case
from benchmark_correlation_q4 import AuditClient
from boundary_rescue_strategy import BoundaryRescuePolicy
from correlated_simulator import CorrelatedSimulator
from geometry import polygon_distance,dist
from guarded_route_strategy import GuardedRoutePolicy
from mission_cost_strategy import MissionCostPolicy
from service_risk_strategy import ServiceRiskPolicy
from precision_search_strategy import PrecisionSearchPolicy
from simulator import Source

ROOT=Path('results/boundary_rescue_20260913')
FILES=list(dict.fromkeys([*PRECISION_FILES,'service_risk_strategy.py',
                         'boundary_rescue_strategy.py','benchmark_boundary_rescue_q4.py']))
VARIANTS=('fifth','sixth','seventh','precision1024','optical','cross')


def hashes():return {n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in FILES}


def make_case(seed,mode):
    if mode not in ('boundary_shell','boundary_inward','boundary_oblique'):return make_benchmark_case(seed,mode)
    rng=random.Random(seed+9013);sources=make_benchmark_case(seed,'edge_outward')
    output=[]
    for s in sources:
        angle=math.atan2(s.y,s.x)
        radius=rng.uniform(1700.,1780.) if mode=='boundary_shell' else 1799.
        orientation=s.orientation
        if orientation is not None and mode=='boundary_inward':orientation=(orientation+180.)%360
        if orientation is not None and mode=='boundary_oblique':orientation=(orientation+rng.uniform(-65.,65.))%360
        output.append(Source(s.channel,radius*math.cos(angle),radius*math.sin(angle),s.radius,orientation))
    return output


def factory(client,variant):
    if variant=='fifth':return GuardedRoutePolicy(client,station_margin=500.,skip_gain=.08)
    if variant=='sixth':return MissionCostPolicy(client,travel_discount=0.,detour_weight=.5,effect_blend=.5)
    if variant=='seventh':return ServiceRiskPolicy(client,return_weight=.5,wait_mode='risk')
    if variant=='precision1024':return PrecisionSearchPolicy(client,circle_sides=1024,search_level='standard',joint_cost=False)
    return BoundaryRescuePolicy(client,rescue=variant)


def cases(split):
    if split=='development':
        groups=[('mixed',911000,4),('edge_outward',912000,4),('rotated_cluster',913000,2),
                ('minimum_range',914000,2),('boundary_shell',915000,2)]
        scales=(100.,900.)
    elif split=='heldout':
        groups=[('mixed',921000,14),('edge_outward',922000,6),('rotated_cluster',923000,2),
                ('minimum_range',924000,2),('boundary_shell',925000,3),('boundary_inward',926000,2),
                ('boundary_oblique',927000,2)]
        scales=(100.,300.,900.)
    else:return [(292002,'edge_outward',300.),(292003,'edge_outward',300.),(872001,'edge_outward',300.)]
    return [(s,m,l) for m,start,n in groups for s in range(start,start+n) for l in scales]


def run_one(seed,mode,scale,variant,trace=None):
    sim=CorrelatedSimulator(make_case(seed,mode),seed,scale)
    client=AuditClient(sim,log_path=trace);policy=factory(client,variant);client.policy=policy
    try:
        result=policy.run();evaluation=sim.evaluation()
        truth={s['channel']:(s['x'],s['y']) for s in evaluation.pop('sources')}
        regions={(ch,tuple(p)) for ch,p in client.snapshots}
        regions.update((ch,tuple(t.polygon)) for ch,t in policy.tracks.items())
        violations=sum(polygon_distance(truth[ch],list(p))>1e-4 for ch,p in regions)
        assert violations==client.nearby_bearings==0
        if result['completed'] and len(policy.discovered)<16:assert sorted(policy.visited_ids)==list(range(21))
        profile={k:dict(v) for k,v in client.profile.items()}
        assert abs(sum(p['total_s'] for p in profile.values())-evaluation['virtual_time_s'])<1e-5
        points=[policy.station_points[i] for i in policy.visited_ids]
        distance=sum(dist(a,b) for a,b in zip([(0.,0.)]+points,points))
        return dict(seed=seed,mode=mode,length_scale_m=scale,variant=variant,
                    **{**result,**evaluation},containment_violations=violations,
                    nearby_bearings_le30m=client.nearby_bearings,action_profile=profile,
                    direct_station_path_m=distance,extra_service_path_m=evaluation['distance_m']-distance)
    finally:client.close()


def summarize(rows):
    groups={};variants=list(dict.fromkeys(r['variant'] for r in rows))
    for mode in ['all',*dict.fromkeys(r['mode'] for r in rows)]:
        a=[r for r in rows if mode=='all' or r['mode']==mode];values={}
        for v in variants:
            b=[r for r in a if r['variant']==v];d=describe(b)
            d.update(fallback_runs=sum(r['fallback_channels']>0 for r in b),
                     fallback_channels=sum(r['fallback_channels'] for r in b),
                     fallback_attempts_total=sum(r['fallback_attempts'] for r in b),
                     boundary_episodes=sum(r.get('boundary_episodes',0) for r in b),
                     boundary_successes=sum(r.get('boundary_successes',0) for r in b),
                     boundary_optical_attempts=sum(r.get('boundary_optical_attempts',0) for r in b),
                     boundary_bearings=sum(r.get('boundary_bearings',0) for r in b),
                     boundary_cost_s=sum(r.get('boundary_cost_s',0) for r in b))
            values[v]=d
        comparisons={f'{x}->{y}':paired(a,x,y) for x in variants for y in variants if x!=y}
        groups[mode]=dict(summary=values,paired=comparisons)
    return groups


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=['development','heldout','replay'],required=True)
    ap.add_argument('--variants',nargs='+',choices=VARIANTS,required=True)
    a=ap.parse_args();out=ROOT/a.split
    if out.exists():raise SystemExit('Refusing to overwrite results')
    if a.split=='heldout':
        frozen=json.loads((ROOT/'frozen_config.json').read_text(encoding='utf-8'))
        assert a.variants==frozen['heldout_variants'] and hashes()==frozen['source_sha256']
    out.mkdir(parents=True);scenes=cases(a.split)
    manifest=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),variants=a.variants,cases=scenes,
                  runs=len(scenes)*len(a.variants),source_sha256=hashes(),
                  kind='SELF_BUILT_SMOOTH_CORRELATED_OFFLINE')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for n in FILES:shutil.copy2(n,snapshot/n)
    started=time.perf_counter();rows=[]
    with (out/'runs.jsonl').open('w',encoding='utf-8') as f:
        for v in a.variants:
            for j,(seed,mode,scale) in enumerate(scenes):
                trace=out/f'trace_{v}_{seed}_{int(scale)}.jsonl' if a.split=='replay' else None
                row=run_one(seed,mode,scale,v,trace)
                rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                if (j+1)%14==0 or j+1==len(scenes):
                    print(f'{v}: {j+1}/{len(scenes)}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
    result=dict(runs=len(rows),completed=sum(r['completed'] for r in rows),
                wall_runtime_s=time.perf_counter()-started,groups=summarize(rows))
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({m:{v:dict(mean=round(s['time_per_source_s'],3),fallback=s['fallback_attempts_total'],
                    rescue=s['boundary_successes']) for v,s in g['summary'].items()}
                    for m,g in result['groups'].items()},indent=2))


if __name__=='__main__':main()
