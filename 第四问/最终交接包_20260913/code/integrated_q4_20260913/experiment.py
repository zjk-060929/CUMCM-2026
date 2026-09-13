"""Finite, common-scene experiments with separate processes for each source tree."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
Q4=ROOT.parent
SOURCES={
    'fifth':Q4/'results/recourse_iteration8_20260913/frozen_source',
    'eighth':Q4/'results/recourse_iteration8_20260913/frozen_source',
    'boundary':Q4/'results/boundary_rescue_20260913/heldout/source_snapshot',
    'reliable':Q4/'action_planner_20260913/results/reliable_frozen_source',
}
VARIANTS={
    'integrated':dict(search='standard',patrol='gated'),
    'expanded':dict(search='expanded',patrol='gated'),
    'free':dict(search='standard',patrol='off'),
    'no_sensing':dict(search='standard',patrol='gated',sensing=False),
}


def hashes(root):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}


def scenarios(stage):
    if stage=='development':
        groups=[('mixed',1011000,7),('edge_outward',1011100,2),('rotated_cluster',1011200,1),('minimum_range',1011300,1)]
        scales=(100.,900.)
    elif stage=='heldout':
        groups=[('mixed',1012000,14),('edge_outward',1012100,3),('rotated_cluster',1012200,2),
                ('minimum_range',1012300,2),('edge_inset',1012400,2)]
        scales=(100.,300.,900.)
    else:
        return [(42,'mixed',300.),(292002,'edge_outward',300.),(291017,'mixed',300.)]
    return [(seed,mode,scale) for mode,start,count in groups for seed in range(start,start+count) for scale in scales]


def worker(variant, jobs, out):
    source=SOURCES.get(variant,ROOT)
    sys.path.insert(0,str(source))
    from simulator import make_case, Source
    from common_simulator import CorrelatedSimulator
    from geometry import dist, polygon_distance
    if variant in SOURCES and variant!='reliable':
        from client import Client as BaseClient
    else:
        from predictive_strategy_q4 import PredictiveClient as BaseClient

    class AuditClient(BaseClient):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.policy=None;self.truth={};self.checked={};self.containment=0;self.nearby=0
            self.first={};self.phase='other';self.parts={}

        def check(self):
            if self.policy is None:return
            for ch,t in self.policy.tracks.items():
                signature=(len(t.observations),tuple(t.polygon))
                if self.checked.get(ch)==signature:continue
                self.checked[ch]=signature
                self.containment+=int(polygon_distance(self.truth[ch],t.polygon)>1e-4)
                self.nearby+=sum(dist(a,b)<30.-1e-7 for i,(a,_) in enumerate(t.observations)
                                 for b,_ in t.observations[:i])

        def action(self,path,p=None,ch=None):
            self.check()
            old=self.virtual;origin=self.position;channel=self.channel
            policy=self.policy
            fallback_before=(sum(t.fallback_attempts for t in policy.tracks.values()) if policy else 0)
            reply=super().action(path,p,ch)
            if p is not None:
                move=dist(origin,p)/5
                part=self.parts.setdefault(self.phase,dict(movement_s=0.,operation_s=0.,actions=0))
                part['movement_s']+=move;part['operation_s']+=self.virtual-old-move;part['actions']+=1
                if path=='/measure' and reply['measure_result']!='no_signal':self.first.setdefault(ch,self.virtual)
            return reply

    def case(seed,mode):
        if mode=='rotated_cluster':
            a=random.Random(seed+70513).uniform(0,2*math.pi);co,si=math.cos(a),math.sin(a)
            return [Source(s.channel,co*s.x-si*s.y,si*s.x+co*s.y,s.radius,
                           None if s.orientation is None else (s.orientation+math.degrees(a))%360)
                    for s in make_case(seed,'cluster')]
        if mode=='edge_inset':
            rng=random.Random(seed+210319)
            result=[]
            for s in make_case(seed,'edge_outward'):
                factor=(1800-rng.uniform(20,100))/1800
                result.append(Source(s.channel,s.x*factor,s.y*factor,s.radius,s.orientation))
            return result
        return make_case(seed,mode)

    out=Path(out);cases=json.loads(Path(jobs).read_text())
    source_hash=hashes(source)
    with (out/f'{variant}.jsonl').open('x',encoding='utf-8') as stream:
        for i,(seed,mode,scale) in enumerate(cases):
            scene=case(seed,mode);sim=CorrelatedSimulator(scene,seed,scale)
            trace=out/f'trace_{variant}_{seed}_{mode}_{int(scale)}.jsonl' if out.name=='smoke' else None
            client=AuditClient(sim,log_path=trace)
            client.truth={s.channel:(s.x,s.y) for s in scene}
            if variant=='fifth':
                from guarded_route_strategy import GuardedRoutePolicy
                policy=GuardedRoutePolicy(client,station_margin=500.,skip_gain=.08)
            elif variant=='eighth':
                from recourse_strategy import RecoursePolicy
                policy=RecoursePolicy(client,scope='global')
            elif variant=='boundary':
                from boundary_rescue_strategy import BoundaryRescuePolicy
                policy=BoundaryRescuePolicy(client,rescue='cross')
            elif variant=='reliable':
                from reliable_action_strategy import ReliableActionPolicy
                policy=ReliableActionPolicy(client)
            else:
                from integrated_strategy import IntegratedPolicy
                policy=IntegratedPolicy(client,**VARIANTS[variant])
            client.policy=policy
            if hasattr(policy,'execute'):
                execute=policy.execute
                def tagged(action,execute=execute):
                    client.phase=action[3]
                    return execute(action)
                policy.execute=tagged
            try:
                result=policy.run();client.check();evaluation=sim.evaluation();evaluation.pop('sources')
                row=dict(seed=seed,mode=mode,length_scale_m=scale,variant=variant,**{**result,**evaluation})
                row.update(containment_violations=client.containment,nearby_effective_bearings=client.nearby,
                           mean_first_discovery_s=statistics.mean(client.first.values()),
                           last_first_discovery_s=max(client.first.values()),paid_parts=client.parts,
                           case_sha256=hashlib.sha256(json.dumps([vars(s) for s in scene],sort_keys=True).encode()).hexdigest(),
                           error_samples=[sim.error(p) for p in [(0,0),(100,200),(-950,1230),(1870,0)]])
                rebuilt=row['distance_m']/5+5*row['measures']+row['switches']+5*row['cleared']+3*row['clear_failures']
                assert abs(rebuilt-row['virtual_time_s'])<1e-5
                assert client.containment==client.nearby==0
                if row['completed']:
                    assert row['clear_ratio']==1 and row['discovery_certificate']
                    if len(policy.discovered)<16:assert sorted(policy.visited_ids)==list(range(21))
                if hasattr(policy,'execute'):
                    assert row['decisions']==row['schedule_replans']==row['measures']+row['cleared']+row['clear_failures']
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(variant,i+1,len(cases),mode,round(row['time_per_source_s'],2),row['completed'],flush=True)
            finally:client.close()
    assert hashes(source)==source_hash


def summarize(rows):
    groups={}
    for mode in sorted({r['mode'] for r in rows}):
        groups[mode]={}
        for variant in sorted({r['variant'] for r in rows}):
            a=[r for r in rows if r['mode']==mode and r['variant']==variant]
            if not a:continue
            values=sorted(r['time_per_source_s'] for r in a)
            info=dict(runs=len(a),complete=sum(r['completed'] for r in a),
                      median=statistics.median(values),p90=values[int(.9*(len(values)-1))],maximum=max(values),
                      le400=sum(v<=400 for v in values),fallback_runs=sum(r.get('fallback_attempts',0)>0 for r in a))
            for key in ('time_per_source_s','virtual_time_s','distance_m','measures','switches','clear_failures',
                        'fallback_attempts','stations_visited','runtime_s','mean_first_discovery_s','last_first_discovery_s'):
                info[key]=statistics.mean(r.get(key,0) for r in a)
            keys=set(k for r in a for k in r.get('integration_counts',{}))
            info['integration_totals']={k:sum(r.get('integration_counts',{}).get(k,0) for r in a) for k in keys}
            groups[mode][variant]=info
    return groups


def stage(name,variants):
    cases=scenarios(name);out=ROOT/'results'/name
    if out.exists():raise FileExistsError('Use a new stage, never overwrite records')
    out.mkdir(parents=True)
    sources={v:hashes(SOURCES.get(v,ROOT)) for v in variants}
    manifest=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),cases=cases,variants=variants,
                  source_sha256=sources,common_error_model='four_wave_smooth_spatial',max_whole_runs_this_iteration=550)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    jobs=out/'jobs.json';jobs.write_text(json.dumps(cases),encoding='utf-8')
    def invoke(v):
        with (out/f'{v}.log').open('w',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',v,'--jobs',str(jobs),'--out',str(out)],
                                cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return v,proc.returncode
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(invoke,v) for v in variants]):
            v,code=f.result();print('finished',v,'exit',code,'elapsed',round(time.perf_counter()-started,1),flush=True)
            if code:print((out/f'{v}.log').read_text(encoding='utf-8')[-6000:],flush=True)
    rows=[json.loads(line) for v in variants for line in (out/f'{v}.jsonl').read_text().splitlines()]
    assert len(rows)==len(cases)*len(variants),(len(rows),len(cases)*len(variants))
    for seed,mode,scale in cases:
        pair=[r for r in rows if (r['seed'],r['mode'],r['length_scale_m'])==(seed,mode,scale)]
        assert len({r['case_sha256'] for r in pair})==1
        assert len({tuple(r['error_samples']) for r in pair})==1
    assert all(hashes(SOURCES.get(v,ROOT))==sources[v] for v in variants)
    summary=dict(runs=len(rows),groups=summarize(rows),wall_s=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({m:{v:round(s['time_per_source_s'],3) for v,s in g.items()} for m,g in summary['groups'].items()},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['smoke','development','heldout'])
    ap.add_argument('--variants',nargs='+')
    ap.add_argument('--worker');ap.add_argument('--jobs');ap.add_argument('--out')
    args=ap.parse_args()
    if args.worker:worker(args.worker,args.jobs,args.out)
    else:stage(args.stage,args.variants)
