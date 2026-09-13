"""Offline reproduction of the corrected opportunity-sensing planner."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import statistics
from benchmark_action import summary
from simulator import make_case
from spatial_error_q4 import SpatialErrorSimulator
from predictive_strategy_q4 import PredictiveClient,PredictivePolicy,PREDICTIVE_CONFIG
from reliable_action_strategy import ReliableActionPolicy


def run_one(seed,mode,scale,variant='reliable',log=None):
    sim=SpatialErrorSimulator(make_case(seed,mode),seed,scale)
    client=PredictiveClient(sim,log_path=log)
    policy=PredictivePolicy(client,**PREDICTIVE_CONFIG) if variant=='previous' else ReliableActionPolicy(client)
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--scale',type=float,default=300.)
    ap.add_argument('--mode',choices=['mixed','edge_outward','cluster','minimum_range'],default='mixed')
    ap.add_argument('--output');args=ap.parse_args()
    root=Path(__file__).resolve().parent
    frozen=json.loads((root/'results/reliable_frozen_config.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in frozen['source_sha256'].items())
    out=Path(args.output) if args.output else root/'results'/('reliable_entry_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Use a fresh output directory')
    row=run_one(args.seed,args.mode,args.scale,log=out/'actions.jsonl')
    (out/'summary.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    print(json.dumps({k:row[k] for k in ('completed','virtual_time_s','time_per_source_s','distance_m','measures','clear_failures','fallback_attempts','runtime_s','action_kinds')},indent=2))


if __name__=='__main__':main()
