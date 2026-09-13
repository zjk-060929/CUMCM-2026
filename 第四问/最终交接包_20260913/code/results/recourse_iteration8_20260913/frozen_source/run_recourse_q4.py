"""Offline entry for the frozen iteration-8 policy."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from benchmark_efficiency_q4 import make_benchmark_case
from client import Client
from correlated_simulator import CorrelatedSimulator
from recourse_strategy import RecoursePolicy,VARIANTS


SELECTED = 'insert'  # Chosen on development and frozen before heldout.

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--mode',choices=['mixed','edge_outward','rotated_cluster','minimum_range'],default='mixed')
    ap.add_argument('--length-scale',type=float,default=300.)
    ap.add_argument('--output')
    a=ap.parse_args()
    out=Path(a.output or 'results/recourse_demo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if out.exists() and any(out.iterdir()):ap.error('Use a fresh empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    sim=CorrelatedSimulator(make_benchmark_case(a.seed,a.mode),a.seed,a.length_scale)
    client=Client(sim,log_path=out/'actions.jsonl')
    try:
        result=RecoursePolicy(client,**VARIANTS[SELECTED]).run()
        evaluation=sim.evaluation()
        (out/'truth_after_exit.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
        result.update(selected_variant=SELECTED,test_kind='SELF_BUILT_CORRELATED_OFFLINE',
                      synthetic_length_scale_m=a.length_scale,
                      evaluation={k:v for k,v in evaluation.items() if k!='sources'})
        (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2));print(f'Output: {out.resolve()}')
    finally:client.close()


if __name__=='__main__':main()
