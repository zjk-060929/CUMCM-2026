"""Frozen three-cost predictor; default transport is an offline smooth field."""
import argparse,json
from pathlib import Path
from datetime import datetime
from client import HttpTransport
from simulator import make_case
from spatial_error_q4 import SpatialErrorSimulator
from predictive_strategy_q4 import PredictiveClient,PredictivePolicy,PREDICTIVE_CONFIG

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--scale',type=float,default=300.)
    ap.add_argument('--mode',choices=['mixed','edge_outward','cluster','minimum_range'],default='mixed')
    ap.add_argument('--http',action='store_true');ap.add_argument('--url',default='http://127.0.0.1:2027')
    ap.add_argument('--robot-id',default='SELF-Q4');ap.add_argument('--output')
    args=ap.parse_args()
    out=Path(args.output or 'results/predictive_demo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if out.exists() and any(out.iterdir()):ap.error('Use a new empty directory')
    out.mkdir(parents=True,exist_ok=True)
    transport=HttpTransport(args.url) if args.http else SpatialErrorSimulator(make_case(args.seed,args.mode),args.seed,args.scale,args.robot_id)
    client=PredictiveClient(transport,args.robot_id,out/'actions.jsonl')
    try:
        result=PredictivePolicy(client,**PREDICTIVE_CONFIG).run()
        if not args.http:
            evaluation=transport.evaluation()
            (out/'truth_after_exit.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
            result['evaluation']={k:v for k,v in evaluation.items() if k!='sources'}
        result['test_kind']='HTTP_ENDPOINT_UNVERIFIED' if args.http else 'SELF_BUILT_SMOOTH_FIELD'
        (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
    except Exception as exc:
        failure=dict(completed=False,stop_reason=repr(exc),last_confirmed_position=client.position,
                     last_confirmed_virtual_time_s=client.virtual,entered=client.entered,exited=client.exited)
        client.log('fatal_error',**failure)
        (out/'failure_summary.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        raise
    finally:client.close()

if __name__=='__main__':main()
