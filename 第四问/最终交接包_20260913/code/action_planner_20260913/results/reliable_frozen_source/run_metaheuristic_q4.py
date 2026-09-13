"""Frozen heuristic policy. A new offline simulation is the default."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from client import Client, HttpTransport
from simulator import LocalSimulator, make_case
from metaheuristic_strategy import MetaheuristicPolicy, METAHEURISTIC_CONFIG


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--mode',choices=['mixed','edge_outward','minimum_range','cluster','near_origin'],default='mixed')
    ap.add_argument('--error',choices=['fixed_hash','plus_one','minus_one','checker','smooth','zero'],default='fixed_hash')
    ap.add_argument('--http',action='store_true')
    ap.add_argument('--url',default='http://127.0.0.1:2027')
    ap.add_argument('--robot-id',default='SELF-Q4')
    ap.add_argument('--output')
    args = ap.parse_args()
    out = Path(args.output or 'results/metaheuristic_demo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if out.exists() and any(out.iterdir()):
        ap.error('Use a new, empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    transport = HttpTransport(args.url) if args.http else LocalSimulator(make_case(args.seed,args.mode),args.seed,args.error,robot_id=args.robot_id)
    client = Client(transport,args.robot_id,out/'actions.jsonl')
    try:
        result = MetaheuristicPolicy(client,**METAHEURISTIC_CONFIG).run()
        if not args.http:
            evaluation = transport.evaluation()
            (out/'truth_after_exit.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
            result['evaluation'] = {k:v for k,v in evaluation.items() if k!='sources'}
        result['test_kind'] = 'HTTP_ENDPOINT_UNVERIFIED' if args.http else 'SELF_BUILT_OFFLINE'
        (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        print(f'Output: {out.resolve()}')
    except Exception as exc:
        failure = dict(completed=False,stop_reason=repr(exc),last_confirmed_position=client.position,
                       last_confirmed_virtual_time_s=client.virtual,entered=client.entered,exited=client.exited)
        client.log('fatal_error',**failure)
        (out/'failure_summary.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        raise
    finally:
        client.close()


if __name__ == '__main__':
    main()
