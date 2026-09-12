"""Default: one fully offline Q4 case. --http explicitly selects a local HTTP service."""
import argparse
import json
from pathlib import Path
from client import Client,HttpTransport
from simulator import LocalSimulator,make_case
from strategy import Policy
from geometry import DEFAULT_SIDE

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--grid',choices=['triangle','square'],default='triangle')
    ap.add_argument('--side',type=float,default=DEFAULT_SIDE)
    ap.add_argument('--no-active',action='store_true')
    ap.add_argument('--http',action='store_true')
    ap.add_argument('--url',default='http://127.0.0.1:2027')
    ap.add_argument('--robot-id',default='SELF-Q4')
    ap.add_argument('--output',default='results/demo')
    a=ap.parse_args()
    if not 100<=a.side<1000: ap.error('triangle side must be in [100,1000) metres')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    transport=HttpTransport(a.url) if a.http else LocalSimulator(make_case(a.seed),a.seed,robot_id=a.robot_id)
    client=Client(transport,a.robot_id,out/'actions.jsonl')
    try:
        result=Policy(client,a.grid,not a.no_active,a.side).run()
        if not a.http:
            evaluation=transport.evaluation()
            (out/'truth_after_exit.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2))
            result['evaluation']={k:v for k,v in evaluation.items() if k!='sources'}
        result['test_kind']='HTTP_ENDPOINT_UNVERIFIED' if a.http else 'SELF_BUILT_OFFLINE'
        (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except Exception as e:
        client.log('fatal_error',error=repr(e))
        raise
    finally: client.close()

if __name__=='__main__': main()
