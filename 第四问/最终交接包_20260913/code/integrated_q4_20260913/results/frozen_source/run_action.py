"""Offline entry for the action-by-action Q4 planner; no HTTP requests."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from benchmark_action import run_one,VARIANTS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--scale',type=float,default=300.)
    ap.add_argument('--mode',choices=['mixed','edge_outward','cluster','minimum_range'],default='mixed')
    ap.add_argument('--variant',choices=VARIANTS)
    ap.add_argument('--output')
    args=ap.parse_args();root=Path(__file__).resolve().parent
    frozen=json.loads((root/'results/frozen_config.json').read_text(encoding='utf-8'))
    for name,digest in frozen['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Source changed since validation: '+name)
    variant=args.variant or frozen['selected_variant']
    out=Path(args.output) if args.output else root/'results'/('entry_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Use a fresh output directory')
    result=run_one(args.seed,args.mode,args.scale,variant,out/'actions.jsonl')
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
