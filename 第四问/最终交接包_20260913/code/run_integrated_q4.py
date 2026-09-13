"""Run the frozen integration candidate in the common offline simulator."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--scale',type=float,default=300.)
    ap.add_argument('--mode',choices=['mixed','edge_outward','rotated_cluster','minimum_range','edge_inset'],default='mixed')
    ap.add_argument('--variant',choices=['integrated','expanded','free','no_sensing'],help='Override frozen default for an explicit ablation')
    ap.add_argument('--output')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent/'integrated_q4_20260913'
    config=json.loads((root/'results/frozen_config.json').read_text(encoding='utf-8'))
    for name,digest in config['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen integration source changed: '+name)
    variant=args.variant or config['selected_variant']
    out=Path(args.output) if args.output else root/'results'/('entry_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    jobs=out/'jobs.json';jobs.write_text(json.dumps([(args.seed,args.mode,args.scale)]),encoding='utf-8')
    subprocess.run([sys.executable,'-B',str(root/'experiment.py'),'--worker',variant,'--jobs',str(jobs),'--out',str(out)],
                   cwd=str(root),check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row=json.loads((out/f'{variant}.jsonl').read_text(encoding='utf-8'))
    (out/'summary.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:row[k] for k in ['completed','virtual_time_s','time_per_source_s','distance_m','measures',
                                       'clear_failures','fallback_attempts','integration_counts']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
