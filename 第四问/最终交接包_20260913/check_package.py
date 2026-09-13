"""Verify delivery bytes and the five archived algorithm snapshots. Read-only."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT.resolve()),name
        assert path.is_file(), 'Missing: '+name
        assert sha(path)==digest, 'Changed: '+name
    paths={'integrated':'integrated_q4_20260913','fifth':'results/recourse_iteration8_20260913/frozen_source',
           'eighth':'results/recourse_iteration8_20260913/frozen_source',
           'boundary':'results/boundary_rescue_20260913/heldout/source_snapshot',
           'reliable':'action_planner_20260913/results/reliable_frozen_source'}
    old=json.loads((ROOT/'code/integrated_q4_20260913/results/heldout/manifest.json').read_text(encoding='utf-8'))
    for variant,rel in paths.items():
        actual={p.name:sha(p) for p in (ROOT/'code'/rel).glob('*.py')}
        assert actual==old['source_sha256'][variant],variant+' snapshot mismatch'
    print(f'PASS: {len(manifest["files"])} delivery files and all five frozen policy snapshots.')

if __name__=='__main__':main()
