"""Exactly 96 additional runs under two smooth spatial-bias assumptions."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

from benchmark_efficiency_q4 import ProfiledClient, make_benchmark_case
from correlated_simulator import CorrelatedSimulator
from correlation_aware_strategy import CorrelationAwarePolicy, CORRELATION_CONFIG
from efficient_strategy import EfficientPolicy, EFFICIENT_CONFIG
from geometry import polygon_distance, dist
from planned_strategy import PlannedPolicy, PLANNED_CONFIG


OUT = Path('results/spatial_correlation_correction_20260913')
VARIANTS = ('previous', 'endpoint', 'correlation_guard')
SCENES = [(s, 'mixed') for s in range(241000, 241008)]
SCENES += [(s, 'edge_outward') for s in range(242000, 242004)]
SCENES += [(s, 'rotated_cluster') for s in range(243000, 243004)]
SCALES = (100., 300.)


class AuditClient(ProfiledClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_positions = {}
        self.nearby_bearings = 0
        self.snapshots = []

    def action(self, path, position=None, channel=None):
        if self.policy is not None:
            # Capture states before each next action and verify only after exit.
            self.snapshots.extend((ch, list(t.polygon)) for ch, t in self.policy.tracks.items())
        reply = super().action(path, position, channel)
        if path == '/measure' and reply['measure_result'] == 'direction':
            past = self.positive_positions.setdefault(channel, [])
            if any(dist(position, p) <= 30. for p in past):
                self.nearby_bearings += 1
            past.append(position)
        return reply


def run_one(seed, mode, scale, variant):
    sim = CorrelatedSimulator(make_benchmark_case(seed, mode), seed, scale)
    client = AuditClient(sim)
    if variant == 'previous':
        policy = PlannedPolicy(client, **PLANNED_CONFIG)
    elif variant == 'endpoint':
        policy = EfficientPolicy(client, **EFFICIENT_CONFIG)
    else:
        policy = CorrelationAwarePolicy(client, **CORRELATION_CONFIG)
    client.policy = policy
    try:
        result = policy.run()
        evaluation = sim.evaluation()
        truth = {s['channel']:(s['x'], s['y']) for s in evaluation.pop('sources')}
        snapshots = client.snapshots+[(ch, t.polygon) for ch, t in policy.tracks.items()]
        violations = sum(polygon_distance(truth[ch], poly) > 1e-4 for ch, poly in snapshots)
        assert violations == 0
        if len(policy.discovered) < 16:
            assert sorted(policy.visited_ids) == list(range(21))
        profile = {k:dict(v) for k, v in client.profile.items()}
        assert abs(sum(p['total_s'] for p in profile.values())-evaluation['virtual_time_s']) < 1e-5
        if variant == 'correlation_guard':
            assert client.nearby_bearings == 0
        return dict(seed=seed, mode=mode, length_scale_m=scale, variant=variant,
                    **{**result, **evaluation}, action_profile=profile,
                    nearby_bearings_le30m=client.nearby_bearings,
                    truth_containment_violations=violations)
    finally:
        client.close()


def describe(rows):
    keys = ['time_per_source_s', 'virtual_time_s', 'distance_m', 'measures', 'clear_failures',
            'stations_visited', 'nearby_bearings_le30m', 'correlated_measure_skips']
    result = {k:statistics.mean(r.get(k, 0) for r in rows) for k in keys}
    result.update(runs=len(rows), complete=sum(r['completed'] for r in rows),
                  sources=sum(r['source_count'] for r in rows),
                  le400=sum(r['time_per_source_s'] <= 400 for r in rows))
    return result


def compare(rows, before, after):
    a = {(r['seed'],r['length_scale_m']):r for r in rows if r['variant'] == before}
    b = {(r['seed'],r['length_scale_m']):r for r in rows if r['variant'] == after}
    differences = [a[k]['time_per_source_s']-b[k]['time_per_source_s'] for k in a]
    mean = statistics.mean(differences)
    margin = 1.96*statistics.stdev(differences)/len(differences)**.5
    return dict(mean_saved_per_source_s=mean, normal_approx_95ci=[mean-margin, mean+margin],
                faster=sum(d>1e-6 for d in differences), slower=sum(d < -1e-6 for d in differences),
                tied=sum(abs(d)<=1e-6 for d in differences))


def main():
    if OUT.exists():
        raise SystemExit('Output already exists; no accidental reruns')
    OUT.mkdir(parents=True)
    names = ['correlated_simulator.py', 'correlation_aware_strategy.py', 'benchmark_correlation_q4.py',
             'efficient_strategy.py', 'planned_strategy.py', 'refined_strategy.py',
             'optimized_strategy.py', 'refined_geometry.py', 'optimized_geometry.py',
             'strategy.py', 'geometry.py', 'simulator.py', 'client.py']
    frozen = dict(whole_policy_runs=96, scenes=SCENES, scales=SCALES, variants=VARIANTS,
                  correction_config=CORRELATION_CONFIG,
                  source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names},
                  assumptions='Synthetic smooth spatial bias, not calibrated to the official simulator')
    (OUT/'frozen_config.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    rows, started = [], time.perf_counter()
    with (OUT/'runs.jsonl').open('w', encoding='utf-8') as stream:
        for variant in VARIANTS:
            for scale in SCALES:
                for seed, mode in SCENES:
                    row = run_one(seed, mode, scale, variant)
                    rows.append(row)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                print(f'{variant}, scale={scale}: {len(rows)}/96 complete', flush=True)
    groups = {}
    for mode in ('mixed', 'edge_outward', 'rotated_cluster'):
        for scale in (None,)+SCALES:
            selected = [r for r in rows if r['mode'] == mode and (scale is None or r['length_scale_m'] == scale)]
            key = mode+'/'+('both_scales' if scale is None else str(scale))
            groups[key] = dict(summary={v:describe([r for r in selected if r['variant'] == v]) for v in VARIANTS},
                               guard_vs_previous=compare(selected, 'previous', 'correlation_guard'),
                               guard_vs_endpoint=compare(selected, 'endpoint', 'correlation_guard'))
    report = dict(test_kind='SELF_BUILT_SPATIALLY_CORRELATED_SENSITIVITY', new_runs=len(rows),
                  complete=sum(r['completed'] for r in rows), groups=groups,
                  source_sha256=frozen['source_sha256'], wall_runtime_s=time.perf_counter()-started,
                  note='Two fields on shared source scenes are sensitivity cases, not independent real-world samples.')
    (OUT/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'runs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k:v for k,v in groups.items() if k.endswith('both_scales')}, indent=2))


if __name__ == '__main__':
    main()
