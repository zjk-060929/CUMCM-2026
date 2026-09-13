"""Regenerate publication tables from archived raw rows; no mission simulation."""
import csv
import json
from pathlib import Path
import random
import statistics as st

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'code/integrated_q4_20260913/results'
VERSIONS=['fifth','eighth','boundary','reliable','integrated']

def rows(stage,variant):
    return [json.loads(s) for s in (DATA/stage/(variant+'.jsonl')).read_text(encoding='utf-8').splitlines()]

def write(name,data):
    with (ROOT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)

def main():
    (ROOT/'tables').mkdir(exist_ok=True)
    held={v:rows('heldout',v) for v in VERSIONS}
    summary=json.loads((DATA/'heldout/summary.json').read_text(encoding='utf-8'))['groups']
    groups=[];ordinary=[];paired=[];paid=[]
    for variant,records in held.items():
        assert len(records)==69
        for r in records:
            assert r['completed'] and r['cleared']==r['source_count'] and r['clear_ratio']==1
            assert r['containment_violations']==r['nearby_effective_bearings']==0
            total=r['distance_m']/5+5*r['measures']+r['switches']+5*r['cleared']+3*r['clear_failures']
            assert abs(total-r['virtual_time_s'])<1e-5
            assert abs(total/r['source_count']-r['time_per_source_s'])<1e-6
        for mode in summary:
            a=[r for r in records if r['mode']==mode];values=sorted(r['time_per_source_s'] for r in a)
            mean=st.mean(values)
            assert abs(mean-summary[mode][variant]['time_per_source_s'])<1e-8
            row=dict(variant=variant,mode=mode,cases=len(a),layouts=len({r['seed'] for r in a}),
                     mean_seconds_per_source=mean,median=st.median(values),p90=values[int(.9*(len(values)-1))],
                     maximum=max(values),cases_at_most_400=sum(t<=400 for t in values),
                     mean_distance_m=st.mean(r['distance_m'] for r in a),mean_measures=st.mean(r['measures'] for r in a),
                     fallback_cases=sum(r['fallback_attempts']>0 for r in a),mean_program_seconds=st.mean(r['runtime_s'] for r in a))
            groups.append(row)
            if mode=='mixed':ordinary.append(row)
        a=[r for r in records if r['mode']=='mixed']
        components=dict(variant=variant,cases=len(a),layouts=len({r['seed'] for r in a}))
        for name,key,mult in [('movement','distance_m',.2),('measurement','measures',5.),('switching','switches',1.),
                              ('successful_clear','cleared',5.),('failed_clear','clear_failures',3.)]:
            components[name+'_seconds_per_source']=st.mean(mult*r[key]/r['source_count'] for r in a)
        paid.append(components)
    keyed={v:{(r['seed'],r['mode'],r['length_scale_m']):r for r in rs} for v,rs in held.items()}
    for key in keyed['integrated']:
        pairs=[keyed[v][key] for v in VERSIONS]
        assert len({r['case_sha256'] for r in pairs})==1
        assert len({tuple(r['error_samples']) for r in pairs})==1
    for variant in VERSIONS[:-1]:
        a={k:r for k,r in keyed[variant].items() if r['mode']=='mixed'}
        diff={k:r['time_per_source_s']-keyed['integrated'][k]['time_per_source_s'] for k,r in a.items()}
        cluster=[st.mean(v for k,v in diff.items() if k[0]==seed) for seed in sorted({k[0] for k in a})]
        rng=random.Random(612031);boot=sorted(st.mean(rng.choices(cluster,k=len(cluster))) for _ in range(5000))
        paired.append(dict(comparator=variant,cases=len(a),layouts=len(cluster),saved_seconds_per_source=st.mean(diff.values()),
                           ci95_low=boot[125],ci95_high=boot[4874],faster=sum(x>1e-6 for x in diff.values()),
                           slower=sum(x< -1e-6 for x in diff.values()),bootstrap_draws=5000))
    development=[]
    for v in ['integrated','expanded','free','no_sensing']:
        a=[r for r in rows('development',v) if r['mode']=='mixed']
        development.append(dict(variant=v,cases=len(a),layouts=len({r['seed'] for r in a}),
                                mean_seconds_per_source=st.mean(r['time_per_source_s'] for r in a),
                                mean_distance_m=st.mean(r['distance_m'] for r in a),
                                mean_measures=st.mean(r['measures'] for r in a),
                                mean_program_seconds=st.mean(r['runtime_s'] for r in a)))
    write('mean_by_scene.csv',groups);write('ordinary_distribution.csv',ordinary)
    write('paired_improvement.csv',paired);write('paid_time_components.csv',paid)
    write('development_ablation.csv',development)
    print('PASS: 345 archived validation rows; wrote five tables. No new simulations.')

if __name__=='__main__':main()
