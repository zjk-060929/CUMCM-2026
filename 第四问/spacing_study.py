"""Spacing comparison on seeds disjoint from the main benchmark and stress tests."""
import csv
import json
import statistics
from pathlib import Path
from client import Client
from simulator import LocalSimulator,make_case
from strategy import Policy

def main():
    rows=[]; summaries={}
    for side in (905.,920.,950.,990.):
        group=[]
        for seed in range(5000,5100):
            sim=LocalSimulator(make_case(seed),seed)
            result=Policy(Client(sim),side=side).run(); e=sim.evaluation()
            row={'seed':seed,'side_m':side,'virtual_time_s':e['virtual_time_s'],
                 'clear_ratio':e['clear_ratio'],'completed':result['completed'],'fallback_channels':result['fallback_channels']}
            rows.append(row); group.append(row)
        summaries[str(int(side))]={'cases':100,'all_cleared_runs':sum(r['clear_ratio']==1 and r['completed'] for r in group),
                                  'mean_virtual_time_s':statistics.mean(r['virtual_time_s'] for r in group),
                                  'max_virtual_time_s':max(r['virtual_time_s'] for r in group)}
    out=Path('results'); out.mkdir(exist_ok=True)
    with (out/'spacing_study.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'spacing_study_summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__': main()
