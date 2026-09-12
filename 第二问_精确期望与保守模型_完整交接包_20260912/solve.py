import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from objectives import point_from_parameters,evaluate

OUT=Path(__file__).parent/'results'

def main():
    OUT.mkdir(exist_ok=True)
    t=time.perf_counter();history=[];solutions=[]
    for objective in ['mean_diameter_m','conservative_mean_m']:
        def loss(z,n=2049):
            a,b=point_from_parameters(float(z[0]),float(z[1]))
            rec=evaluate(a,b,n)
            history.append({'objective':objective,'length':float(z[0]),'angle_fraction':float(z[1]),**rec})
            return rec[objective]
        # Full safe half-domain; the negative-b half is exactly symmetric.
        res=differential_evolution(loss,[(.001,1000.),(0.,1.)],seed=20260912,popsize=10,maxiter=60,tol=2e-5,polish=False)
        print(objective,'global_screen',res.x,res.fun,flush=True)
        candidates=[res.x]
        def boundary_loss(frac):return loss([1000.,frac],8193)
        bound=minimize_scalar(boundary_loss,bounds=(0.01,1.),method='bounded',options={'xatol':1e-7})
        candidates.append(np.array([1000.,bound.x]))
        for start in [res.x,np.array([1000.,bound.x])]:
            refined=minimize(lambda z:loss(z,8193),start,method='Nelder-Mead',bounds=[(.001,1000.),(0.,1.)],
                             options={'xatol':2e-4,'fatol':1e-6,'maxiter':160})
            candidates.append(refined.x)
        scored=[]
        for z in candidates:
            a,b=point_from_parameters(*z)
            rec=evaluate(a,b,32769);scored.append((rec[objective],z,rec))
        value,z,rec=min(scored,key=lambda x:x[0])
        rec.update(objective=objective,length_parameter=float(z[0]),angle_fraction=float(z[1]))
        solutions.append(rec)
        print('solution',json.dumps(rec),flush=True)
        (OUT/'optima_initial.json').write_text(json.dumps(solutions,indent=2),encoding='utf-8')
    with (OUT/'optimization_history.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    print('evaluations',len(history),'seconds',time.perf_counter()-t,flush=True)

if __name__=='__main__':main()
