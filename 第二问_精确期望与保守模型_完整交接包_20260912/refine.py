import json,time,math
from pathlib import Path
from scipy.optimize import minimize_scalar,minimize
from objectives import mean_quad,conservative_quad,point_from_parameters,evaluate

def main():
    out=Path('results');initial=json.loads((out/'optima_initial.json').read_text())
    results=[];t=time.perf_counter()
    for metric in ['mean','conservative']:
        evaluator=mean_quad if metric=='mean' else conservative_quad
        def fun(beta):
            a,b=1000*math.cos(beta),1000*math.sin(beta)
            return evaluator(a,b,tolerance=.00002)[0]
        opt=minimize_scalar(fun,bounds=(math.radians(25),math.radians(35)),method='bounded',options={'xatol':1e-8})
        a,b=1000*math.cos(opt.x),1000*math.sin(opt.x)
        vals=[]
        for n in [1025,4097,16385]:
            val,err,peaks=conservative_quad(a,b,tolerance=1e-6,peak_n=n,return_details=True)
            vals.append({'peak_grid':n,'value_m':val,'quad_error_m':err,'peaks':peaks})
        rec={'optimized_metric':metric,**evaluate(a,b,65537),'mean_adaptive':mean_quad(a,b,1e-6),'conservative_adaptive':vals}
        results.append(rec);print(json.dumps(rec),flush=True)
        (out/'optima_refined.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('seconds',time.perf_counter()-t,flush=True)

if __name__=='__main__':main()
