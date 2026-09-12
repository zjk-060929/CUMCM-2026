"""Search coverage, angular convergence, and correctly weighted rounded reads."""
import json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.ndimage import maximum_filter1d
from geometry import ALPHA,PRIOR_AREA,geometry_array,near_geometry
from objectives import evaluate,point_from_parameters
from benchmark import support

OUT=Path('results')

def rounded_objectives(beta,length=1000.,order=5):
    a,b=length*math.cos(beta),length*math.sin(beta)
    lo,hi=support(a,b);h=math.pi/18000
    grid=np.arange(math.floor((lo-ALPHA-h)/h),math.ceil((hi+ALPHA+h)/h)+1)*h
    # Integrate the ORIGINAL continuous-read density over each rounding bin.
    # Widened posterior area is not the probability of the rounded outcome.
    nodes,weights=np.polynomial.legendre.leggauss(order)
    masses=np.zeros(len(grid))
    for x,w in zip(nodes,weights):
        area,_,_=geometry_array(a,b,grid+x*h/2)
        masses+=w*h/2*area/(2*ALPHA*PRIOR_AREA)
    _,d,_=geometry_array(a,b,grid,ALPHA+h/2)
    true_bin_area,_,_=geometry_array(a,b,grid,h/2)
    worst=maximum_filter1d(d,size=201,mode='constant',cval=0.)
    na,nd=near_geometry(a,b)
    return dict(mean=float(np.dot(masses,d)+na*nd/PRIOR_AREA),
                conservative=float(np.dot(true_bin_area/PRIOR_AREA,worst)+na*nd/PRIOR_AREA),
                observation_mass=float(masses.sum()+na/PRIOR_AREA),
                source_mass=float(true_bin_area.sum()/PRIOR_AREA+na/PRIOR_AREA))

def main():
    start=time.perf_counter();sol=json.loads((OUT/'optima_refined.json').read_text())
    grid=[];profiles=[]
    # Full sufficient domain, both objective values at all points. Reflection
    # is exact, so the upper half is sufficient.
    for length in np.linspace(25,1000,40):
        row=[]
        for fraction in np.linspace(0,1,81):
            a,b=point_from_parameters(float(length),float(fraction))
            x=evaluate(a,b,2049);row.append(x);grid.append(x)
        profiles.append(dict(length_m=float(length),mean=min(x['mean_diameter_m'] for x in row),
                             conservative=min(x['conservative_mean_m'] for x in row)))
    coverage={'grid_points':len(grid),'length_range_m':[25,1000],
              'fractions_of_full_feasible_angle':81,'profiles':profiles,
              'grid_mean_best':min(grid,key=lambda x:x['mean_diameter_m']),
              'grid_conservative_best':min(grid,key=lambda x:x['conservative_mean_m'])}
    # Include small distances with dense angular integration and the origin.
    coverage['short_distance_checks']=[evaluate(*point_from_parameters(l,f),32769) for l in [0.,1.,5.,10.] for f in [0.,.5,1.]]
    coverage['radial_boundary_checks']=[evaluate(l*math.cos(math.radians(30.13)),l*math.sin(math.radians(30.13)),32769) for l in [950.,990.,995.,999.,1000.]]
    (OUT/'search_coverage.json').write_text(json.dumps(coverage,indent=2))
    print('full grid done',len(grid),'seconds',time.perf_counter()-start,flush=True)
    convergence=[]
    for s in sol:
        convergence.append([evaluate(s['a_m'],s['b_m'],n) for n in [2049,8193,32769,131073]])
    (OUT/'quadrature_convergence.json').write_text(json.dumps(convergence,indent=2))
    rounded=[]
    for metric in ['mean','conservative']:
        # Quantization introduces small local wiggles: scan before refining.
        betas=np.radians(np.linspace(29.5,30.7,49))
        scores=np.array([rounded_objectives(b)[metric] for b in betas]);i=int(scores.argmin())
        fit=minimize_scalar(lambda b:rounded_objectives(b)[metric],bounds=(betas[max(0,i-1)],betas[min(len(betas)-1,i+1)]),method='bounded',options={'xatol':1e-9})
        beta=float(fit.x)
        rec=dict(metric=metric,heading_deg=math.degrees(beta),length_m=1000.,a_m=1000*math.cos(beta),b_m=1000*math.sin(beta),
                 values_order5=rounded_objectives(beta),values_order9=rounded_objectives(beta,order=9),
                 scope='Local angle reoptimization near continuous optimum on L=1000; no global rounded optimum certificate.')
        rounded.append(rec);print('rounded',rec,flush=True)
    baselines=[]
    for s in sol:
        beta=math.atan2(s['b_m'],s['a_m']);baselines.append(rounded_objectives(beta,order=9))
    (OUT/'rounded_reoptimization.json').write_text(json.dumps({'optimized':rounded,'continuous_optima_rounded':baselines,'rounding_deg':.01,'posterior_half_angle_deg':1.005},indent=2))
    print('all checks done seconds',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
