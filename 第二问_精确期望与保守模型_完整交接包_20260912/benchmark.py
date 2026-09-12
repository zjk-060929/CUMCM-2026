import json, math, time
import numpy as np
from scipy.integrate import quad
from geometry import ALPHA, PRIOR_AREA, observation_geometry, near_geometry, geometry_array

def support(a,b):
    if math.hypot(a,b)<10:
        # Near the inner circular boundary, angular extrema can be tangencies.
        return -math.pi,math.pi
    if a>=5*math.cos(ALPHA) and abs(b)<=a*math.tan(ALPHA):
        return -math.pi,math.pi
    ang=np.sort(np.array([math.atan2(r*math.sin(s*ALPHA)-b,r*math.cos(ALPHA)-a)%(2*math.pi)
                          for r in (5.,1500.) for s in (-1,1)]))
    gaps=np.diff(np.r_[ang,ang[0]+2*math.pi])
    j=int(np.argmax(gaps))
    start=ang[(j+1)%4]
    return start,start+2*math.pi-gaps[j]

def mean_quad(a,b,tolerance=.002):
    lo,hi=support(a,b)
    na,nd=near_geometry(a,b)
    def f(y):
        area,d=observation_geometry(a,b,y)
        return area*d/(2*ALPHA*PRIOR_AREA)
    points=[lo,hi]
    for r in (5.,1500.):
        for s in (-1,1):
            phi=math.atan2(r*math.sin(s*ALPHA)-b,r*math.cos(ALPHA)-a)
            while phi<lo-1e-10:phi+=2*math.pi
            for e in (-ALPHA,0,ALPHA):
                if lo-ALPHA<phi+e<hi+ALPHA:points.append(phi+e)
    val,err=quad(f,lo-ALPHA,hi+ALPHA,points=sorted(set(points)),epsabs=tolerance,epsrel=1e-8,limit=400)
    return val+na*nd/PRIOR_AREA,err

if __name__=='__main__':
    a,b=9000/11,2000*math.sqrt(10)/11
    t=time.perf_counter()
    print('compile/sample',observation_geometry(a,b,math.atan2(-b,1500-a)),flush=True)
    print('compile_s',time.perf_counter()-t,flush=True)
    t=time.perf_counter()
    lo,hi=support(a,b)
    ang=np.linspace(lo-ALPHA,hi+ALPHA,10001)
    ar,d,q=geometry_array(a,b,ang)
    print('first_vector_s',time.perf_counter()-t,flush=True)
    t=time.perf_counter();geometry_array(a,b,ang)
    print('warm_10001_s',time.perf_counter()-t,flush=True)
    print('prob_mass',np.trapezoid(ar/(2*ALPHA*PRIOR_AREA),ang),'mean_grid',np.trapezoid(ar*d/(2*ALPHA*PRIOR_AREA),ang),flush=True)
    t=time.perf_counter();print('mean_quad',mean_quad(a,b),'seconds',time.perf_counter()-t,flush=True)
    for p in [(0,0),(500,0),(1000,0),(500,500),(800,500)]:
        print('near',p,near_geometry(*p),'mean',mean_quad(*p),flush=True)
