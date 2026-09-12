"""One-dimensional pushforward integrals for mean and conservative models."""
import math
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from geometry import ALPHA, PRIOR_AREA, geometry_array, near_geometry, observation_geometry, true_bearing_density
from benchmark import support, mean_quad


def point_from_parameters(length, fraction):
    # At radius L, safety requires cos(beta+alpha)>=L/2000.
    beta_max=math.acos(length/2000)-ALPHA
    beta=fraction*beta_max
    return length*math.cos(beta),length*math.sin(beta)


def safe(a,b):
    sq=a*a+b*b
    return sq<=1e6+1e-6 and sq<=2000*(a*math.cos(ALPHA)-abs(b)*math.sin(ALPHA))+1e-6


def distributions(a,b,n=4097):
    a,b=float(a),float(b)
    lo,hi=support(a,b)
    step=(hi-lo)/(n-1)
    pad=int(math.ceil(ALPHA/step))+2
    angles=lo+np.arange(-pad,n+pad)*step
    areas,dens_d,dens_true=geometry_array(a,b,angles)
    obs_density=areas/(2*ALPHA*PRIOR_AREA)
    h=int(math.floor(ALPHA/step))
    maxima=maximum_filter1d(dens_d,size=2*h+1,mode='constant',cval=0.)
    maxima=np.maximum(maxima,np.interp(angles-ALPHA,angles,dens_d,left=0,right=0))
    maxima=np.maximum(maxima,np.interp(angles+ALPHA,angles,dens_d,left=0,right=0))
    near_area,near_d=near_geometry(a,b)
    near_prob=near_area/PRIOR_AREA
    return dict(angles=angles,diameters=dens_d,observation_density=obs_density,
                true_density=dens_true,worst_diameters=maxima,step=step,
                near_probability=near_prob,near_diameter=near_d)


def evaluate(a,b,n=4097,retain=False):
    if not safe(a,b):
        raise ValueError('unsafe point')
    x=distributions(a,b,n)
    angles=x['angles'];d=x['diameters'];f=x['observation_density'];q=x['true_density'];worst=x['worst_diameters']
    np_=x['near_probability'];nd=x['near_diameter']
    result=dict(a_m=float(a),b_m=float(b),travel_m=math.hypot(a,b),
        heading_deg=math.degrees(math.atan2(b,a)),move_measure_time_s=math.hypot(a,b)/5+5,
        mean_diameter_m=float(np.trapezoid(d*f,angles)+np_*nd),
        conservative_mean_m=float(np.trapezoid(worst*q,angles)+np_*nd),
        observation_mass=float(np.trapezoid(f,angles)+np_),
        true_bearing_mass=float(np.trapezoid(q,angles)+np_),
        near_probability=np_,near_diameter_m=nd,n_true_grid=n,angle_step_deg=math.degrees(x['step']))
    if retain:
        return result,x
    return result


def weighted_stats(values,weights,extra_value=0.,extra_weight=0.):
    v=np.r_[values,extra_value];w=np.r_[weights,extra_weight]
    good=w>0
    v,w=v[good],w[good]
    raw_mass=float(w.sum());w=w/raw_mass
    order=np.argsort(v);vs,ws=v[order],w[order];cs=np.cumsum(ws)
    mean=float(np.dot(v,w));var=float(np.dot((v-mean)**2,w));sd=math.sqrt(var)
    out={'mass_before_normalization':raw_mass,'mean_m':mean,'std_m':sd,'rms_m':math.sqrt(float(np.dot(v*v,w))),
         'min_positive_weight_m':float(v.min()),'max_positive_weight_m':float(v.max())}
    for p in [.01,.05,.1,.25,.5,.75,.9,.95,.99,.999]:
        out[f'q{100*p:g}_m']=float(vs[min(np.searchsorted(cs,p),len(vs)-1)])
    for t in [10,20,30,40,50,60,80,100,120,150,200,500]:
        out[f'prob_D_le_{t}']=float(w[v<=t].sum())
    for p in [.9,.95,.99]:
        # Fractional last atom gives the exact upper-tail mean on this quadrature.
        taken=np.clip(cs-p,0,1-p)-np.clip(cs-ws-p,0,1-p)
        out[f'cvar{100*p:g}_m']=float(np.dot(vs,taken)/(1-p))
    return out


def full_statistics(a,b,n=65537):
    result,x=evaluate(a,b,n,True)
    trapezoid=np.ones(len(x['angles']))*x['step'];trapezoid[[0,-1]]*=.5
    result['ordinary_observation_statistics']=weighted_stats(x['diameters'],x['observation_density']*trapezoid,x['near_diameter'],x['near_probability'])
    result['per_source_worst_statistics']=weighted_stats(x['worst_diameters'],x['true_density']*trapezoid,x['near_diameter'],x['near_probability'])
    exact,err=mean_quad(a,b,tolerance=2e-5)
    result['adaptive_mean_m']=exact;result['adaptive_mean_error_estimate_m']=err
    return result,x


def corner_angles(a,b,lo):
    result=[]
    for r in (5.,1500.):
        for s in (-1.,1.):
            angle=math.atan2(r*math.sin(s*ALPHA)-b,r*math.cos(ALPHA)-a)
            while angle<lo-1e-10:angle+=2*math.pi
            result.append(angle)
    return result


def local_diameter_peaks(a,b,n=4097):
    lo,hi=support(a,b)
    y=np.linspace(lo-ALPHA,hi+ALPHA,n)
    _,d,_=geometry_array(a,b,y)
    peaks=[]
    for i in np.flatnonzero((d[1:-1]>=d[:-2])&(d[1:-1]>=d[2:])&(d[1:-1]>0))+1:
        if max(d[i]-d[i-1],d[i]-d[i+1])<1e-9:continue
        res=minimize_scalar(lambda z:-observation_geometry(a,b,float(z))[1],bounds=(y[i-1],y[i+1]),method='bounded',options={'xatol':1e-13})
        peaks.append((float(res.x),float(-res.fun)))
    return peaks


def conservative_quad(a,b,tolerance=.0002,peak_n=4097,return_details=False):
    """Sliding supremum from endpoint values plus interior local maxima.

    Peak discovery is numerical, checked separately by angular refinement.
    This is not a symbolic global-supremum certificate.
    """
    a,b=float(a),float(b)
    lo,hi=support(a,b);peaks=local_diameter_peaks(a,b,peak_n)
    na,nd=near_geometry(a,b)
    def f(phi):
        density=true_bearing_density(a,b,phi)
        if density<=0:return 0.
        best=max(observation_geometry(a,b,phi-ALPHA)[1],observation_geometry(a,b,phi+ALPHA)[1])
        for y,d in peaks:
            if phi-ALPHA<=y<=phi+ALPHA:best=max(best,d)
        return density*best
    points=[]
    for angle in corner_angles(a,b,lo):
        for shift in (-2*ALPHA,-ALPHA,0,ALPHA,2*ALPHA):
            if lo<angle+shift<hi:points.append(angle+shift)
    for y,d in peaks:
        for shift in (-ALPHA,0,ALPHA):
            if lo<y+shift<hi:points.append(y+shift)
    value,error=quad(f,lo,hi,points=sorted(set(points)),epsabs=tolerance,epsrel=1e-8,limit=600)
    value+=na*nd/PRIOR_AREA
    return (value,error,peaks) if return_details else (value,error)
