import csv,json,math,time
from pathlib import Path
import numpy as np
from geometry import mec_array,sample_diameters,ALPHA
from objectives import full_statistics,weighted_stats,conservative_quad

OUT=Path('results')

def policies():
    sol=json.loads((OUT/'optima_refined.json').read_text())
    a,b=9000/11,2000*math.sqrt(10)/11
    return [('mean_optimum',sol[0]['a_m'],sol[0]['b_m']),
            ('conservative_optimum',sol[1]['a_m'],sol[1]['b_m']),
            ('original_35deg_1000m',a,b),('45deg_1000m',1000/math.sqrt(2),1000/math.sqrt(2)),
            ('59deg_1000m',1000*math.cos(math.radians(59)),1000*math.sin(math.radians(59))),
            ('original_35deg_750m',.75*a,.75*b),
            ('mean_angle_750m',.75*sol[0]['a_m'],.75*sol[0]['b_m'])]

def sample_stats(v):
    return {'count':len(v),'mean':float(v.mean()),'std':float(v.std(ddof=1)),
            'mean_se':float(v.std(ddof=1)/math.sqrt(len(v))),
            'quantiles':{str(q):float(np.quantile(v,q)) for q in [.01,.05,.25,.5,.75,.9,.95,.99,.999]},
            'min':float(v.min()),'max':float(v.max())}

def main():
    t=time.perf_counter();out=[];compact=[]
    for name,a,b in policies():
        rec,x=full_statistics(a,b,131073)
        rec['policy']=name
        rec['conservative_adaptive']=conservative_quad(a,b,1e-5,8193)
        w=x['observation_density']*x['step'];w[[0,-1]]*=.5
        radii=mec_array(a,b,x['angles'])
        if np.any((radii>1e6)&(w>1e-12)):raise RuntimeError('MEC failure')
        rec['mec_radius_statistics']={k.replace('prob_D_le_','prob_radius_le_'):v for k,v in weighted_stats(radii,w).items()}
        rec['guaranteed_clear_probability']=float(w[radii<=20].sum()+x['near_probability'])
        areas=x['observation_density']*(2*ALPHA)* (ALPHA*(1500**2-5**2))
        rec['posterior_area_statistics']={k.replace('_m','_m2').replace('prob_D_le_','prob_area_m2_le_'):v for k,v in weighted_stats(areas,w).items()}
        out.append(rec)
        flat={k:rec[k] for k in ['policy','a_m','b_m','travel_m','heading_deg','move_measure_time_s','adaptive_mean_m','guaranteed_clear_probability','observation_mass','true_bearing_mass']}
        flat['conservative_mean_m']=rec['conservative_adaptive'][0]
        for k,v in rec['ordinary_observation_statistics'].items():flat['D_'+k]=v
        for k,v in rec['per_source_worst_statistics'].items():flat['worst_'+k]=v
        compact.append(flat)
        np.savez_compressed(OUT/(name+'_angular_distribution.npz'),**x,mec_radii=radii)
        (OUT/'policy_statistics.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
        print(name,'mean',rec['adaptive_mean_m'],'robust',rec['conservative_adaptive'][0],'clear_prob',rec['guaranteed_clear_probability'],flush=True)
    with (OUT/'policy_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    # Independent paired Monte Carlo: same G and same latent standardized error
    # across policies. This coupling controls variance; it is not a physical
    # assertion that different locations have identical errors.
    rng=np.random.default_rng(2026091201);count=200000
    rr=np.sqrt(rng.uniform(25,1500**2,count));theta=rng.uniform(-ALPHA,ALPHA,count)
    sx,sy=rr*np.cos(theta),rr*np.sin(theta);err=rng.uniform(-ALPHA,ALPHA,count)
    raw={'source_x':sx,'source_y':sy,'error':err};mc=[];base=None
    for name,a,b in policies():
        ds,inc,radii,reads=sample_diameters(float(a),float(b),sx,sy,err)
        if not inc.all():raise RuntimeError('Continuous containment failure')
        if base is None:base=ds.copy()
        delta=ds-base
        rec={'policy':name,'diameter':sample_stats(ds),'clear_probability':float(np.mean(radii<=20)),
             'delta_vs_mean_optimum':sample_stats(delta),'fraction_smaller_than_mean_optimum':float(np.mean(delta<-1e-8)),
             'all_sources_contained':bool(inc.all())}
        mc.append(rec);raw[name+'_diameter']=ds;raw[name+'_mec_radius']=radii
        print('MC',name,rec['diameter']['mean'],'SE',rec['diameter']['mean_se'],flush=True)
    np.savez_compressed(OUT/'paired_200000_scenes.npz',**raw)
    (OUT/'paired_monte_carlo.json').write_text(json.dumps({'seed':2026091201,'count':count,'coupling':'Common G and common error value for variance-controlled policy differences; marginal models unchanged.','policies':mc},indent=2),encoding='utf-8')
    quant=[]
    for name,a,b in policies()[:3]:
        raw_d,raw_inc,raw_r,reads=sample_diameters(float(a),float(b),sx,sy,err,True,False)
        expanded_d,expanded_inc,expanded_r,_=sample_diameters(float(a),float(b),sx,sy,err,True,True)
        quant.append({'policy':name,'rounded_with_1deg':{'containment_rate':float(raw_inc.mean()),'excluded_sources':int((~raw_inc).sum()),'diameter':sample_stats(raw_d)},
                      'rounded_with_1_005deg':{'containment_rate':float(expanded_inc.mean()),'excluded_sources':int((~expanded_inc).sum()),'diameter':sample_stats(expanded_d),'clear_probability':float(np.mean(expanded_r<=20))}})
        print('rounding',name,int((~raw_inc).sum()),int((~expanded_inc).sum()),flush=True)
    (OUT/'rounding_validation.json').write_text(json.dumps({'count':count,'mechanism':'Second true bearing + uniform error, then nearest 0.01 degree. First prior retained as agreed; widened second wedge is the feasible support of the rounded reading.','policies':quant},indent=2),encoding='utf-8')
    print('seconds',time.perf_counter()-t,flush=True)

if __name__=='__main__':main()
