"""Independent high-resolution polygon geometry vs circle/line formulas."""
import json,math,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from shapely.geometry import Polygon,Point
from shapely import minimum_bounding_radius
from geometry import ALPHA,observation_geometry,near_geometry,PRIOR_AREA,mec_array
from objectives import point_from_parameters,evaluate

def main():
    t=time.perf_counter();rng=np.random.default_rng(716231)
    angular=np.linspace(-ALPHA,ALPHA,1025)
    prior=Polygon(np.vstack([1500*np.c_[np.cos(angular),np.sin(angular)],
                             5*np.c_[np.cos(angular[::-1]),np.sin(angular[::-1])]]))
    def poly_diameter(shape):
        if shape.is_empty:return 0.
        hull=shape.convex_hull
        if hull.geom_type=='Point':return 0.
        co=np.array(hull.exterior.coords if hull.geom_type=='Polygon' else hull.coords)
        return float(pdist(co).max()) if len(co)>1 else 0.
    records=[]
    for i in range(360):
        if i<80:
            a=float(rng.uniform(0,1000));b=float(rng.uniform(-1,1)*a*math.tan(ALPHA))
        else:
            a,b=point_from_parameters(float(rng.uniform(.01,1000)),float(rng.uniform(0,1)))
        phi=float(rng.uniform(-ALPHA,ALPHA));r=math.sqrt(rng.uniform(25,1500**2))
        gx,gy=r*math.cos(phi),r*math.sin(phi)
        y=math.atan2(gy-b,gx-a)+float(rng.uniform(-ALPHA,ALPHA))
        if i<30:y=math.atan2(gy-b,gx-a)+[-ALPHA,0,ALPHA][i%3]
        wedge=Polygon([(a,b),(a+6000*math.cos(y-ALPHA),b+6000*math.sin(y-ALPHA)),
                       (a+6000*math.cos(y+ALPHA),b+6000*math.sin(y+ALPHA))])
        near=Point(a,b).buffer(5,quad_segs=1024)
        shape=prior.intersection(wedge).difference(near)
        area,d=observation_geometry(a,b,y)
        expected_d=poly_diameter(shape)
        na,nd=near_geometry(a,b);ns=prior.intersection(near)
        record=dict(case=i,a=a,b=b,area_error=abs(area-shape.area),diameter_error=abs(d-expected_d),
                    mec_radius_error=abs(mec_array(float(a),float(b),np.array([y]))[0]-minimum_bounding_radius(shape)),
                    near_area_error=abs(na-ns.area),near_diameter_error=abs(nd-poly_diameter(ns)))
        records.append(record)
    summary={'case_count':len(records),'independent_reference':'Shapely, sector arcs 1024 subdivisions, near circle 4096 edges',
             'max_errors':{k:max(x[k] for x in records) for k in ['area_error','diameter_error','near_area_error','near_diameter_error','mec_radius_error']},
             'worst_diameter_case':max(records,key=lambda x:x['diameter_error']),
             'worst_near_case':max(records,key=lambda x:x['near_diameter_error'])}
    out=Path('results');out.mkdir(exist_ok=True)
    (out/'geometry_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    assert summary['max_errors']['area_error']<.001
    assert summary['max_errors']['diameter_error']<.002
    assert summary['max_errors']['near_area_error']<.001
    assert summary['max_errors']['near_diameter_error']<.002
    assert summary['max_errors']['mec_radius_error']<.002
    print('all geometry checks passed, seconds',time.perf_counter()-t,flush=True)

if __name__=='__main__':main()
