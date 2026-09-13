from pathlib import Path
import json,math,sys
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'support'))
from geometry import observation_geometry,ALPHA

def intersections(p,reading):
    rows=[]
    for i,alpha in enumerate([-ALPHA,ALPHA],1):
        for j,beta in enumerate([reading-ALPHA,reading+ALPHA],1):
            den=math.sin(alpha-beta)
            if abs(den)<1e-13:continue
            t=(p[1]*math.cos(beta)-p[0]*math.sin(beta))/den
            s=(p[1]*math.cos(alpha)-p[0]*math.sin(alpha))/den
            rows.append(dict(label=f'P{i}{j}',x=t*math.cos(alpha),y=t*math.sin(alpha),t=t,s=s))
    return rows

def main():
    p=np.array([864.8772502049685,501.9834081699237]);out=[]
    for r,theta,error in [(1400.,.3,.0),(1495.,.7,.5),(100.,0.,0.)]:
        target=r*np.array([math.cos(math.radians(theta)),math.sin(math.radians(theta))])
        phi=math.atan2(target[1]-p[1],target[0]-p[0]);reading=phi+math.radians(error)
        pts=intersections(p,reading)
        xy=np.array([[v['x'],v['y']] for v in pts]);d4=float(np.max(np.linalg.norm(xy[:,None]-xy[None,:],axis=2)))
        diag=max(np.linalg.norm(xy[0]-xy[3]),np.linalg.norm(xy[1]-xy[2]))
        area,diameter=observation_geometry(float(p[0]),float(p[1]),reading)
        out.append(dict(source_radius_m=r,source_angle_deg=theta,second_error_deg=error,point=p.tolist(),target=target.tolist(),true_bearing_rad=phi,reading_rad=reading,
                        intersections=pts,max_four_points_m=d4,two_diagonals_m=float(diag),exact_clipped_diameter_m=diameter,exact_area_m2=area))
    (BASE/'geometry_examples.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    from shapely.geometry import Polygon,Point
    angles=np.linspace(-ALPHA,ALPHA,2049)
    prior=Polygon(np.vstack([1500*np.c_[np.cos(angles),np.sin(angles)],5*np.c_[np.cos(angles[::-1]),np.sin(angles[::-1])]]))
    checks=[]
    for rec in out:
        y=rec['reading_rad'];wedge=Polygon([p,p+4000*np.array([math.cos(y-ALPHA),math.sin(y-ALPHA)]),p+4000*np.array([math.cos(y+ALPHA),math.sin(y+ALPHA)])])
        shape=prior.intersection(wedge).difference(Point(*p).buffer(5))
        pts=np.array(shape.convex_hull.exterior.coords)
        ref=float(np.max(np.linalg.norm(pts[:,None]-pts[None,:],axis=2)))
        error=abs(ref-rec['exact_clipped_diameter_m']);assert error<1e-4
        checks.append(dict(source_radius_m=rec['source_radius_m'],reference_diameter_m=ref,absolute_error_m=error))
    (BASE/'independent_geometry_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    for rec in out:
        print('r',rec['source_radius_m'],'four-point',rec['max_four_points_m'],'exact',rec['exact_clipped_diameter_m'],'max vertex r',max(abs(v['t']) for v in rec['intersections']))

if __name__=='__main__':main()
