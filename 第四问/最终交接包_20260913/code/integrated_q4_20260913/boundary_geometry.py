import math
from geometry import dist,dot,sub,add,mul,polygon_distance,unit

def circle_arcs(poly, radius=1800.):
    """Analytic circle/half-plane crossings; return feasible angular intervals.

    Intervals are candidate generation only. The localization polygon is untouched.
    """
    if len(poly)<3:return []
    cuts=[0.,2*math.pi]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        edge=sub(b,a);normal=(edge[1],-edge[0]);length=math.hypot(*normal)
        if length<1e-12:continue
        c=dot(normal,a)/(radius*length)
        if c>=1.-1e-12:continue
        if c < -1.-1e-12:return []
        phi=math.atan2(normal[1],normal[0]);angle=math.acos(max(-1.,min(1.,c)))
        cuts += [(phi-angle)%(2*math.pi),(phi+angle)%(2*math.pi)]
    cuts=sorted(set(cuts));arcs=[]
    for a,b in zip(cuts,cuts[1:]):
        mid=(a+b)/2;p=(radius*math.cos(mid),radius*math.sin(mid))
        if polygon_distance(p,poly)<=1e-5:
            if arcs and abs(arcs[-1][1]-a)<1e-10:arcs[-1]=(arcs[-1][0],b)
            else:arcs.append((a,b))
    if len(arcs)>1 and arcs[0][0]<1e-10 and abs(arcs[-1][1]-2*math.pi)<1e-10:
        arcs=[(arcs[-1][0],arcs[0][1]+2*math.pi)]+arcs[1:-1]
    return arcs


def arc_probe_points(poly, anchor, count=3):
    arcs=circle_arcs(poly)
    if not arcs:return []
    point=lambda t:(1800*math.cos(t),1800*math.sin(t))
    a,b=min(arcs,key=lambda ab:dist(anchor,point(sum(ab)/2)))
    n=max(1,min(count,math.ceil(1800*(b-a)/32.)))
    points=[point(a+(b-a)*(i+.5)/n) for i in range(n)]
    return sorted(points,key=lambda p:dist(p,point((a+b)/2)))


