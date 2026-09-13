"""Exact circle/line geometry for the agreed Q2 prior (floating point).

No polygonal circle approximation and no bearing-strip approximation.
The first observation is a direction, so source radii are in (5, 1500].
We use closures when computing diameter; open boundaries have the same supremum.
"""
import math
import numpy as np
from numba import njit

ALPHA = math.pi / 180
R = 1500.0
NEAR = 5.0
PRIOR_AREA = ALPHA * (R * R - NEAR * NEAR)
TAU = 2 * math.pi
TOL = 2e-8


@njit(cache=True)
def constraints(a, b, bearing, half_angle=ALPHA, direction=True):
    n = np.zeros((4, 2))
    c = np.zeros(4)
    n[0, 0], n[0, 1] = math.sin(ALPHA), math.cos(ALPHA)
    n[1, 0], n[1, 1] = math.sin(ALPHA), -math.cos(ALPHA)
    if direction:
        n[2, 0], n[2, 1] = -math.sin(bearing-half_angle), math.cos(bearing-half_angle)
        n[3, 0], n[3, 1] = math.sin(bearing+half_angle), -math.cos(bearing+half_angle)
        c[2] = n[2, 0]*a+n[2, 1]*b
        c[3] = n[3, 0]*a+n[3, 1]*b
    return n, c, 4 if direction else 2


@njit(cache=True)
def inside_convex(x, y, n, c, m, centers, radii, nd):
    for i in range(m):
        if n[i, 0]*x+n[i, 1]*y < c[i]-TOL:
            return False
    for j in range(nd):
        dx, dy = x-centers[j, 0], y-centers[j, 1]
        if dx*dx+dy*dy > radii[j]*radii[j]+TOL*max(1.0,radii[j]):
            return False
    return True


@njit(cache=True)
def circle_partition(cx, cy, rad, n, c, m, centers, radii, nd, own):
    angles = np.zeros(40)
    k = 1
    for i in range(m):
        value = (c[i]-n[i, 0]*cx-n[i, 1]*cy)/rad
        if -1 <= value <= 1:
            phi = math.atan2(n[i, 1],n[i, 0])
            delta = math.acos(value)
            angles[k], angles[k+1] = (phi-delta)%TAU, (phi+delta)%TAU
            k += 2
    for j in range(nd):
        if j == own:
            continue
        dx, dy = centers[j, 0]-cx, centers[j, 1]-cy
        dist = math.hypot(dx,dy)
        if dist > 1e-12:
            value = (rad*rad+dist*dist-radii[j]*radii[j])/(2*rad*dist)
            if -1 <= value <= 1:
                phi, delta = math.atan2(dy,dx), math.acos(value)
                angles[k], angles[k+1] = (phi-delta)%TAU, (phi+delta)%TAU
                k += 2
    angles[k] = TAU
    return np.sort(angles[:k+1])


@njit(cache=True)
def convex_area(n, c, m, centers, radii, nd):
    """Green's theorem on feasible straight segments and circular arcs."""
    total = 0.0
    for i in range(m):
        # n.x>=c; directed boundary has feasible interior on its left.
        px, py = n[i,0]*c[i], n[i,1]*c[i]
        dx, dy = n[i,1], -n[i,0]
        lo, hi = -1e100, 1e100
        redundant = False
        for j in range(m):
            if i == j:
                continue
            slope = n[j,0]*dx+n[j,1]*dy
            value = c[j]-n[j,0]*px-n[j,1]*py
            if abs(slope) < 2e-14:
                if value > TOL:
                    hi = lo-1
                    break
                if j < i and abs(value)<1e-9 and n[j,0]*n[i,0]+n[j,1]*n[i,1]>0:
                    redundant = True
            elif slope > 0:
                lo = max(lo,value/slope)
            else:
                hi = min(hi,value/slope)
        if redundant:
            continue
        for j in range(nd):
            ox, oy = px-centers[j,0], py-centers[j,1]
            projection = ox*dx+oy*dy
            disc = radii[j]*radii[j]-(ox*ox+oy*oy-projection*projection)
            if disc < -1e-8:
                hi = lo-1
                break
            root = math.sqrt(max(0.0,disc))
            lo, hi = max(lo,-projection-root), min(hi,-projection+root)
        if hi > lo:
            total += .5*(px*dy-py*dx)*(hi-lo)
    for j in range(nd):
        cx, cy, rad = centers[j,0], centers[j,1], radii[j]
        duplicate = False
        for h in range(j):
            if abs(cx-centers[h,0])+abs(cy-centers[h,1])+abs(rad-radii[h]) < 1e-12:
                duplicate = True
        if duplicate:
            continue
        angles = circle_partition(cx,cy,rad,n,c,m,centers,radii,nd,j)
        for k in range(len(angles)-1):
            t0, t1 = angles[k], angles[k+1]
            mid = (t0+t1)*.5
            if inside_convex(cx+rad*math.cos(mid),cy+rad*math.sin(mid),n,c,m,centers,radii,nd):
                total += .5*(rad*rad*(t1-t0)+rad*cx*(math.sin(t1)-math.sin(t0))-rad*cy*(math.cos(t1)-math.cos(t0)))
    return max(0.0,total)


@njit(cache=True)
def feasible(x,y,a,b,n,c,m,is_near):
    for i in range(m):
        if n[i,0]*x+n[i,1]*y < c[i]-TOL:
            return False
    r2=x*x+y*y
    if r2 < NEAR*NEAR-1e-7 or r2 > R*R+1e-5:
        return False
    d2=(x-a)*(x-a)+(y-b)*(y-b)
    return d2 <= NEAR*NEAR+1e-7 if is_near else d2 >= NEAR*NEAR-1e-7


@njit(cache=True)
def add_point(points,k,x,y,a,b,n,c,m,is_near):
    if feasible(x,y,a,b,n,c,m,is_near):
        points[k,0],points[k,1] = x,y
        return k+1
    return k


@njit(cache=True)
def boundary_candidates(n,c,m,a,b,is_near=False):
    pts=np.zeros((100,2))
    k=0
    for i in range(m):
        for j in range(i):
            det=n[i,0]*n[j,1]-n[i,1]*n[j,0]
            if abs(det)>2e-14:
                x=(c[i]*n[j,1]-n[i,1]*c[j])/det
                y=(n[i,0]*c[j]-c[i]*n[j,0])/det
                k=add_point(pts,k,x,y,a,b,n,c,m,is_near)
    centers=np.array([[0.,0.],[0.,0.],[a,b]])
    radii=np.array([R,NEAR,NEAR])
    for j in range(3):
        if is_near and j==0:
            continue
        cx,cy,rad=centers[j,0],centers[j,1],radii[j]
        for i in range(m):
            h=c[i]-n[i,0]*cx-n[i,1]*cy
            if abs(h)<=rad+1e-10:
                t=math.sqrt(max(0.,rad*rad-h*h))
                x,y=cx+h*n[i,0],cy+h*n[i,1]
                k=add_point(pts,k,x+t*n[i,1],y-t*n[i,0],a,b,n,c,m,is_near)
                k=add_point(pts,k,x-t*n[i,1],y+t*n[i,0],a,b,n,c,m,is_near)
    # The two 5 m circles can intersect; the 1500 m circle cannot meet
    # the second 5 m circle anywhere in the chosen safe domain (L<=1000).
    dist=math.hypot(a,b)
    if 1e-12<dist<=10.:
        h=math.sqrt(max(0.,25.-dist*dist/4))
        k=add_point(pts,k,a/2-h*b/dist,b/2+h*a/dist,a,b,n,c,m,is_near)
        k=add_point(pts,k,a/2+h*b/dist,b/2-h*a/dist,a,b,n,c,m,is_near)
    return pts[:k]


@njit(cache=True)
def diameter(n,c,m,a,b,is_near=False):
    pts=boundary_candidates(n,c,m,a,b,is_near)
    k=len(pts)
    best=0.
    for i in range(k):
        for j in range(i):
            best=max(best,(pts[i,0]-pts[j,0])**2+(pts[i,1]-pts[j,1])**2)
    if not is_near:
        # Outer R-circle arc interiors cannot be farthest from another
        # point in the same 2-degree origin sector. Hole-arc interiors
        # are not extreme points. Thus the finite candidates suffice.
        return math.sqrt(best)
    # Near: an extremum may be a vertex vs interior of the p-centered arc.
    for i in range(k):
        vx,vy=a-pts[i,0],b-pts[i,1]
        d=math.hypot(vx,vy)
        if d>1e-12:
            x,y=a+NEAR*vx/d,b+NEAR*vy/d
            if feasible(x,y,a,b,n,c,m,True):
                best=max(best,(x-pts[i,0])**2+(y-pts[i,1])**2)
    # Or two antipodal feasible circle points, yielding diameter exactly 10.
    cent=np.array([[0.,0.]])
    rr=np.array([NEAR])
    base=circle_partition(a,b,NEAR,n,c,m,cent,rr,1,-1)
    angles=np.empty(2*len(base)+1)
    for i in range(len(base)):
        angles[i]=base[i]%TAU
        angles[i+len(base)]=(base[i]+math.pi)%TAU
    angles[-1]=TAU
    angles=np.sort(angles)
    for i in range(len(angles)-1):
        t=.5*(angles[i]+angles[i+1])
        dx,dy=NEAR*math.cos(t),NEAR*math.sin(t)
        if feasible(a+dx,b+dy,a,b,n,c,m,True) and feasible(a-dx,b-dy,a,b,n,c,m,True):
            return 10.
    return math.sqrt(best)


@njit(cache=True)
def observation_geometry(a,b,bearing,half_angle=ALPHA):
    a,b=float(a),float(b)
    n,c,m=constraints(a,b,bearing,half_angle,True)
    cent=np.array([[0.,0.],[a,b]])
    rr=np.array([R,NEAR])
    area=convex_area(n,c,m,cent,rr,1)
    rr[0]=NEAR
    area-=convex_area(n,c,m,cent,rr,1)
    nearcent=np.array([[a,b]])
    nearrr=np.array([NEAR])
    area-=convex_area(n,c,m,nearcent,nearrr,1)
    if a*a+b*b<100.:
        area+=convex_area(n,c,m,cent,rr,2)
    return max(0.,area),diameter(n,c,m,a,b,False)


@njit(cache=True)
def near_geometry(a,b):
    a,b=float(a),float(b)
    n,c,m=constraints(a,b,0.,ALPHA,False)
    cent=np.array([[a,b],[0.,0.]])
    rr=np.array([NEAR,NEAR])
    area=convex_area(n,c,m,cent,rr,1)
    if a*a+b*b<100.:
        area-=convex_area(n,c,m,cent,rr,2)
    if area<1e-9:
        return 0.,0.
    return area,diameter(n,c,m,a,b,True)


@njit(cache=True)
def true_bearing_density(a,b,phi):
    dx,dy=math.cos(phi),math.sin(phi)
    proj=a*dx+b*dy
    disc=proj*proj+R*R-a*a-b*b
    lo,hi=NEAR,-proj+math.sqrt(max(0.,disc))
    s,co=math.sin(ALPHA),math.cos(ALPHA)
    for sign in [-1.,1.]:
        value=s*a+sign*co*b
        slope=s*dx+sign*co*dy
        if abs(slope)<1e-14:
            if value < 0:
                return 0.
        elif slope>0:
            lo=max(lo,-value/slope)
        else:
            hi=min(hi,-value/slope)
    if hi<=lo:
        return 0.
    mass=.5*(hi*hi-lo*lo)
    innerdisc=proj*proj+NEAR*NEAR-a*a-b*b
    if innerdisc>0:
        sr=math.sqrt(innerdisc)
        il,ih=max(lo,-proj-sr),min(hi,-proj+sr)
        if ih>il:
            mass-=.5*(ih*ih-il*il)
    return max(0.,mass)/PRIOR_AREA


@njit(cache=True)
def geometry_array(a,b,angles,half_angle=ALPHA):
    areas=np.empty(len(angles))
    diameters=np.empty(len(angles))
    densities=np.empty(len(angles))
    for i in range(len(angles)):
        areas[i],diameters[i]=observation_geometry(a,b,angles[i],half_angle)
        densities[i]=true_bearing_density(a,b,angles[i])
    return areas,diameters,densities


@njit(cache=True)
def mec_radius(n,c,m,a,b):
    pts=boundary_candidates(n,c,m,a,b,False)
    # Remove duplicates; otherwise coincident active boundaries inflate work.
    unique=np.zeros_like(pts);k=0
    for i in range(len(pts)):
        repeated=False
        for j in range(k):
            if (pts[i,0]-unique[j,0])**2+(pts[i,1]-unique[j,1])**2<1e-14:repeated=True
        if not repeated:
            unique[k]=pts[i];k+=1
    if k<=1:return 0.
    best=1e100
    for i in range(k):
        for j in range(i):
            cx,cy=(unique[i,0]+unique[j,0])/2,(unique[i,1]+unique[j,1])/2
            rr=(unique[i,0]-cx)**2+(unique[i,1]-cy)**2
            valid=True
            for h in range(k):
                if (unique[h,0]-cx)**2+(unique[h,1]-cy)**2>rr+1e-7:valid=False;break
            if valid:best=min(best,rr)
            for h in range(j):
                ux,uy=unique[j,0]-unique[i,0],unique[j,1]-unique[i,1]
                vx,vy=unique[h,0]-unique[i,0],unique[h,1]-unique[i,1]
                det=2*(ux*vy-uy*vx)
                if abs(det)<1e-12:continue
                uu,vv=ux*ux+uy*uy,vx*vx+vy*vy
                ox,oy=(uu*vy-uy*vv)/det,(ux*vv-uu*vx)/det
                cx,cy=unique[i,0]+ox,unique[i,1]+oy;rr=ox*ox+oy*oy
                if rr>=best:continue
                valid=True
                for h2 in range(k):
                    if (unique[h2,0]-cx)**2+(unique[h2,1]-cy)**2>rr+1e-7:valid=False;break
                if valid:best=rr
    return math.sqrt(best)


@njit(cache=True)
def mec_array(a,b,angles,half_angle=ALPHA):
    result=np.empty(len(angles))
    for i in range(len(angles)):
        n,c,m=constraints(a,b,angles[i],half_angle,True)
        result[i]=mec_radius(n,c,m,a,b)
    return result


@njit(cache=True)
def sample_diameters(a,b,source_x,source_y,errors,quantized=False,expanded=False):
    count=len(errors)
    diam=np.empty(count);contained=np.ones(count,dtype=np.bool_);radii=np.empty(count)
    reads=np.empty(count)
    na,nd=near_geometry(a,b)
    half=ALPHA+(math.pi/36000 if expanded else 0.)
    for i in range(count):
        dx,dy=source_x[i]-a,source_y[i]-b
        phi=math.atan2(dy,dx)
        reading=phi+errors[i]
        if quantized:reading=round(math.degrees(reading)*100)/100*math.pi/180
        reads[i]=reading
        if dx*dx+dy*dy<=25:
            diam[i]=nd;radii[i]=5.
        else:
            area,diam[i]=observation_geometry(a,b,reading,half)
            delta=(reading-phi+math.pi)%TAU-math.pi
            contained[i]=abs(delta)<=half+1e-12
            n,c,m=constraints(a,b,reading,half,True)
            radii[i]=mec_radius(n,c,m,a,b)
    return diam,contained,radii,reads
