"""Observation-gated boundary probes and exterior cross-bearing rescue.

Boundary location is a hypothesis, never a restriction on the true feasible set.
All failed clears/negative bearings are paid; the original finite fallback stays.
"""
import math

from geometry import dist, dot, sub, add, mul, polygon_distance, unit
from optimized_geometry import SAFE_RADIUS
from precision_search_strategy import PrecisionSearchPolicy


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


class BoundaryRescuePolicy(PrecisionSearchPolicy):
    def __init__(self,client,*,rescue='cross'):
        if rescue not in ('optical','cross'):raise ValueError('Unknown boundary rescue')
        super().__init__(client,circle_sides=1024,search_level='standard',joint_cost=False)
        self.rescue=rescue
        self.boundary_station_channels=set()
        self.boundary_attempted=set()
        self.boundary_episodes=self.boundary_successes=self.boundary_failures=0
        self.boundary_optical_attempts=self.boundary_bearings=0
        self.boundary_cost_s=0.

    def measure(self,p,ch):
        result=super().measure(p,ch)
        if self.in_station_scan and dist(p,(0.,0.))<=1800. and result in ('direction','near'):
            self.boundary_station_channels.add(ch)
        return result

    def boundary_rescue(self,ch):
        if ch in self.boundary_attempted or ch in self.cleared:return False
        t=self.tracks[ch]
        if t.radius<=SAFE_RADIUS:return False
        # Sparse INTERNAL STATION scans, not a scenario label or simulator truth.
        if len(self.interior_stations)<3 or len(self.boundary_station_channels)>2:return False
        if self.boundary_failures>=2 and not self.boundary_successes:return False
        exterior=[p for p,_ in t.observations if dist(p,(0.,0.))>1800.]
        if not exterior:return False
        anchor=exterior[-1]
        points=arc_probe_points(t.polygon,anchor,3 if self.rescue=='optical' else 1)
        if not points:return False
        self.boundary_attempted.add(ch);self.boundary_episodes+=1
        before=self.client.virtual
        old=(self.reactive_busy,self.piggyback_busy,self.local_busy)
        self.reactive_busy=self.piggyback_busy=self.local_busy=True
        try:
            for q in points:
                if any(dist(q,p)<=5 for p in self.failed_clear.get(ch,[])):continue
                self.boundary_optical_attempts+=1;self.optimistic_clears+=1
                if self.clear(q,ch):return True
            if self.rescue!='cross':return False
            q=points[0];normal=mul(q,1/1800.);lateral=(-normal[1],normal[0])
            candidates=[add(add(q,mul(normal,60.)),mul(lateral,side*90.)) for side in (-1,1)]
            candidates.sort(key=lambda p:dist(p,self.client.position))
            for p in candidates:
                if self.redundant_reason(p,ch):continue
                if any(dist(p,x)<=30 for x in self.negative.get(ch,[])):continue
                self.boundary_bearings+=1
                self.measure(p,ch)
                if t.cleared:return True
                if t.radius<=SAFE_RADIUS:
                    return self.certified_clear(ch)
                center,radius=self.estimate(ch)
                if radius<=80 and all(dist(center,x)>5 for x in self.failed_clear.get(ch,[])):
                    self.boundary_optical_attempts+=1;self.optimistic_clears+=1
                    if self.clear(center,ch):return True
            return False
        finally:
            self.reactive_busy,self.piggyback_busy,self.local_busy=old
            success=t.cleared
            self.boundary_successes+=success;self.boundary_failures+=not success
            self.boundary_cost_s+=self.client.virtual-before
            self.client.log('boundary_rescue',channel=ch,success=success,
                            elapsed_s=self.client.virtual-before,rescue=self.rescue,
                            station_interior_channels=len(self.boundary_station_channels))
            if success and self.local and not self.local_busy:self.local_bundle()

    def bounded_scan_service(self,ch):
        if not self.boundary_rescue(ch):super().bounded_scan_service(ch)

    def service(self,ch):
        if not self.boundary_rescue(ch):super().service(ch)

    def run(self):
        result=super().run()
        result.update(policy_version='q4_boundary_rescue_20260913',boundary_rescue_mode=self.rescue,
                      boundary_episodes=self.boundary_episodes,boundary_successes=self.boundary_successes,
                      boundary_failures=self.boundary_failures,
                      boundary_optical_attempts=self.boundary_optical_attempts,
                      boundary_bearings=self.boundary_bearings,boundary_cost_s=self.boundary_cost_s)
        return result
