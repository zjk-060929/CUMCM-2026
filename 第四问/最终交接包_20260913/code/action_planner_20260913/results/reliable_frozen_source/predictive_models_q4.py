"""Observation-only time forecasts. Discrete beliefs rank actions, never certify.

No simulator error scale, seed, source truth or independent angle noise enters
these forecasts. Future angular bounds retain their full width.
"""
import math

from bayesian_belief import detection_mask,elementary
from geometry import (dist,sub,dot,unit,clip,ALPHA_DEG,polygon_distance)
from heuristic_search import RouteModel
from optimized_geometry import sample_polygon,possible_emitter_at,SAFE_RADIUS


def cap_samples(points,limit=24):
    if len(points)<=limit:return list(points)
    return [points[min(len(points)-1,int((i+.5)*len(points)/limit))] for i in range(limit)]


def box_circle(poly):
    if not poly:return (0.,0.),math.inf
    xmin,xmax=min(p[0] for p in poly),max(p[0] for p in poly)
    ymin,ymax=min(p[1] for p in poly),max(p[1] for p in poly)
    return ((xmin+xmax)/2,(ymin+ymax)/2),math.hypot(xmax-xmin,ymax-ymin)/2


def bearing_region(poly,p,bearing):
    lo,hi=unit(bearing-ALPHA_DEG),unit(bearing+ALPHA_DEG)
    for normal in ((lo[1],-lo[0]),(-hi[1],hi[0])):
        poly=clip(poly,normal,dot(normal,p))
    # Forecast alone omits range-circle clipping: larger regions are retained.
    return poly


def optical_recovery(points,start):
    """Mean sample discovery time along a bounded greedy optical tour, in seconds.

    Equal representative mass is a heuristic. It does not cover the continuous
    region and is never used as the actual finite guaranteed optical fallback.
    """
    if not points:return 5.
    pending=list(points);current=start;elapsed=total=0.
    while pending:
        q=max(pending,key=lambda q:sum(dist(q,g)<=19.8 for g in pending)/(dist(current,q)/5+3.))
        elapsed+=dist(current,q)/5+3.
        covered=[g for g in pending if dist(q,g)<=19.8]
        total+=len(covered)*(elapsed+2.)
        pending=[g for g in pending if dist(q,g)>19.8]
        current=q
    return total/len(points)


class TargetForecast:
    def __init__(self,track,negatives,failed,estimate,recovery_scale=1.):
        self.poly=list(track.polygon);self.radius=track.radius
        self.positives=[p for p,_ in track.observations]
        self.negatives=list(negatives);self.failed=list(failed)
        self.center=estimate
        self.recovery_scale=recovery_scale
        candidates=sample_polygon(self.poly)
        feasible=[g for g in candidates if dist(g,(0.,0.))<=1800.+1e-7
                  and all(dist(g,p)<=1500.+1e-7 for p in self.positives)
                  and all(dist(g,p)>20. for p in failed)
                  and possible_emitter_at(g,self.positives,self.negatives)]
        self.depleted=not feasible
        self.points=cap_samples(feasible or candidates)
        self.states=[]
        for g in self.points:
            required=max([1000.]+[dist(g,p) for p in self.positives])
            states=[]
            if required<=1500.+1e-6:
                for reach in (required,(required+1500.)/2,1500.):
                    for normal in [unit(22.5*k) for k in range(16)]+[None]*16:
                        if normal is not None and any(dot(normal,sub(p,g))< -1e-7 for p in self.positives):continue
                        if any(dist(p,g)<=reach and (normal is None or dot(normal,sub(p,g))>= -1e-7) for p in self.negatives):continue
                        states.append((reach,normal))
            self.states.append(states)
        self.cache={}
        self.initial_recovery=optical_recovery(self.points,self.center)

    def reception_at(self,q):
        probabilities=[]
        for g,states in zip(self.points,self.states):
            if states:
                probability=sum(dist(g,q)<=r and (n is None or dot(n,sub(q,g))>=0) for r,n in states)/len(states)
            else:
                probability=.25 if dist(g,q)<=1500. else 0.
            probabilities.append(probability)
        return probabilities

    def clear_prediction(self,q):
        if self.radius<=SAFE_RADIUS and dist(q,self.center)+self.radius<=20.:
            return dict(success=1.,failure_recovery=0.,operations=5.)
        hit=sum(dist(g,q)<=20. for g in self.points)/max(1,len(self.points))
        # Finite samples may miss thin tails. Never call sampled certainty proof.
        hit=min(.95,hit)
        missed=[g for g in self.points if dist(g,q)>20.]
        recovery=self.recovery_scale*optical_recovery(missed or self.points,q)
        if self.depleted:hit=min(hit,.5)
        return dict(success=hit,failure_recovery=recovery,
                    operations=hit*5+(1-hit)*(3+recovery))

    def measurement_prediction(self,q,baseline=25.):
        key=tuple(q),baseline
        if key in self.cache:return self.cache[key]
        possible=polygon_distance(q,self.poly)<=1500.
        if not possible or any(dist(q,p)<baseline for p in self.positives):
            ans=dict(reception=0.,radius=self.radius,remaining=self.initial_recovery,
                     no_signal_remaining=optical_recovery(self.points,q),valid=False)
            self.cache[key]=ans;return ans
        probabilities=self.reception_at(q)
        selected=cap_samples(list(range(len(self.points))),6)
        radius_sum=cost_sum=mass=0.
        # At each representative source, use three full-bias extremes. Take
        # their maximum, not independent draws or 1/sqrt(n) error shrinkage.
        for index in selected:
            g=self.points[index];probability=probabilities[index]
            if not probability:continue
            angle=math.degrees(math.atan2(g[1]-q[1],g[0]-q[0]))
            worst_r=worst_cost=0.
            for error in (-1.,0.,1.):
                poly=bearing_region(self.poly,q,angle+error)
                center,r=box_circle(poly)
                r=min(self.radius,r)
                continuation=5.+max(0.,self.initial_recovery-5.)*min(1.,r/max(self.radius,1.))
                continuation+=dist(q,center)/5
                worst_r=max(worst_r,r);worst_cost=max(worst_cost,continuation)
            radius_sum+=probability*worst_r;cost_sum+=probability*worst_cost;mass+=probability
        reception=sum(probabilities)/max(1,len(probabilities))
        missed=[g for g,p in zip(self.points,probabilities) if p<.95]
        no_signal=optical_recovery(missed or self.points,q)
        received=cost_sum/mass if mass else no_signal
        received_radius=radius_sum/mass if mass else self.radius
        ans=dict(reception=reception,radius=(1-reception)*self.radius+reception*received_radius,
                 remaining=reception*received+(1-reception)*no_signal,
                 received_remaining=received,no_signal_remaining=no_signal,valid=True)
        self.cache[key]=ans;return ans


class ForecastRouteModel(RouteModel):
    def __init__(self,*args,posterior=None,station_masks=None,delay_weight=.08,
                 service_costs=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.posterior=posterior;self.station_masks=station_masks or {}
        self.delay_weight=delay_weight;self.service_costs=service_costs
        self.last_forecast=None

    def cost(self,route):
        # Keep original geometric/radius score when failure-cost prediction is
        # disabled, so the development ablations have a defined baseline.
        value=super().cost(route)
        costs=dict(self.service_costs or {})
        if costs:
            radii=dict(self.task_radii)
            for node in route:
                if node in self.stations:
                    for task,predicted,possible in self.effects.get(node,[]):
                        if task in radii and possible:radii[task]=min(radii[task],predicted)
                else:
                    r=radii.pop(node,0.);old_r=self.task_radii[node]
                    old=5+self.weight*max(0.,r-19.8)+(6 if r>19.8 else 0)
                    residual=5+max(0.,costs[node]-5)*min(1.,r/max(old_r,1.))
                    value+=residual-old
        if not self.posterior:return value
        p=self.posterior
        masks=dict(p['masks']);sizes={c:m.bit_count() for c,m in masks.items()}
        found={c:0. for c in masks};elapsed=delay_sum=expected_scan=0.
        previous=None;m16=16-(20-self.unknown)
        denominator=p['coefficients'][m16] if 0<=m16<len(p['coefficients']) else 0.
        for node in route:
            travel=self.from_start[node] if previous is None else self.distance[previous][node]
            elapsed+=travel
            if node in self.stations:
                continuation=1.
                if denominator>0 and m16>0 and p['probability_n'].get(16,0.)>.001:
                    numerator=elementary([p['likelihood'][c]*found[c] for c in masks],m16)[m16]
                    continuation-=p['probability_n'].get(16,0.)*min(1.,numerator/denominator)
                scans=6*(self.unknown-sum(p['existence'][c]*found[c] for c in masks))
                value+=continuation*scans-6*self.unknown-(1-continuation)*travel
                visible=self.station_masks[node];gain=0.
                for c,m in masks.items():
                    fraction=(m & visible).bit_count()/sizes[c]
                    gain+=p['existence'][c]*fraction
                    found[c]+=fraction;masks[c]&=~visible
                delay_sum+=gain*(elapsed+scans/2)
                elapsed+=scans;expected_scan+=continuation*(travel+scans)
            else:
                elapsed+=(self.service_costs or {}).get(node,5+self.weight*max(0.,self.task_radii[node]-19.8))
            previous=node
        residue=sum(p['existence'][c]*(1-found[c]) for c in masks)
        delay_sum+=residue*elapsed
        latency=delay_sum/max(p['expected_unknown'],1.)
        self.last_forecast=dict(mean_remaining_discovery_delay_s=latency,expected_scan_s=expected_scan,
                                expected_unknown=p['expected_unknown'])
        return value+self.delay_weight*latency
