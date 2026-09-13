"""Iteration 5: mission-wide station-path budget and priced sensing decisions."""
import math
import time

from adaptive_service_strategy import AdaptiveServicePolicy
from geometry import dist
from heuristic_search import RouteModel, optimize
from optimized_geometry import SAFE_RADIUS
from planned_strategy import insert_tasks


VARIANTS = {
    'previous': None,
    'reference': 'reference',
    'cap0': dict(station_margin=0.),
    'cap500': dict(station_margin=500.),
    'posterior': dict(station_margin=500., posterior=True),
    'skip': dict(station_margin=500., skip_gain=.08),
    'combined': dict(station_margin=500., posterior=True, skip_gain=.08),
    'uncapped': dict(station_margin=None, posterior=True, skip_gain=.08),
}


def population_posterior(discovered, unseen_fraction):
    """Finite uniform-count prior, iid visibility approximation, ranking only.

    Conditional count weights are C(n,d)*u**(n-d); positive-observation
    likelihood factors common to n cancel. This is not an absence certificate.
    """
    u=max(1e-8,min(1.,unseen_fraction))
    raw={n:math.comb(n,discovered)*u**(n-discovered) for n in range(max(10,discovered),17)}
    normalizer=sum(raw.values())
    probability={n:w/normalizer for n,w in raw.items()}
    return sum((n-discovered)*p for n,p in probability.items()),probability.get(16,0.)


class BudgetedRouteModel(RouteModel):
    def __init__(self,*args,station_origin,station_prefix_m,station_limit_m,
                 unseen_fraction=None,skip_gain=0.,**kwargs):
        super().__init__(*args,**kwargs)
        points=args[1]
        self.station_from_anchor={i:dist(station_origin,points[i])/5 for i in self.stations}
        self.station_prefix_s=station_prefix_m/5
        self.station_limit_s=math.inf if station_limit_m is None else station_limit_m/5
        self.skip_gain=skip_gain
        d=20-self.unknown
        if unseen_fraction is None:
            self.expected_remaining=(max(10,d)+16)/2-d
            self.probability16=1/(17-max(10,d))
        else:
            self.expected_remaining,self.probability16=population_posterior(d,unseen_fraction)
        self.rejected_orders=0

    def station_path_s(self,route):
        stations=[i for i in route if i in self.stations]
        if not stations:return self.station_prefix_s
        return (self.station_prefix_s+self.station_from_anchor[stations[0]]+
                sum(self.distance[a][b] for a,b in zip(stations,stations[1:])))

    def cost(self,route):
        self.calls+=1
        if self.station_path_s(route) > self.station_limit_s+1e-6:
            self.rejected_orders+=1
            return 1e12
        if not route:return 0.
        score=self.from_start[route[0]]+sum(self.distance[a][b] for a,b in zip(route,route[1:]))
        discovery_costs={}
        if self.discovery and self.unknown>4:
            masks,hidden=self.discovery
            total=hidden.bit_count()
            last=None
            for node in route:
                if node in self.stations:
                    seen=1-hidden.bit_count()/total if total else 0.
                    continuing=1-self.probability16*seen**(self.unknown-4)
                    edge=self.from_start[node] if last is None else self.distance[last][node]
                    score-=(1-continuing)*edge
                    discovery_costs[node]=continuing*6*(self.unknown-self.expected_remaining*seen)
                    hidden &= ~masks[node]
                last=node
        radii=dict(self.task_radii)
        remaining_stations=len(self.stations)
        for node in route:
            if node in self.stations:
                remaining_stations-=1
                score+=discovery_costs.get(node,6*self.unknown)
                for task,predicted,possible in self.effects.get(node,[]):
                    radius=radii.get(task,0.)
                    if possible and radius>SAFE_RADIUS:
                        if self.skip_gain and (radius-predicted)/max(radius,1.) < self.skip_gain:
                            continue
                        score+=6.
                        radii[task]=min(radius,predicted)
            else:
                if node in self.blocked and remaining_stations:score+=1e7
                radius=radii.pop(node,0.)
                score+=5+self.weight*max(0.,radius-SAFE_RADIUS)+(6 if radius>SAFE_RADIUS else 0)
        return score


class GuardedRoutePolicy(AdaptiveServicePolicy):
    def __init__(self,client,*,station_margin=500.,posterior=False,skip_gain=0.):
        if station_margin is not None and (not math.isfinite(station_margin) or station_margin<0):
            raise ValueError('station_margin must be nonnegative or None')
        if not 0<=skip_gain<=1:raise ValueError('skip_gain must be between 0 and 1')
        super().__init__(client,order='beam',endpoint_gate='evidence')
        self.station_margin,self.use_population_posterior,self.skip_gain=station_margin,posterior,skip_gain
        self.original_station_distance=sum(dist(a,b) for a,b in zip([(0.,0.)]+list(self.station_points),self.station_points))
        self.station_limit=None if station_margin is None else self.original_station_distance+station_margin
        self.station_prefix_distance=0.
        self.last_station_position=(0.,0.)
        self.point_to_station={p:i for i,p in enumerate(self.station_points)}
        self.rejected_route_candidates=self.low_gain_measure_skips=0
        self.maximum_accepted_station_potential=0.
        self.posterior_records=[]
        self.full_hypothesis_count=self.unseen.bit_count()

    def measure(self,p,ch):
        if self.skip_gain and self.in_station_scan and ch in self.discovered and ch not in self.cleared:
            t=self.tracks[ch]
            i=self.point_to_station.get(tuple(p))
            if i is not None and t.radius>SAFE_RADIUS:
                center,_=self.estimate(ch)
                predicted,_=self.predicted_station_effects(ch,center)[i]
                gain=(t.radius-predicted)/max(t.radius,1.)
                if gain<self.skip_gain:
                    self.low_gain_measure_skips+=1
                    self.client.log('low_gain_station_measure_skipped',channel=ch,position=p,
                                    estimated_fractional_radius_gain=gain)
                    # No fabricated no-signal record and no geometric narrowing.
                    return 'skipped_low_gain'
        return super().measure(p,ch)

    def scan_station(self,station_id):
        super().scan_station(station_id)
        p=self.station_points[station_id]
        self.station_prefix_distance+=dist(self.last_station_position,p)
        self.last_station_position=p

    def plan(self):
        started=time.perf_counter()
        stations=[('station',i) for i in self.unvisited]
        tasks=[('task',ch) for ch in sorted(self.discovered-self.cleared)]
        labels=stations+tasks
        positions={k:self.station_points[k[1]] for k in stations}
        blocked=set()
        for k in tasks:
            t=self.tracks[k[1]]
            positions[k]=t.center if t.radius<=SAFE_RADIUS else self.estimate(k[1])[0]
            if t.radius>SAFE_RADIUS and k[1] in self.scan_attempted:blocked.add(k)
        active=[k for k in tasks if k not in blocked]
        initial=insert_tasks(stations,active,positions,self.client.position)+sorted(blocked)
        index={k:i for i,k in enumerate(labels)}
        effects={}
        for k in tasks:
            predictions=self.predicted_station_effects(k[1],positions[k])
            for station in stations:
                radius,possible=predictions[station[1]]
                effects.setdefault(index[station],[]).append((index[k],radius,possible))
        fraction=self.unseen.bit_count()/self.full_hypothesis_count
        model=BudgetedRouteModel(labels,[positions[k] for k in labels],self.client.position,
              {index[k]:self.tracks[k[1]].radius for k in tasks},effects,{index[k] for k in blocked},
              self.uncertainty_weight,unknown_channels=20-len(self.discovered),
              discovery=({index[k]:self.visibility[k[1]] for k in stations},self.unseen),
              station_origin=self.last_station_position,station_prefix_m=self.station_prefix_distance,
              station_limit_m=self.station_limit,unseen_fraction=fraction if self.use_population_posterior else None,
              skip_gain=self.skip_gain)
        initial_ids=[index[k] for k in initial]
        assert model.station_path_s(initial_ids)<=model.station_limit_s+1e-6
        warm=[index[k] for k in self.last_plan if k in index and k not in blocked]
        warm += [index[k] for k in self.last_plan if k in blocked]
        order,stats=optimize(model,initial_ids,method='beam',iterations=48,
                             seed=8137+self.schedule_replans,warm=warm)
        potential=model.station_path_s(order)*5
        assert self.station_limit is None or potential<=self.station_limit+1e-5
        self.maximum_accepted_station_potential=max(self.maximum_accepted_station_potential,potential)
        self.rejected_route_candidates+=model.rejected_orders
        result=[labels[i] for i in order]
        self.unvisited=[k[1] for k in result if k[0]=='station']
        self.last_plan=result
        self.schedule_replans+=1
        self.schedule_planning_s+=time.perf_counter()-started
        self.meta_evaluations+=stats['cost_evaluations']
        self.meta_accepted_worse+=stats['accepted_worse']
        self.meta_proxy_savings+=stats['initial_cost']-stats['final_cost']
        self.posterior_records.append(dict(discovered=len(self.discovered),unseen_fraction=fraction,
                                          expected_remaining=model.expected_remaining,probability16=model.probability16))
        self.client.log('guarded_route_plan',order=result,station_potential_m=potential,
                        station_limit_m=self.station_limit,rejected_candidates=model.rejected_orders,
                        expected_remaining=model.expected_remaining,probability16=model.probability16,**stats)
        return result

    def run(self):
        result=super().run()
        result.update(policy_version='q4_guarded_route_iteration5_20260913',
                      station_margin_m=self.station_margin,station_limit_m=self.station_limit,
                      original_station_distance_m=self.original_station_distance,
                      station_prefix_distance_m=self.station_prefix_distance,
                      maximum_accepted_station_potential_m=self.maximum_accepted_station_potential,
                      rejected_route_candidates=self.rejected_route_candidates,
                      population_posterior_enabled=self.use_population_posterior,
                      low_gain_threshold=self.skip_gain,low_gain_measure_skips=self.low_gain_measure_skips)
        self.client.log('guarded_route_summary',**result)
        return result
