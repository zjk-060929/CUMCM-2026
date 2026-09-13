"""Iteration 8: reinsert a failed service along the remaining route.

Future feedback remains a finite-cost approximation, not a simulated oracle.
The local variant only rearranges near-term tasks in a fixed station sequence.
"""
import math
import time

from service_risk_strategy import ServiceRiskPolicy, ServiceRiskModel, should_wait
from mission_cost_strategy import GeometricRouteModel
from optimized_geometry import SAFE_RADIUS
from heuristic_search import optimize
from planned_strategy import insert_tasks


VARIANTS = {
    'previous': 'iteration7_riskgate',
    'reference': 'iteration5_skip',
    'insert': dict(scope='global'),
    'capped': dict(scope='global',cap_s=60.),
    'local': dict(scope='local',wait_mode='off'),
    'local_capped': dict(scope='local',cap_s=60.,wait_mode='off'),
    'ungated': dict(scope='global',cap_s=60.,wait_mode='off'),
}


def head_candidates(order,stations,blocked):
    """At most three tasks through the next two stations; stations never swap."""
    station_indices=[i for i,k in enumerate(order) if k in stations]
    end=station_indices[1]+1 if len(station_indices)>1 else len(order)
    tasks=[k for k in order[:end] if k not in stations and k not in blocked][:3]
    results={tuple(order):list(order)}
    for task in tasks:
        rest=[k for k in order if k!=task]
        placements=[0]+[i+1 for i,k in enumerate(rest) if k in stations][:2]
        for i in placements:
            candidate=rest[:i]+[task]+rest[i:]
            last=max((j for j,k in enumerate(candidate) if k in stations),default=-1)
            if any(j<last and k in blocked for j,k in enumerate(candidate)):
                continue
            results[tuple(candidate)]=candidate
    return list(results.values())


class RecourseModel(ServiceRiskModel):
    def __init__(self,*args,cap_s=None,**kwargs):
        super().__init__(*args,**kwargs)
        if cap_s is not None and (not math.isfinite(cap_s) or cap_s<0):
            raise ValueError('cap_s must be nonnegative and finite, or None')
        self.cap_s=cap_s
        self.risk_enabled=True

    def radius_after_station(self,task,station,radius):
        if radius<=SAFE_RADIUS:
            return radius
        for k,predicted,possible in self.effects.get(station,[]):
            if k==task and possible:
                if not self.skip_gain or (radius-predicted)/max(radius,1.)>=self.skip_gain:
                    radius=min(radius,predicted)
        return radius

    def retry_insertion_cost(self,route,index,radius):
        """Cheapest future insertion after at least one additional station.

        Unlike return-from-tail pricing, charge only the extra path replacing
        an existing future edge. The copied radius is a forecast, never truth.
        """
        task=route[index]
        visited_new_station=False
        best=math.inf
        for j in range(index+1,len(route)):
            anchor=route[j]
            if anchor in self.stations:
                visited_new_station=True
                radius=self.radius_after_station(task,anchor,radius)
            if not visited_new_station:
                continue
            extra=self.distance[anchor][task]
            if j+1<len(route):
                following=route[j+1]
                extra+=self.distance[task][following]-self.distance[anchor][following]
            service=5. if radius<=SAFE_RADIUS else 17.+self.weight*max(0.,radius-SAFE_RADIUS)
            best=min(best,max(0.,extra)+service)
        return best

    def failure_return_charge(self,route):
        if not self.risk_enabled or not self.return_weight:
            return 0.
        radii=dict(self.task_radii)
        left=len(self.stations)
        charge=0.
        for i,node in enumerate(route):
            if node in self.stations:
                left-=1
                for task in radii:
                    radii[task]=self.radius_after_station(task,node,radii[task])
            else:
                radius=radii.pop(node,0.)
                profile=self.service_profiles.get(node)
                if not left or not profile or radius<=SAFE_RADIUS:
                    continue
                residual=max(0.,radius-SAFE_RADIUS)/max(1.,self.task_radii[node]-SAFE_RADIUS)
                fraction=profile['blind_fraction']*min(1.,residual)
                if fraction<=0.:
                    continue
                retry=self.retry_insertion_cost(route,i,radius)
                assert math.isfinite(retry)
                penalty=self.return_weight*fraction*retry
                charge+=penalty if self.cap_s is None else min(self.cap_s,penalty)
        return charge


class RecoursePolicy(ServiceRiskPolicy):
    def __init__(self,client,*,scope='global',cap_s=None,wait_mode='risk'):
        if scope not in ('global','local'):
            raise ValueError('scope must be global or local')
        if scope=='local' and wait_mode!='off':
            raise ValueError('Local comparison uses no global waiting constraints')
        super().__init__(client,return_weight=.5,wait_mode=wait_mode)
        self.recourse_scope,self.recourse_cap=scope,cap_s
        self.local_order_changes=self.local_head_changes=0
        self.local_candidate_evaluations=0

    def plan(self):
        started = time.perf_counter()
        stations = [('station',i) for i in self.unvisited]
        tasks = [('task',ch) for ch in sorted(self.discovered-self.cleared)]
        labels = stations+tasks
        positions = {k:self.station_points[k[1]] for k in stations}
        blocked = set()
        profiles = {k:self.service_profile(k[1]) for k in tasks}
        waiting = set()
        for k in tasks:
            t = self.tracks[k[1]]
            positions[k] = t.center if t.radius<=SAFE_RADIUS else self.estimate(k[1])[0]
            wait = should_wait(self.wait_mode,len(t.observations),bool(self.negative.get(k[1])),profiles[k]['sample_radius'],profiles[k]['blind_fraction'])
            if wait:
                waiting.add(k[1])
            if t.radius>SAFE_RADIUS and (k[1] in self.scan_attempted or wait):
                blocked.add(k)
        initial = insert_tasks(stations,[k for k in tasks if k not in blocked],positions,self.client.position)+sorted(blocked)
        index = {k:i for i,k in enumerate(labels)}
        initial_ids = [index[k] for k in initial]
        warm = [index[k] for k in self.last_plan if k in index and k not in blocked]
        warm += [index[k] for k in self.last_plan if k in blocked]
        effects = {}
        for k in tasks:
            predictions = self.predicted_station_effects(k[1],positions[k])
            for station in stations:
                radius,possible = predictions[station[1]]
                # Conservative ranking only; actual polygon clipping is untouched.
                radius += self.effect_blend*max(0.,self.tracks[k[1]].radius-radius)
                effects.setdefault(index[station],[]).append((index[k],radius,possible))
        common = (labels,[positions[k] for k in labels],self.client.position,
                  {index[k]:self.tracks[k[1]].radius for k in tasks},effects,{index[k] for k in blocked})
        geometric_reference = 0.
        if self.detour_weight:
            geometry = GeometricRouteModel(*common)
            geometric_order,geometry_stats = optimize(geometry,initial_ids,method='vnd',
                                                       seed=9817+self.schedule_replans,warm=warm)
            geometric_reference = geometry.path_seconds(geometric_order)
            self.geometric_route_evaluations += geometry_stats['cost_evaluations']
            # Include this candidate in the SAME objective's final comparison.
        model = RecourseModel(*common,cap_s=self.recourse_cap,service_profiles={index[k]:profiles[k] for k in tasks},
                    return_weight=self.return_weight,uncertainty_weight=self.uncertainty_weight,
                    unknown_channels=20-len(self.discovered),
                    discovery=({index[k]:self.visibility[k[1]] for k in stations},self.unseen),
                    station_origin=self.last_station_position,station_prefix_m=self.station_prefix_distance,
                    station_limit_m=None,skip_gain=self.skip_gain,
                    travel_discount=self.travel_discount,detour_weight=self.detour_weight,
                    geometric_reference_s=geometric_reference)
        model.risk_enabled=self.recourse_scope=='global'
        order,stats = optimize(model,initial_ids,method='beam',iterations=48,
                               seed=8137+self.schedule_replans,warm=warm)
        if self.detour_weight and model.cost(geometric_order)<model.cost(order)-1e-7:
            order = geometric_order
            stats['final_cost'] = model.cost(order)
        local_base=list(order)
        if self.recourse_scope=='local':
            model.risk_enabled=True
            candidates=head_candidates(order,model.stations,model.blocked)
            order=min(candidates,key=model.cost)
            assert [k for k in order if k in model.stations]==[k for k in local_base if k in model.stations]
            self.local_candidate_evaluations+=len(candidates)
            self.local_order_changes+=order!=local_base
            self.local_head_changes+=bool(order) and order[0]!=local_base[0]
            stats['initial_cost']=model.cost(initial_ids)
            stats['final_cost']=model.cost(order)
        stats['cost_evaluations'] = model.calls
        potential = model.station_path_s(order)*5
        self.maximum_accepted_station_potential = max(self.maximum_accepted_station_potential,potential)
        result = [labels[i] for i in order]
        self.unvisited = [k[1] for k in result if k[0]=='station']
        self.last_plan = result
        self.schedule_replans += 1
        self.schedule_planning_s += time.perf_counter()-started
        self.meta_evaluations += stats['cost_evaluations']
        self.meta_proxy_savings += stats['initial_cost']-stats['final_cost']
        record = dict(station_potential_m=potential,full_remaining_route_m=model.path_seconds(order)*5,
                      geometric_reference_m=geometric_reference*5,predicted_remaining_cost_s=stats['final_cost'])
        record['predicted_failure_return_s'] = model.failure_return_charge(order)
        record['waiting_channels'] = sorted(waiting)
        record['local_base_order']=[labels[i] for i in local_base] if self.recourse_scope=='local' else None
        self.waiting_plan_occurrences += len(waiting)
        self.waited_channels.update(waiting)
        self.planning_records.append(record)
        self.client.log('recourse_plan',order=result,**record,**stats)
        return result

    def run(self):
        result=super().run()
        result.update(policy_version='q4_recourse_iteration8_20260913',
                      recourse_scope=self.recourse_scope,recourse_cap_s=self.recourse_cap,
                      local_order_changes=self.local_order_changes,
                      local_head_changes=self.local_head_changes,
                      local_candidate_evaluations=self.local_candidate_evaluations)
        self.client.log('recourse_summary',**result)
        return result
