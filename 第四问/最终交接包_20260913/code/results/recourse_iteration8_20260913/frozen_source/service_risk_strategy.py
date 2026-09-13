"""Iteration 7: price a failed early service and wait for usable evidence.

Finite transmitter hypotheses rank actions only. Joint non-reception is tested
on the SAME transmitter hypothesis; no independent angular-noise draws occur.
"""
import time

from geometry import dist, dot, sub, add, mul, unit
from optimized_geometry import SAFE_RADIUS, sample_polygon, possible_emitter_at
from mission_cost_strategy import MissionCostPolicy, MissionCostModel, GeometricRouteModel
from heuristic_search import optimize
from planned_strategy import insert_tasks


VARIANTS = {
    'previous': 'iteration6_cautious',
    'reference': 'iteration5_skip',
    'risk': dict(return_weight=.5),
    'riskstrong': dict(return_weight=1.),
    'weakgate': dict(return_weight=0., wait_mode='weak'),
    'combined': dict(return_weight=.5, wait_mode='weak'),
    'riskgate': dict(return_weight=.5, wait_mode='risk'),
}


def receives(g, reach, normal, p):
    return dist(g,p)<=reach+1e-7 and (normal is None or dot(normal,sub(p,g))>=-1e-7)


def joint_blind_fraction(hypotheses, points, clear_center=None):
    """Weighted fraction of the finite set blind at ALL nominal points.

    A candidate already within optical range can be cleared directly. This
    sample fraction is not a calibrated probability or geometric certificate.
    """
    denominator = sum(w for g,reach,normal,w in hypotheses)
    if not denominator:
        return None
    blind = 0.
    for g,reach,normal,w in hypotheses:
        if clear_center is not None and dist(g,clear_center)<=20.:
            continue
        if not any(receives(g,reach,normal,p) for p in points):
            blind += w
    return blind/denominator


def should_wait(mode, positive_count, has_negative, estimated_radius, blind_fraction):
    if estimated_radius<=80.:
        return False
    if mode=='weak':
        return positive_count<2 and has_negative
    if mode=='risk':
        return blind_fraction>=.6 and (positive_count<2 or has_negative)
    return False


class ServiceRiskModel(MissionCostModel):
    def __init__(self,*args,service_profiles=None,return_weight=0.,**kwargs):
        super().__init__(*args,**kwargs)
        self.service_profiles = service_profiles or {}
        self.return_weight = return_weight

    def failure_return_charge(self,route):
        if not route or not self.return_weight:
            return 0.
        radii = dict(self.task_radii)
        left = len(self.stations)
        charge = 0.
        for node in route:
            if node in self.stations:
                left -= 1
                for task,predicted,possible in self.effects.get(node,[]):
                    radius = radii.get(task,0.)
                    if possible and radius>SAFE_RADIUS:
                        if self.skip_gain and (radius-predicted)/max(radius,1.)<self.skip_gain:
                            continue
                        radii[task] = min(radius,predicted)
            else:
                radius = radii.pop(node,0.)
                profile = self.service_profiles.get(node)
                if not left or not profile or radius<=SAFE_RADIUS:
                    continue
                # Earlier stations may reduce uncertainty before this service.
                # Retain the unresolved task in a one-failure tail approximation.
                residual = max(0.,radius-SAFE_RADIUS)/max(1.,self.task_radii[node]-SAFE_RADIUS)
                failure_fraction = profile['blind_fraction']*min(1.,residual)
                return_s = self.distance[route[-1]][node]
                retry_s = 12.+self.weight*max(0.,radius-SAFE_RADIUS)
                charge += self.return_weight*failure_fraction*(return_s+retry_s)
        return charge

    def cost(self,route):
        return super().cost(route)+self.failure_return_charge(route)


class ServiceRiskPolicy(MissionCostPolicy):
    def __init__(self,client,*,return_weight=.5,wait_mode='off'):
        if return_weight<0 or wait_mode not in ('off','weak','risk'):
            raise ValueError('Invalid service-risk settings')
        super().__init__(client,travel_discount=0.,detour_weight=.5,effect_blend=.5)
        self.return_weight,self.wait_mode = return_weight,wait_mode
        self.service_profile_cache = {}
        self.service_profile_evaluations = self.service_profile_fallbacks = 0
        self.waiting_plan_occurrences = self.risk_hypothesis_count = 0
        self.waited_channels = set()

    def service_profile(self,ch):
        t = self.tracks[ch]
        key = (ch,len(t.observations),len(self.negative.get(ch,[])),len(self.failed_clear.get(ch,[])))
        if key in self.service_profile_cache:
            return self.service_profile_cache[key]
        center,radius = self.estimate(ch)
        if t.radius<=SAFE_RADIUS:
            result = dict(blind_fraction=0.,sample_radius=radius,hypotheses=0)
            self.service_profile_cache[key] = result
            return result
        last,bearing = t.observations[-1]
        length = dist(last,center)
        toward = mul(sub(last,center),1/length) if length else unit(bearing+180)
        base = add(center,mul(toward,min(100.,max(25.,radius*.35))))
        lateral = mul(unit(bearing+90),min(150.,max(45.,radius*.4))*.5)
        # Two nominal service positions before feedback; actual service still
        # replans after observations. These do not simulate independent noise.
        points = [add(base,lateral),sub(base,lateral)]
        points = [p for p in points if not self.redundant_reason(p,ch)]
        positives = [p for p,_ in t.observations]
        negatives = self.negative.get(ch,[])
        samples = [g for g in sample_polygon(t.polygon,step=max(15.,t.radius/6))
                   if dist(g,(0.,0.))<=1800.+1e-7
                   and all(dist(g,p)<=1500.+1e-7 for p in positives)
                   and all(dist(g,p)>20. for p in self.failed_clear.get(ch,[]))
                   and possible_emitter_at(g,positives,negatives)]
        if len(samples)>9:
            samples = [samples[round(k*(len(samples)-1)/8)] for k in range(9)]
        hypotheses = []
        normals = [(unit(k*22.5),1.) for k in range(16)]+[(None,16.)]
        for g in samples:
            required = max([1000.]+[dist(g,p) for p in positives])
            for reach in dict.fromkeys((required,(required+1500.)/2,1500.)):
                for normal,weight in normals:
                    if all(receives(g,reach,normal,p) for p in positives) and not any(receives(g,reach,normal,p) for p in negatives):
                        hypotheses.append((g,reach,normal,weight))
        fraction = joint_blind_fraction(hypotheses,points,clear_center=center if radius<=80. else None)
        if fraction is None:
            self.service_profile_fallbacks += 1
            fraction = .5 if radius>80. else .1
        result = dict(blind_fraction=fraction,sample_radius=radius,hypotheses=len(hypotheses),
                      nominal_points=points)
        self.risk_hypothesis_count += len(hypotheses)
        self.service_profile_evaluations += 1
        self.service_profile_cache[key] = result
        return result

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
        model = ServiceRiskModel(*common,service_profiles={index[k]:profiles[k] for k in tasks},
                    return_weight=self.return_weight,uncertainty_weight=self.uncertainty_weight,
                    unknown_channels=20-len(self.discovered),
                    discovery=({index[k]:self.visibility[k[1]] for k in stations},self.unseen),
                    station_origin=self.last_station_position,station_prefix_m=self.station_prefix_distance,
                    station_limit_m=None,skip_gain=self.skip_gain,
                    travel_discount=self.travel_discount,detour_weight=self.detour_weight,
                    geometric_reference_s=geometric_reference)
        order,stats = optimize(model,initial_ids,method='beam',iterations=48,
                               seed=8137+self.schedule_replans,warm=warm)
        if self.detour_weight and model.cost(geometric_order)<model.cost(order)-1e-7:
            order = geometric_order
            stats['final_cost'] = model.cost(order)
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
        self.waiting_plan_occurrences += len(waiting)
        self.waited_channels.update(waiting)
        self.planning_records.append(record)
        self.client.log('service_risk_plan',order=result,**record,**stats)
        return result

    def run(self):
        result = super().run()
        result.update(policy_version='q4_service_risk_iteration7_20260913',
                      return_weight=self.return_weight,service_wait_mode=self.wait_mode,
                      service_profile_evaluations=self.service_profile_evaluations,
                      service_profile_fallbacks=self.service_profile_fallbacks,
                      risk_hypothesis_count=self.risk_hypothesis_count,
                      waiting_plan_occurrences=self.waiting_plan_occurrences,
                      waited_channels=sorted(self.waited_channels))
        self.client.log('service_risk_summary',**result)
        return result
