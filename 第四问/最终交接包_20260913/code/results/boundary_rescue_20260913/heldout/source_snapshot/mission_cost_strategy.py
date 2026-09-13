"""Iteration 6: charge whole-route travel and soften optimistic planning.

All future costs remain ranking surrogates. Coverage, bounded angular errors,
nearby-observation filtering and the finite optical fallback are inherited.
"""
import math
import time

from guarded_route_strategy import GuardedRoutePolicy, BudgetedRouteModel
from heuristic_search import RouteModel, optimize
from optimized_geometry import SAFE_RADIUS
from planned_strategy import insert_tasks


VARIANTS = {
    'previous': 'iteration5_skip',
    'reference': 'iteration2_fair',
    'uncapped': dict(),
    'fulltravel': dict(travel_discount=0.),
    'soft': dict(travel_discount=0., detour_weight=.5),
    'softstrong': dict(travel_discount=0., detour_weight=1.),
    'nolocal': dict(travel_discount=0., detour_weight=.5, local=False),
    'cautious': dict(travel_discount=0., detour_weight=.5, effect_blend=.5),
}


class GeometricRouteModel(RouteModel):
    def path_seconds(self, route):
        return (self.from_start[route[0]]+
                sum(self.distance[a][b] for a,b in zip(route,route[1:]))) if route else 0.

    def cost(self, route):
        self.calls += 1
        last_station = max((j for j,k in enumerate(route) if k in self.stations), default=-1)
        return self.path_seconds(route)+1e7*sum(j<last_station and k in self.blocked for j,k in enumerate(route))


class MissionCostModel(BudgetedRouteModel):
    def __init__(self, *args, travel_discount=1., detour_weight=0.,
                 geometric_reference_s=0., detour_allowance_m=250., **kwargs):
        super().__init__(*args, **kwargs)
        if self.station_limit_s != math.inf:
            raise ValueError('Iteration 6 does not hard-cap the station skeleton')
        self.travel_discount = travel_discount
        self.detour_weight = detour_weight
        self.geometric_reference_s = geometric_reference_s
        self.detour_allowance_s = detour_allowance_m/5

    path_seconds = GeometricRouteModel.path_seconds

    def cost(self, route):
        score = super().cost(route)
        # Undo some/all of the old prior-based discount on future station legs.
        # A yet-undiscovered 16th source must not make physical travel free.
        if self.travel_discount != 1. and self.discovery and self.unknown>4:
            masks, hidden = self.discovery
            total = hidden.bit_count()
            last = None
            for node in route:
                if node in self.stations:
                    seen = 1-hidden.bit_count()/total if total else 0.
                    stop_probability = self.probability16*seen**(self.unknown-4)
                    edge = self.from_start[node] if last is None else self.distance[last][node]
                    score += (1-self.travel_discount)*stop_probability*edge
                    hidden &= ~masks[node]
                last = node
        # Soft charge on the FULL route through stations AND pending targets.
        # It never rejects a route merely because station-to-station travel grows.
        excess = self.path_seconds(route)-self.geometric_reference_s-self.detour_allowance_s
        return score+self.detour_weight*max(0.,excess)


class MissionCostPolicy(GuardedRoutePolicy):
    def __init__(self, client, *, travel_discount=1., detour_weight=0.,
                 effect_blend=0., local=True):
        if not 0<=travel_discount<=1 or detour_weight<0 or not 0<=effect_blend<=1:
            raise ValueError('Invalid mission-cost option')
        super().__init__(client, station_margin=None, skip_gain=.08)
        self.travel_discount = travel_discount
        self.detour_weight = detour_weight
        self.effect_blend = effect_blend
        self.local = local
        self.geometric_route_evaluations = 0
        self.planning_records = []

    def plan(self):
        started = time.perf_counter()
        stations = [('station',i) for i in self.unvisited]
        tasks = [('task',ch) for ch in sorted(self.discovered-self.cleared)]
        labels = stations+tasks
        positions = {k:self.station_points[k[1]] for k in stations}
        blocked = set()
        for k in tasks:
            t = self.tracks[k[1]]
            positions[k] = t.center if t.radius<=SAFE_RADIUS else self.estimate(k[1])[0]
            if t.radius>SAFE_RADIUS and k[1] in self.scan_attempted:
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
        model = MissionCostModel(*common,uncertainty_weight=self.uncertainty_weight,
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
        self.planning_records.append(record)
        self.client.log('mission_cost_plan',order=result,**record,**stats)
        return result

    def run(self):
        result = super().run()
        result.update(policy_version='q4_mission_cost_iteration6_20260913',
                      travel_discount=self.travel_discount,detour_weight=self.detour_weight,
                      effect_blend=self.effect_blend,geometric_route_evaluations=self.geometric_route_evaluations)
        self.client.log('mission_cost_summary',**result)
        return result
