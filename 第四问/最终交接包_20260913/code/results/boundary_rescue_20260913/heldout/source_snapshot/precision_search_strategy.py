"""1024-sided localization and bounded, expanded joint-route search.

Geometry and search are independently configurable for paired ablations.
The policy sees observations only; simulator truth is reserved for post-run audit.
"""
import random
import time

from guarded_route_strategy import GuardedRoutePolicy, BudgetedRouteModel
from heuristic_search import optimize, descent, neighbour
from mission_cost_strategy import GeometricRouteModel, MissionCostModel
from optimized_geometry import SAFE_RADIUS
from planned_strategy import insert_tasks


SEARCH = {
    'standard': dict(depth=3, width=4, branch=6, starts=0, rounds=0, probes=0, alns=0),
    'expanded': dict(depth=6, width=16, branch=10, starts=4, rounds=4, probes=24, alns=48),
    'intensive': dict(depth=8, width=32, branch=16, starts=8, rounds=6, probes=32, alns=96),
}
VARIANTS = {
    'baseline128': dict(circle_sides=128, search_level='standard', joint_cost=False),
    'precision1024': dict(circle_sides=1024, search_level='standard', joint_cost=False),
    'search1024': dict(circle_sides=1024, search_level='expanded', joint_cost=False),
    'joint_standard': dict(circle_sides=1024, search_level='standard', joint_cost=True),
    'joint_expanded': dict(circle_sides=1024, search_level='expanded', joint_cost=True),
    'joint_intensive': dict(circle_sides=1024, search_level='intensive', joint_cost=True),
}


def expanded_optimize(model, initial, seed, warm, level):
    """Always retain the ordinary search incumbent under the SAME objective."""
    best, stats = optimize(model, initial, method='beam', seed=seed, warm=warm)
    value = model.cost(best)
    standard_value = value
    config = SEARCH[level]
    if level != 'standard':
        candidate, _ = optimize(model, best, method='beam', seed=seed+100003,
                                warm=warm, beam_depth=config['depth'],
                                beam_width=config['width'], beam_branch=config['branch'])
        score = model.cost(candidate)
        if score < value:
            best, value = candidate, score
        rng = random.Random(seed+200003)
        for i in range(config['starts']):
            if i == 0:
                candidate = model.repair([], range(model.n), 3)
            else:
                candidate = list(best)
                for _ in range(2+i):
                    candidate = neighbour(candidate, rng, rng.randrange(4))
                # Restore the discovery-before-deferred-service precedence.
                candidate = [k for k in candidate if k not in model.blocked]+[
                    k for k in candidate if k in model.blocked]
            candidate = descent(model, candidate, rng, rounds=config['rounds'], probes=config['probes'])
            score = model.cost(candidate)
            if score < value:
                best, value = candidate, score
        candidate, extra = optimize(model, best, method='alns', iterations=config['alns'],
                                    seed=seed+300007, warm=best)
        score = model.cost(candidate)
        if score < value:
            best, value = candidate, score
        stats['accepted_worse'] += extra['accepted_worse']
    assert value <= standard_value+1e-6
    stats.update(final_cost=value, cost_evaluations=model.calls, search_level=level,
                 standard_search_cost=standard_value, expanded_proxy_saving_s=standard_value-value,
                 search_config=config)
    return best, stats


class PrecisionSearchPolicy(GuardedRoutePolicy):
    def __init__(self, client, *, circle_sides=1024, search_level='expanded', joint_cost=True):
        if circle_sides not in (128, 1024) or search_level not in SEARCH:
            raise ValueError('Unsupported precision/search configuration')
        super().__init__(client, station_margin=None if joint_cost else 500., skip_gain=.08)
        self.circle_sides = circle_sides
        self.search_level, self.joint_cost = search_level, joint_cost
        self.geometric_route_evaluations = 0
        self.expanded_proxy_saving_s = 0.
        self.max_replan_runtime_s = 0.

    def plan(self):
        started = time.perf_counter()
        stations = [('station', i) for i in self.unvisited]
        tasks = [('task', ch) for ch in sorted(self.discovered-self.cleared)]
        labels = stations+tasks
        positions = {k:self.station_points[k[1]] for k in stations}
        blocked = set()
        for k in tasks:
            track = self.tracks[k[1]]
            positions[k] = track.center if track.radius<=SAFE_RADIUS else self.estimate(k[1])[0]
            if track.radius>SAFE_RADIUS and k[1] in self.scan_attempted:
                blocked.add(k)
        initial = insert_tasks(stations, [k for k in tasks if k not in blocked], positions,
                               self.client.position)+sorted(blocked)
        index = {k:i for i,k in enumerate(labels)}
        initial_ids = [index[k] for k in initial]
        warm = [index[k] for k in self.last_plan if k in index and k not in blocked]
        warm += [index[k] for k in self.last_plan if k in blocked]
        effects = {}
        for k in tasks:
            predictions = self.predicted_station_effects(k[1],positions[k])
            for station in stations:
                radius, possible = predictions[station[1]]
                if self.joint_cost:
                    radius += .5*max(0.,self.tracks[k[1]].radius-radius)
                effects.setdefault(index[station], []).append((index[k],radius,possible))
        common = (labels, [positions[k] for k in labels], self.client.position,
                  {index[k]:self.tracks[k[1]].radius for k in tasks}, effects,
                  {index[k] for k in blocked})
        options = dict(uncertainty_weight=self.uncertainty_weight,
                       unknown_channels=20-len(self.discovered),
                       discovery=({index[k]:self.visibility[k[1]] for k in stations},self.unseen),
                       station_origin=self.last_station_position,
                       station_prefix_m=self.station_prefix_distance,
                       station_limit_m=self.station_limit, skip_gain=self.skip_gain)
        geometric_order = None
        if self.joint_cost:
            geometry = GeometricRouteModel(*common)
            geometric_order, gs = optimize(geometry,initial_ids,method='vnd',
                                           seed=9817+self.schedule_replans,warm=warm)
            self.geometric_route_evaluations += gs['cost_evaluations']
            model = MissionCostModel(*common, **options, travel_discount=0., detour_weight=.5,
                                     geometric_reference_s=geometry.path_seconds(geometric_order))
        else:
            model = BudgetedRouteModel(*common, **options)
            assert model.station_path_s(initial_ids)<=model.station_limit_s+1e-6
        order, stats = expanded_optimize(model, initial_ids, 8137+self.schedule_replans,
                                         warm, self.search_level)
        if geometric_order is not None and model.cost(geometric_order)<stats['final_cost']:
            order = geometric_order
            stats['final_cost'] = model.cost(order)
        potential = model.station_path_s(order)*5
        assert self.station_limit is None or potential<=self.station_limit+1e-5
        self.maximum_accepted_station_potential = max(self.maximum_accepted_station_potential,potential)
        self.rejected_route_candidates += model.rejected_orders
        self.expanded_proxy_saving_s += stats['expanded_proxy_saving_s']
        result = [labels[i] for i in order]
        self.unvisited = [k[1] for k in result if k[0]=='station']
        self.last_plan = result
        self.schedule_replans += 1
        elapsed = time.perf_counter()-started
        self.schedule_planning_s += elapsed
        self.max_replan_runtime_s = max(self.max_replan_runtime_s,elapsed)
        self.meta_evaluations += model.calls
        self.meta_accepted_worse += stats['accepted_worse']
        self.meta_proxy_savings += stats['initial_cost']-stats['final_cost']
        self.client.log('precision_search_plan', order=result, station_potential_m=potential,
                        circle_sides=self.circle_sides, joint_cost=self.joint_cost, **stats)
        return result

    def run(self):
        result = super().run()
        result.update(policy_version='q4_precision_search_20260913', circle_sides=self.circle_sides,
                      search_level=self.search_level, search_config=SEARCH[self.search_level],
                      joint_cost=self.joint_cost, max_replan_runtime_s=self.max_replan_runtime_s,
                      expanded_proxy_saving_s=self.expanded_proxy_saving_s,
                      geometric_route_evaluations=self.geometric_route_evaluations)
        return result
