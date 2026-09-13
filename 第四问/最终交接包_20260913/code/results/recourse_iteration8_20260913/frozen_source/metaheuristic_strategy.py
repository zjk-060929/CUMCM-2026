"""Research-based heuristic scheduling above the certified Q4 geometry."""
import math
import time

from geometry import dist, dot, sub, cross, unit, polygon_distance, ALPHA
from optimized_geometry import SAFE_RADIUS
from planned_strategy import PlannedPolicy, insert_tasks
from strategy import Track
from heuristic_search import RouteModel, optimize

# Selected only on the 40 development scenes: lowest ordinary mean among
# candidates with no mean regression in any of the four stress groups.
METAHEURISTIC_CONFIG = dict(method='beam', uncertainty_weight=.35, iterations=48,
                            local_sweep=False, discovery_model=False)


class MetaheuristicPolicy(PlannedPolicy):
    def __init__(self, client, *, method='alns', uncertainty_weight=.35, iterations=48,
                 local_sweep=False, discovery_model=False):
        # Joint remaining-task surrogate; retain the certified discovery rule:
        # visit all stations unless all 16 possible sources have been found.
        super().__init__(client, joint='all', information=discovery_model, scan_steps=2)
        self.discovery_model = discovery_model
        self.method,self.uncertainty_weight,self.iterations = method,uncertainty_weight,iterations
        self.local_sweep = local_sweep
        self.last_plan = []
        self.effect_cache = {}
        self.meta_evaluations = self.meta_accepted_worse = 0
        self.meta_proxy_savings = 0.
        self.local_sweep_points = []
        self.local_sweep_busy = False
        self.local_clear_attempts = self.local_discoveries = 0

    def predicted_station_effects(self, ch, center):
        t = self.tracks[ch]
        key = (ch,len(t.observations),len(self.negative.get(ch, [])),len(self.failed_clear.get(ch, [])))
        if key in self.effect_cache:
            return self.effect_cache[key]
        positives = [p for p,_ in t.observations]
        negatives = self.negative.get(ch, [])
        required = max([1000.]+[dist(center,p) for p in positives])
        hypotheses = []
        if required <= 1500.+1e-6:
            for reach in (required,(required+1500)/2,1500.):
                for normal in [unit(22.5*k) for k in range(16)]+[None]*16:
                    if normal is not None and any(dot(normal,sub(p,center)) < -1e-6 for p in positives):
                        continue
                    if any(dist(q,center) <= reach and (normal is None or dot(normal,sub(q,center)) >= -1e-6) for q in negatives):
                        continue
                    hypotheses.append((reach,normal))
        effects = {}
        for i,q in enumerate(self.station_points):
            possible = polygon_distance(q,t.polygon) <= 1500.
            predicted = t.radius
            if possible and t.observations:
                reception = (sum(dist(q,center) <= reach and (normal is None or dot(normal,sub(q,center)) >= 0)
                                 for reach,normal in hypotheses)/len(hypotheses)) if hypotheses else .35
                best = t.radius
                for first,bearing in t.observations:
                    delta = sub(center,q)
                    length = dist(center,q)
                    direction = (delta[0]/length,delta[1]/length) if length else unit(bearing+90)
                    sine = abs(cross(unit(bearing),direction))
                    radius = max(6., ALPHA*(dist(first,center)+length)/max(.08,sine))
                    best = min(best,radius)
                predicted = (1-reception)*t.radius+reception*best
            effects[i] = (predicted,possible)
        self.effect_cache[key] = effects
        return effects

    def plan(self):
        started = time.perf_counter()
        stations = [('station',i) for i in self.unvisited]
        tasks = [('task',ch) for ch in sorted(self.discovered-self.cleared)]
        labels = stations+tasks
        positions = {k:self.station_points[k[1]] for k in stations}
        blocked = set()
        for k in tasks:
            t = self.tracks[k[1]]
            positions[k] = t.center if t.radius <= SAFE_RADIUS else self.estimate(k[1])[0]
            if t.radius > SAFE_RADIUS and k[1] in self.scan_attempted:
                blocked.add(k)
        # Deferred targets stay in the plan's tail instead of disappearing from
        # the estimated cost of finishing the mission.
        active = [k for k in tasks if k not in blocked]
        initial = insert_tasks(stations,active,positions,self.client.position)
        initial += sorted(blocked)
        index = {k:i for i,k in enumerate(labels)}
        effects = {}
        if self.uncertainty_weight:
            for k in tasks:
                future = self.predicted_station_effects(k[1],positions[k])
                for station in stations:
                    radius,possible = future[station[1]]
                    effects.setdefault(index[station],[]).append((index[k],radius,possible))
        model = RouteModel(labels,[positions[k] for k in labels],self.client.position,
                           {index[k]:self.tracks[k[1]].radius for k in tasks},effects,
                           {index[k] for k in blocked},self.uncertainty_weight,
                           unknown_channels=20-len(self.discovered),
                           discovery=({index[k]:self.visibility[k[1]] for k in stations},self.unseen)
                           if self.discovery_model else None)
        # A warm start is repaired if feedback makes a formerly active task deferred.
        warm = [index[k] for k in self.last_plan if k in index and k not in blocked]
        warm += [index[k] for k in self.last_plan if k in blocked]
        order,stats = optimize(model,[index[k] for k in initial],method=self.method,
                               iterations=self.iterations,seed=8137+self.schedule_replans,warm=warm)
        result = [labels[i] for i in order]
        self.unvisited = [k[1] for k in result if k[0] == 'station']
        self.last_plan = result
        self.schedule_replans += 1
        self.schedule_planning_s += time.perf_counter()-started
        self.meta_evaluations += stats['cost_evaluations']
        self.meta_accepted_worse += stats['accepted_worse']
        self.meta_proxy_savings += stats['initial_cost']-stats['final_cost']
        self.client.log('metaheuristic_plan',order=result,blocked=sorted(blocked),**stats)
        return result

    def clear(self,p,ch,safe=False):
        success = super().clear(p,ch,safe)
        if success:
            self.opportunistic_local_sweep(self.client.position)
        return success

    def opportunistic_local_sweep(self,p):
        if not self.local_sweep or self.local_sweep_busy or len(self.local_sweep_points) >= 2:
            return
        if any(dist(p,q) < 25 for q in self.local_sweep_points):
            return
        # Trigger only from observed local evidence, never from a scenario label.
        nearby = [ch for ch,t in self.tracks.items() if t.observations and t.radius <= 80 and dist(t.center,p) <= 100]
        if len(nearby) < 3:
            return
        self.local_sweep_busy = True
        self.local_sweep_points.append(p)
        try:
            for ch in sorted(set(range(1,21))-self.discovered):
                reply = self.client.action('/clear',p,ch)
                self.local_clear_attempts += 1
                if reply['clear_result'] == 'success':
                    self.discovered.add(ch)
                    self.cleared.add(ch)
                    self.tracks[ch] = Track(cleared=True)
                    self.local_discoveries += 1
                    if len(self.discovered) == 16:
                        break
                else:
                    self.failed_clear.setdefault(ch,[]).append(p)
                    self.hypothesis_cache.pop(ch,None)
        finally:
            self.local_sweep_busy = False

    def run(self):
        result = super().run()
        result.update(policy_version='q4_metaheuristic_20260913',metaheuristic=self.method,
                      uncertainty_weight=self.uncertainty_weight,meta_iterations=self.iterations,
                      meta_evaluations=self.meta_evaluations,meta_accepted_worse=self.meta_accepted_worse,
                      meta_proxy_savings=self.meta_proxy_savings,local_sweep=self.local_sweep,
                      local_clear_attempts=self.local_clear_attempts,local_discoveries=self.local_discoveries,
                      discovery_model=self.discovery_model,
                      deferred_tasks_in_plan=True)
        self.client.log('metaheuristic_policy_summary',**result)
        return result
