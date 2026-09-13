"""Iteration 2: joint task insertion and information-aware station ordering.

All scheduling estimates are heuristics. The certified 21-station set,
continuous localization region and finite optical fallback remain unchanged.
"""
from functools import lru_cache
import math
import time

from geometry import dist, dot, sub, add, mul, unit, polygon_distance
from optimized_geometry import SAFE_RADIUS, safe_clear_point, improve_route
from refined_strategy import RefinedPolicy, REFINED_CONFIG

# Selected using the 40 development scenes only. The radius-gated alternatives
# had weaker minimum-range/cluster results; unrestricted full service caused
# large fallback costs. Frozen before the independent iteration-2 comparison.
PLANNED_CONFIG = dict(joint='all', information=True, scan_steps=2)


@lru_cache(None)
def visibility_masks(stations):
    """Deterministic synthetic prior, independent of simulator seeds and truth.

    Equal weight omni/directional, three radii, eight directional normals,
    192 area-uniform spiral positions. Used ONLY to order visits, never to
    certify absence or trim a continuous feasible set.
    """
    masks = [0]*len(stations)
    index = 0
    normals = [(math.cos(k*math.pi/4), math.sin(k*math.pi/4)) for k in range(8)]
    for j in range(192):
        angle = j*math.pi*(3-math.sqrt(5))
        radius = 1800*math.sqrt((j+.5)/192)
        g = radius*math.cos(angle), radius*math.sin(angle)
        for reach in (1000., 1250., 1500.):
            for normal in normals+[None]*8:
                bit = 1 << index
                for i, p in enumerate(stations):
                    if dist(p, g) <= reach and (normal is None or dot(normal, sub(p, g)) >= 0):
                        masks[i] |= bit
                index += 1
    return tuple(masks), (1 << index)-1


def improve_labels(order, locations, start, station_order=None):
    """Bounded 2-opt. With a station order, only order-preserving swaps apply."""
    order = list(order)
    ranks = {k:i for i, k in enumerate(station_order)} if station_order is not None else {}
    for _ in range(12):
        changed = False
        for i in range(len(order)):
            a = start if i == 0 else locations[order[i-1]]
            for j in range(i+1, len(order)):
                # A reversal containing two distinct stations reverses their order.
                if ranks and sum(k in ranks for k in order[i:j+1]) >= 2:
                    continue
                before, after = dist(a, locations[order[i]]), dist(a, locations[order[j]])
                if j+1 < len(order):
                    before += dist(locations[order[j]], locations[order[j+1]])
                    after += dist(locations[order[i]], locations[order[j+1]])
                if after < before-1e-6:
                    order[i:j+1] = reversed(order[i:j+1])
                    changed = True
        if not changed:
            break
    return order


def insert_tasks(stations, tasks, locations, start):
    """Joint cheapest insertion, including the open tail after the final station.

    Task labels remain distinct even when estimated positions coincide.
    The fixed station order preserves the selected discovery schedule.
    """
    order = list(stations)
    pending = list(tasks)
    while pending:
        best = None
        for task in pending:
            p = locations[task]
            for i in range(len(order)+1):
                a = start if i == 0 else locations[order[i-1]]
                extra = dist(a, p)
                if i < len(order):
                    b = locations[order[i]]
                    extra += dist(p, b)-dist(a, b)
                choice = extra, task, i
                if best is None or choice < best:
                    best = choice
        _, task, i = best
        pending.remove(task)
        order.insert(i, task)
    return improve_labels(order, locations, start, stations)


class PlannedPolicy(RefinedPolicy):
    def __init__(self, client, *, joint='ready', information=False, early_radius=None, scan_steps=0):
        super().__init__(client, **REFINED_CONFIG)
        if joint not in ('off', 'ready', 'all'):
            raise ValueError('Invalid joint mode')
        self.joint, self.information = joint, information
        self.early_radius, self.scan_steps = early_radius, scan_steps
        self.scan_attempted = set()
        self.bounded_deferred = 0
        self.station_points = tuple(self.route)
        self.unvisited = list(range(len(self.route)))
        self.visited_ids = []
        self.joint_services = self.schedule_replans = 0
        self.schedule_planning_s = 0.
        if information:
            self.visibility, self.unseen = visibility_masks(self.station_points)

    def information_order(self):
        order = list(self.unvisited)
        if not self.information or len(order) <= 2:
            return order
        start = self.client.position
        locations = {i:self.station_points[i] for i in order}
        d = len(self.discovered)
        remaining16 = 16-d
        probability16 = 1/(17-max(10, d))
        mean_remaining = (max(10, d)+16)/2-d
        total = self.unseen.bit_count()
        if not total or remaining16 <= 0:
            return order

        def cost(route):
            hidden, score, current = self.unseen, 0., start
            for i in route:
                seen_fraction = 1-hidden.bit_count()/total
                continuing = 1-probability16*seen_fraction**remaining16
                expected_unknown = 20-d-mean_remaining*seen_fraction
                score += continuing*(dist(current, locations[i])/5+6*expected_unknown)
                hidden &= ~self.visibility[i]
                current = locations[i]
            return score

        # Retain the previous schedule as an explicit candidate. Add a greedy
        # information-per-time seed, then perform bounded objective-based 2-opt.
        greedy, rest, current, hidden = [], list(order), start, self.unseen
        while rest:
            i = max(rest, key=lambda k: ((hidden & self.visibility[k]).bit_count()+1)/
                    (dist(current, locations[k])/5+6*(20-d)+1))
            greedy.append(i)
            rest.remove(i)
            current, hidden = locations[i], hidden & ~self.visibility[i]
        best = min((order, greedy), key=cost)
        best_score = cost(best)
        for _ in range(4):
            candidate, value = None, best_score
            for i in range(len(best)):
                for j in range(i+1, len(best)):
                    route = best[:i]+list(reversed(best[i:j+1]))+best[j+1:]
                    score = cost(route)
                    if score < value-1e-6:
                        candidate, value = route, score
            if candidate is None:
                break
            best, best_score = candidate, value
        return best

    def plan(self):
        started = time.perf_counter()
        self.unvisited = self.information_order()
        station_labels = [('station', i) for i in self.unvisited]
        locations = {k:self.station_points[k[1]] for k in station_labels}
        tasks = []
        if self.joint != 'off':
            for ch in sorted(self.discovered-self.cleared):
                t = self.tracks[ch]
                if t.radius <= SAFE_RADIUS:
                    center, radius = t.center, t.radius
                else:
                    center, radius = self.estimate(ch)
                    if self.joint == 'ready' and (radius > self.probe_radius or ch in self.scan_probed):
                        continue
                    if self.early_radius is not None and t.radius > self.early_radius:
                        continue
                    if self.scan_steps and ch in self.scan_attempted:
                        continue
                label = ('task', ch)
                tasks.append(label)
                locations[label] = center
        order = insert_tasks(station_labels, tasks, locations, self.client.position)
        self.schedule_replans += 1
        self.schedule_planning_s += time.perf_counter()-started
        self.client.log('joint_plan', joint=self.joint, information=self.information,
                        order=order, discovered=len(self.discovered), pending=len(tasks))
        return order

    def bounded_scan_service(self, ch):
        """Try at most scan_steps bearings once per channel during discovery.

        Do not enter the potentially long optical fallback before discovery
        ends. An unresolved channel waits for scheduled measurements and the
        unchanged post-scan guaranteed service routine.
        """
        self.scan_attempted.add(ch)
        t, tried = self.tracks[ch], []
        for iteration in range(self.scan_steps):
            if t.cleared:
                return
            if t.radius <= SAFE_RADIUS:
                self.certified_clear(ch)
                return
            center, radius = self.estimate(ch)
            if radius <= 80 and all(dist(center, p) > 5 for p in tried):
                tried.append(center)
                self.optimistic_clears += 1
                if self.clear(center, ch):
                    return
                center, radius = self.estimate(ch)
            last, bearing = t.observations[-1]
            v = unit(bearing+90)
            norm = dist(last, center)
            toward = mul(sub(last, center), 1/norm) if norm else unit(bearing+180)
            base = add(center, mul(toward, min(100., max(25., radius*.35))))
            offset = min(150., max(45., radius*.4))
            choices = [base, add(base, mul(v, offset*.5)), add(base, mul(v, -offset*.5))]
            observed = [p for p, _ in t.observations]+self.negative.get(ch, [])+tried
            choices = [p for p in choices if all(dist(p, q) > 5 for q in observed)]
            if not choices:
                break
            p = choices[-1] if iteration % 2 else choices[min(1, len(choices)-1)]
            tried.append(p)
            t.active_queries += 1
            if self.measure(p, ch) == 'no_signal':
                self.no_signal_localization += 1
        if not t.cleared and t.radius <= SAFE_RADIUS:
            self.certified_clear(ch)
        if not t.cleared:
            self.bounded_deferred += 1
            self.client.log('bounded_service_deferred', channel=ch, radius=t.radius)

    def scan_station(self, station_id):
        station = self.station_points[station_id]
        unknown = [ch for ch in range(1, 21) if ch not in self.discovered]
        if self.client.channel in unknown:
            unknown.remove(self.client.channel)
            unknown.insert(0, self.client.channel)
        known = [ch for ch, t in self.tracks.items() if not t.cleared and t.radius > SAFE_RADIUS
                 and polygon_distance(station, t.polygon) <= 1500.]
        self.in_station_scan = True
        try:
            for ch in unknown:
                self.measure(station, ch)
                if len(self.discovered) == 16:
                    break
            if len(self.discovered) < 16:
                for ch in known:
                    if not self.tracks[ch].cleared and self.tracks[ch].radius > SAFE_RADIUS:
                        if self.measure(station, ch) == 'no_signal':
                            self.no_signal_localization += 1
        finally:
            self.in_station_scan = False
        self.unvisited.remove(station_id)
        self.visited_ids.append(station_id)
        self.stations_visited += 1
        if self.information:
            self.unseen &= ~self.visibility[station_id]
        self.client.log('station_complete', station_id=station_id, position=station,
                        discovered=len(self.discovered), cleared=len(self.cleared),
                        virtual_time_s=self.client.virtual)

    def scan(self):
        while self.unvisited and len(self.discovered) < 16:
            kind, item = self.plan()[0]
            if kind == 'task':
                if self.joint == 'all' and self.scan_steps and self.tracks[item].radius > SAFE_RADIUS:
                    self.bounded_scan_service(item)
                elif self.joint == 'all':
                    self.service(item)
                elif self.tracks[item].radius <= SAFE_RADIUS:
                    self.certified_clear(item)
                else:
                    self.scan_probed.add(item)
                    self.optimistic_clears += 1
                    center, _ = self.estimate(item)
                    self.clear(center, item)
                self.joint_services += 1
                if item in self.cleared:
                    self.inserted_clears += 1
                continue
            self.remaining_route = [self.station_points[i] for i in self.unvisited]
            if self.joint == 'off':
                self.clear_on_way(self.station_points[item])
            self.scan_station(item)
        self.coverage_complete = True
        kind = 'count_upper_bound' if len(self.discovered) == 16 else 'convex_cell_cover'
        self.client.log('discovery_certificate', kind=kind, stations=self.stations_visited,
                        absent_channels=sorted(set(range(1, 21))-self.discovered))
        if len(self.discovered) < 10:
            raise ArithmeticError('Certified scan found fewer than 10 sources')

    def run(self):
        result = super().run()
        result.update(policy_version='q4_planning_iteration2_20260913', joint_mode=self.joint,
                      information_order=self.information, joint_services=self.joint_services,
                      early_radius_m=self.early_radius, scan_steps=self.scan_steps,
                      bounded_deferred=self.bounded_deferred,
                      schedule_replans=self.schedule_replans, schedule_planning_s=self.schedule_planning_s,
                      visited_station_ids=self.visited_ids)
        self.client.log('planned_policy_summary', **result)
        return result
