"""Bounded Q3-inspired Q4 refinement, preserving the 2026-09-12 baseline."""
import math
from geometry import (dist, dot, sub, add, mul, unit, polygon_distance, enclosing_circle, ALPHA)
from strategy import Track
from optimized_strategy import OptimizedPolicy, RECOMMENDED_CONFIG
from optimized_geometry import SAFE_RADIUS, safe_detour_point
from refined_geometry import source_domain, incorporate_fine, certified_sparse_route

# Selected on 24 development mixed cases and 4 cases in each stress group.
# Frozen before the separate 200 + 4*20 paired held-out scenarios.
REFINED_CONFIG = dict(fine=True, forward=True, piggyback=True, reactive=True,
                      sparse=True, detour=800., active_insertion=False)


class RefinedPolicy(OptimizedPolicy):
    def __init__(self, client, *, fine=False, forward=False, piggyback=False,
                 reactive=False, sparse=False, detour=400., piggyback_limit=2,
                 active_insertion=False, circle_sides=128):
        super().__init__(client, **{**RECOMMENDED_CONFIG, 'detour': detour})
        self.fine, self.forward = fine, forward
        if not isinstance(circle_sides, int) or circle_sides < 4:
            raise ValueError('circle_sides must be an integer >= 4')
        self.circle_sides = circle_sides
        self.use_piggyback, self.reactive = piggyback, reactive
        self.sparse, self.piggyback_limit = sparse, piggyback_limit
        self.remaining_route = []
        self.in_station_scan = False
        self.piggyback_busy = self.reactive_busy = False
        self.piggyback_queries = self.reactive_queries = 0
        self.reactive_per_channel = {}
        self.forward_deferred = 0
        self.active_insertion = active_insertion
        self.scan_measured = set()
        if sparse:
            self.route = list(certified_sparse_route(8, 12, 999., 1864., 0.))
            self.grid, self.mesh = 'certified_sparse21', 'sparse21'

    def measure(self, p, ch):
        if not self.fine:
            result = super().measure(p, ch)
        else:
            reply = self.client.action('/measure', p, ch)
            result = reply['measure_result']
            if result != 'no_signal':
                self.discovered.add(ch)
                if ch not in self.tracks:
                    self.tracks[ch] = Track(polygon=list(source_domain(self.circle_sides)))
                t = self.tracks[ch]
                if result == 'near':
                    self.clear(p, ch, safe=True)
                else:
                    t.observations.append((p, reply['svd_deg']))
                    t.polygon = incorporate_fine(t.polygon, p, reply['svd_deg'], self.circle_sides)
                    t.center, t.radius = enclosing_circle(t.polygon)
            elif ch not in self.cleared:
                self.negative.setdefault(ch, []).append(p)
            self.hypothesis_cache.pop(ch, None)
        if not self.in_station_scan:
            self.piggyback(p, exclude=ch)
        return result

    def clear(self, p, ch, safe=False):
        success = super().clear(p, ch, safe)
        if (not success and self.reactive and not self.reactive_busy
                and self.reactive_per_channel.get(ch, 0) < 2):
            t = self.tracks[ch]
            observed = [q for q, _ in t.observations] + self.negative.get(ch, [])
            if all(dist(p, q) > 1e-5 for q in observed):
                self.reactive_per_channel[ch] = self.reactive_per_channel.get(ch, 0)+1
                self.reactive_queries += 1
                self.reactive_busy = True
                try:
                    self.measure(p, ch)
                    if t.cleared:
                        success = True
                    elif t.radius <= SAFE_RADIUS:
                        success = self.certified_clear(ch)
                finally:
                    self.reactive_busy = False
        if not self.in_station_scan:
            self.piggyback(self.client.position, exclude=ch)
        return success

    def piggyback(self, p, exclude=None):
        if not self.use_piggyback or self.piggyback_busy:
            return
        self.piggyback_busy = True
        try:
            options = []
            for ch in sorted(self.discovered-self.cleared):
                t = self.tracks[ch]
                if ch == exclude or t.radius <= SAFE_RADIUS or not t.observations:
                    continue
                observed = [q for q, _ in t.observations] + self.negative.get(ch, [])
                if any(dist(p, q) <= 5. for q in observed):
                    continue
                # The Q3 inequality proves range feasibility only. Directional
                # visibility is still unknown and checked by the real reading.
                eligible = False
                for first, bearing in t.observations:
                    delta = sub(p, first)
                    length2 = dot(delta, delta)
                    if not 80.**2 <= length2 <= 1000.**2:
                        continue
                    along, across = dot(delta, unit(bearing)), dot(delta, unit(bearing+90))
                    eta = along*math.cos(ALPHA)-abs(across)*math.sin(ALPHA)
                    beta = abs(math.atan2(across, along))
                    if length2 <= 2000*eta and beta >= math.radians(10):
                        eligible = True
                        break
                if not eligible or polygon_distance(p, t.polygon) > 1500.:
                    continue
                center, radius = self.estimate(ch)
                options.append((dist(p, center)/max(radius, 20.), ch))
            for _, ch in sorted(options)[:self.piggyback_limit]:
                if ch not in self.cleared:
                    self.piggyback_queries += 1
                    self.measure(p, ch)
        finally:
            self.piggyback_busy = False

    def clear_on_way(self, next_station):
        if not self.forward:
            return super().clear_on_way(next_station)
        while True:
            current = self.client.position
            choices = []
            for ch in sorted(self.discovered-self.cleared):
                t = self.tracks[ch]
                safe = t.radius <= SAFE_RADIUS
                if safe:
                    center, radius = t.center, t.radius
                    p = safe_detour_point(center, radius, current, next_station)
                    penalty = 0.
                else:
                    if ch in self.scan_probed:
                        continue
                    center, radius = self.estimate(ch)
                    if radius > self.probe_radius:
                        if self.active_insertion and ch not in self.scan_measured:
                            first, bearing = t.observations[0]
                            observed = [q for q, _ in t.observations] + self.negative.get(ch, [])
                            # Defer when a future scheduled station can provide a
                            # geometrically useful range-safe second reading.
                            # Its visibility remains uncertain for a directional source.
                            free_future = False
                            for q in self.remaining_route:
                                d = sub(q, first)
                                a, b = dot(d, unit(bearing)), dot(d, unit(bearing+90))
                                eta = a*math.cos(ALPHA)-abs(b)*math.sin(ALPHA)
                                if (80.**2 <= dot(d, d) <= 1000.**2 and dot(d, d) <= 2000*eta
                                        and abs(math.atan2(b, a)) >= math.radians(10)
                                        and all(dist(q, old) > 5 for old in observed)):
                                    free_future = True
                                    break
                            if not free_future:
                                for sign in (-1, 1):
                                    q = add(first, add(mul(unit(bearing), 535.), mul(unit(bearing+90), sign*105.)))
                                    if any(dist(q, old) <= 5 for old in observed):
                                        continue
                                    now = dist(current, q)+dist(q, next_station)-dist(current, next_station)
                                    later = min((dist(a, q)+dist(q, b)-dist(a, b)
                                                 for a, b in zip(self.remaining_route, self.remaining_route[1:])),
                                                default=math.inf)
                                    if now+25 <= self.detour and now <= later+1e-6:
                                        choices.append((now+25, ch, q, None))
                        continue
                    p, penalty = center, 25.
                extra = dist(current, p)+dist(p, next_station)-dist(current, next_station)
                if extra+penalty > self.detour:
                    continue
                future = math.inf
                for a, b in zip(self.remaining_route, self.remaining_route[1:]):
                    q = safe_detour_point(center, radius, a, b) if safe else center
                    future = min(future, dist(a, q)+dist(q, b)-dist(a, b))
                if extra > future+1e-6:
                    self.forward_deferred += 1
                    continue
                choices.append((extra+penalty, ch, p, safe))
            if not choices:
                return
            _, ch, p, safe = min(choices)
            if safe is None:
                self.scan_measured.add(ch)
                if self.measure(p, ch) == 'no_signal':
                    self.no_signal_localization += 1
            elif safe:
                self.clear(p, ch, safe=True)
                self.inserted_clears += 1
            else:
                self.scan_probed.add(ch)
                self.optimistic_clears += 1
                if self.clear(p, ch):
                    self.inserted_clears += 1
                elif not self.reactive or self.reactive_per_channel.get(ch, 0) == 0:
                    if self.measure(p, ch) == 'no_signal':
                        self.no_signal_localization += 1

    def scan(self):
        for index, station in enumerate(self.route):
            self.remaining_route = self.route[index:]
            self.clear_on_way(station)
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
                        self.stations_visited += 1
                        self.coverage_complete = True
                        self.client.log('discovery_certificate', kind='count_upper_bound',
                                        channels=sorted(self.discovered))
                        return
                for ch in known:
                    if not self.tracks[ch].cleared and self.tracks[ch].radius > SAFE_RADIUS:
                        if self.measure(station, ch) == 'no_signal':
                            self.no_signal_localization += 1
            finally:
                self.in_station_scan = False
            self.stations_visited += 1
        self.coverage_complete = True
        self.client.log('discovery_certificate', kind='convex_cell_cover' if self.sparse else 'full_ring_mesh',
                        stations=self.stations_visited,
                        absent_channels=sorted(set(range(1, 21))-self.discovered))
        if len(self.discovered) < 10:
            raise ArithmeticError('Certified discovery found fewer than 10 sources')

    def run(self):
        result = super().run()
        if self.coverage_complete and len(self.discovered) != 16 and self.sparse:
            result['discovery_certificate'] = 'convex_cell_cover'
        result.update(policy_version='q4_refinement_20260913', fine_circles=self.fine,
                      forward_insertion=self.forward, piggyback_enabled=self.use_piggyback,
                      reactive_enabled=self.reactive, piggyback_queries=self.piggyback_queries,
                      reactive_queries=self.reactive_queries, forward_deferred=self.forward_deferred,
                      active_insertion=self.active_insertion, scan_second_measurements=len(self.scan_measured))
        self.client.log('refined_policy_summary', **result)
        return result
