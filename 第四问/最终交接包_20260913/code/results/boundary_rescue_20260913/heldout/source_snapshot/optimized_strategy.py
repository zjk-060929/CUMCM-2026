"""Independent optimized Q4 policy; original strategy.py remains the baseline."""
from __future__ import annotations
import math
import time

from client import BudgetStop
from geometry import (add, sub, mul, unit, dist, polygon_distance,
                      clipped_optical_cover, open_route)
from strategy import Policy
from optimized_geometry import (SAFE_RADIUS, ring_route, safe_clear_point,
                                safe_detour_point, polygon_optical_cover,
                                held_karp_open, sample_polygon, possible_emitter_at)

# Frozen using seeds 41000..41059 and 42000..42009 only, before held-out runs.
RECOMMENDED_CONFIG = dict(mesh='ring', detour=400., optical='original',
                          service_route='dp', active_mode='hypothesis',
                          outer_first=False, probe_radius=80.)


class OptimizedPolicy(Policy):
    def __init__(self, client, *, mesh='ring', detour=100., optical='polygon',
                 service_route='dp', active_mode='original', outer_first=False,
                 probe_radius=0.):
        super().__init__(client)
        self.mesh, self.detour, self.optical = mesh, detour, optical
        self.service_route, self.active_mode = service_route, active_mode
        self.route = list(ring_route(outer_first=outer_first)) if mesh == 'ring' else self.route
        self.grid = 'ring25' if mesh == 'ring' else 'triangle'
        self.inserted_clears = 0
        self.route_planning_s = 0.
        self.negative = {}
        self.failed_clear = {}
        self.hypothesis_cache = {}
        self.probe_radius = probe_radius
        self.scan_probed = set()

    def measure(self, p, ch):
        result = super().measure(p, ch)
        if result == 'no_signal' and ch not in self.cleared:
            self.negative.setdefault(ch, []).append(p)
        self.hypothesis_cache.pop(ch, None)
        return result

    def clear(self, p, ch, safe=False):
        success = super().clear(p, ch, safe)
        if not success:
            self.failed_clear.setdefault(ch, []).append(p)
            self.hypothesis_cache.pop(ch, None)
        return success

    def estimate(self, ch):
        t = self.tracks[ch]
        if self.active_mode != 'hypothesis' or t.radius <= SAFE_RADIUS:
            return t.center, t.radius
        if ch not in self.hypothesis_cache:
            positives = [p for p, _ in t.observations]
            samples = [g for g in sample_polygon(t.polygon)
                       if dist(g, (0., 0.)) <= 1800.+1e-7
                       and all(dist(g, p) <= 1500.+1e-7 for p in positives)
                       and all(dist(g, p) > 20. for p in self.failed_clear.get(ch, []))
                       and possible_emitter_at(g, positives, self.negative.get(ch, []))]
            if samples:
                center = (sum(p[0] for p in samples)/len(samples), sum(p[1] for p in samples)/len(samples))
                radius = max(dist(center, p) for p in samples)
                self.hypothesis_cache[ch] = center, radius
            else:
                # Sampling can miss a thin true component. Keep the continuous
                # guaranteed polygon and use its enclosing centre in that case.
                self.hypothesis_cache[ch] = t.center, t.radius
        return self.hypothesis_cache[ch]

    def certified_clear(self, ch):
        t = self.tracks[ch]
        p = safe_clear_point(t.center, t.radius, self.client.position)
        return self.clear(p, ch, safe=True)

    def clear_on_way(self, next_station):
        if self.detour < 0:
            return
        while True:
            current = self.client.position
            choices = []
            for ch in sorted(self.discovered-self.cleared):
                t = self.tracks[ch]
                if t.radius > SAFE_RADIUS:
                    if self.probe_radius and ch not in self.scan_probed:
                        p, radius = self.estimate(ch)
                        extra = dist(current, p)+dist(p, next_station)-dist(current, next_station)
                        if radius <= self.probe_radius and extra+25 <= self.detour:
                            choices.append((extra+25, ch, p))
                    continue
                p = safe_detour_point(t.center, t.radius, current, next_station)
                extra = dist(current, p)+dist(p, next_station)-dist(current, next_station)
                if extra <= self.detour:
                    choices.append((extra, ch, p))
            if not choices:
                return
            _, ch, p = min(choices)
            if self.tracks[ch].radius <= SAFE_RADIUS:
                self.clear(p, ch, safe=True)
                self.inserted_clears += 1
            else:
                # At most one opportunistic uncertain attempt per channel in
                # the scan. If it fails, acquire a bearing at no extra travel.
                self.scan_probed.add(ch)
                self.optimistic_clears += 1
                if self.clear(p, ch):
                    self.inserted_clears += 1
                elif self.measure(p, ch) == 'no_signal':
                    self.no_signal_localization += 1

    def scan(self):
        for station in self.route:
            self.clear_on_way(station)
            unknown = [ch for ch in range(1, 21) if ch not in self.discovered]
            if self.client.channel in unknown:
                unknown.remove(self.client.channel)
                unknown.insert(0, self.client.channel)
            # Known channels are observed only when their region can reach this
            # station and they still need information. Negative readings never
            # remove possible source positions.
            known = [ch for ch, t in self.tracks.items()
                     if not t.cleared and t.radius > SAFE_RADIUS
                     and polygon_distance(station, t.polygon) <= 1500.]
            for ch in unknown:
                self.measure(station, ch)
                if len(self.discovered) == 16:
                    self.stations_visited += 1
                    self.coverage_complete = True
                    self.client.log('discovery_certificate', kind='count_upper_bound',
                                    channels=sorted(self.discovered))
                    return
            for ch in known:
                t = self.tracks[ch]
                if not t.cleared and t.radius > SAFE_RADIUS:
                    if self.measure(station, ch) == 'no_signal':
                        self.no_signal_localization += 1
            self.stations_visited += 1
        self.coverage_complete = True
        self.client.log('discovery_certificate', kind='full_ring_mesh' if self.mesh == 'ring' else 'full_mesh',
                        stations=self.stations_visited,
                        absent_channels=sorted(set(range(1, 21))-self.discovered))
        if len(self.discovered) < 10:
            raise ArithmeticError('Certified scan found fewer than 10 sources')

    def service(self, ch):
        t = self.tracks[ch]
        tried = []
        for iteration in range(4):
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
            offset = min(150., max(45., radius*.4))
            choices = [center, add(center, mul(v, offset)), add(center, mul(v, -offset))]
            if self.active_mode in ('approach', 'hypothesis'):
                toward = sub(last, center)
                norm = dist(last, center)
                toward = mul(toward, 1/norm) if norm else unit(bearing+180)
                # Heuristic only: approach from the side of the last reception.
                # Alternating offsets still allow escape from an unseen side.
                base = add(center, mul(toward, min(100., max(25., radius*.35))))
                choices = [base, add(base, mul(v, offset*.5)), add(base, mul(v, -offset*.5))]
            previous = [p for p, _ in t.observations] + tried
            choices = [p for p in choices if all(dist(p, q) > 5 for q in previous)]
            if not choices:
                break
            p = choices[-1] if iteration % 2 else choices[min(1, len(choices)-1)]
            tried.append(p)
            t.active_queries += 1
            if self.measure(p, ch) == 'no_signal':
                self.no_signal_localization += 1
            if t.cleared:
                return
        if t.radius <= SAFE_RADIUS:
            self.certified_clear(ch)
            return
        self.fallback_channels += 1
        first, bearing = t.observations[0]
        candidates = (polygon_optical_cover(t.polygon) if self.optical == 'polygon'
                      else clipped_optical_cover(first, bearing, t.polygon))
        while candidates:
            k = min(range(len(candidates)), key=lambda i: (dist(candidates[i], self.client.position), i))
            p = candidates.pop(k)
            t.fallback_attempts += 1
            if self.clear(p, ch):
                return
        raise ArithmeticError(f'Certified optical coverage exhausted on channel {ch}')

    def next_service(self):
        pending = sorted(self.discovered-self.cleared)
        centers = [self.estimate(ch)[0] for ch in pending]
        if self.service_route == 'greedy':
            return pending[min(range(len(pending)), key=lambda i: (dist(self.client.position, centers[i]), pending[i]))]
        started = time.perf_counter()
        if self.service_route == 'dp':
            route = held_karp_open(centers, self.client.position)
            ch = pending[route[0]]
        else:
            # Preserve labels when multiple sources have identical centres.
            route = open_route(centers, self.client.position)
            ch = pending[centers.index(route[0])]
        self.route_planning_s += time.perf_counter()-started
        return ch

    def run(self):
        started = time.perf_counter()
        self.client.action('/enter')
        completed, reason = False, 'not_started'
        try:
            self.scan()
            while self.discovered-self.cleared:
                self.service(self.next_service())
            completed = self.coverage_complete and self.discovered == self.cleared
            reason = 'all_sources_cleared_with_certificate'
        except BudgetStop as exc:
            reason = str(exc)
        except ArithmeticError as exc:
            reason = 'model_or_geometry_error: ' + str(exc)
            self.client.log('model_or_geometry_error', detail=str(exc))
        self.client.action('/exit')
        result = dict(completed=completed, stop_reason=reason, grid=self.grid,
                      mesh=self.mesh, detour_m=self.detour, optical=self.optical,
                      service_route=self.service_route, active_mode=self.active_mode,
                      stations_total=len(self.route), stations_visited=self.stations_visited,
                      discovered_channels=sorted(self.discovered), cleared_channels=sorted(self.cleared),
                      discovery_certificate=('count_upper_bound' if len(self.discovered) == 16 else
                          ('full_ring_mesh' if self.mesh == 'ring' else 'full_mesh')) if self.coverage_complete else None,
                      virtual_time_s=self.client.virtual, runtime_s=time.perf_counter()-started,
                      route_planning_s=self.route_planning_s, inserted_clears=self.inserted_clears,
                      scan_probes=len(self.scan_probed), probe_radius=self.probe_radius,
                      fallback_channels=self.fallback_channels,
                      no_signal_localization=self.no_signal_localization,
                      safe_clears=self.safe_clears, optimistic_clears=self.optimistic_clears,
                      active_queries=sum(t.active_queries for t in self.tracks.values()),
                      fallback_attempts=sum(t.fallback_attempts for t in self.tracks.values()))
        self.client.log('policy_summary', **result)
        return result
