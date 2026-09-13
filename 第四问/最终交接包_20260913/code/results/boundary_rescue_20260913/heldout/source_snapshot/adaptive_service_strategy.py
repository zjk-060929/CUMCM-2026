"""Iteration 4: observation-gated endpoint probes and geometric service planning.

All ranking models are heuristics. Localization keeps every full angular bound;
no nearby-noise averaging, simulated truth, or field scale enters the policy.
"""
import math

from correlation_aware_strategy import CorrelationAwarePolicy
from geometry import dist, sub, add, mul, unit, clip, dot, ALPHA_DEG, clipped_optical_cover
from optimized_geometry import SAFE_RADIUS
from planned_strategy import visibility_masks
from metaheuristic_strategy import MetaheuristicPolicy


VARIANTS = {
    'current': None,
    'reference': dict(order='information', endpoint_gate='off', local=False),
    'restore': dict(order='information', endpoint_gate='legacy'),
    'gated': dict(order='information', endpoint_gate='evidence'),
    'gated_fixed': dict(order='fixed', endpoint_gate='evidence'),
    'gated_beam': dict(order='beam', endpoint_gate='evidence'),
    'lookahead': dict(order='information', endpoint_gate='evidence', geometric_service=True),
    'lookahead_beam': dict(order='beam', endpoint_gate='evidence', geometric_service=True),
}


def predicted_radius(poly, p, g):
    """Worst enclosing-box radius after representative bearings at +/-1.005 deg.

    Used solely for ranking. Does not modify poly or infer a reduced error bound.
    We allow full bias extremes rather than draw independent noise samples.
    """
    bearing = math.degrees(math.atan2(g[1]-p[1], g[0]-p[0]))
    worst = 0.
    for error in (-ALPHA_DEG, 0., ALPHA_DEG):
        angle = bearing+error
        lo, hi = unit(angle-ALPHA_DEG), unit(angle+ALPHA_DEG)
        candidate = poly
        for normal in ((lo[1], -lo[0]), (-hi[1], hi[0])):
            candidate = clip(candidate, normal, dot(normal, p))
        if not candidate:
            return math.inf
        width = max(q[0] for q in candidate)-min(q[0] for q in candidate)
        height = max(q[1] for q in candidate)-min(q[1] for q in candidate)
        worst = max(worst, math.hypot(width, height)/2)
    return worst


class AdaptiveServicePolicy(CorrelationAwarePolicy):
    def __init__(self, client, *, order='information', endpoint_gate='evidence',
                 local=True, geometric_service=False):
        super().__init__(client)
        if order not in ('information', 'fixed', 'beam') or endpoint_gate not in ('off', 'legacy', 'evidence'):
            raise ValueError('Invalid iteration-4 option')
        self.order_mode, self.endpoint_gate = order, endpoint_gate
        self.local = local
        self.endpoint = endpoint_gate != 'off'
        self.geometric_service = geometric_service
        self.information = order != 'fixed'
        self.stable = not self.information
        if self.information:
            self.visibility, self.unseen = visibility_masks(self.station_points)
        self.interior_stations = set()
        self.interior_detected = set()
        self.endpoint_evidence_seen = False
        self.endpoint_gate_skips = 0
        self.geometric_candidates = self.actual_geometric_queries = 0
        # Reuse the existing bounded beam planner, leaving its files unchanged.
        self.discovery_model = self.information
        self.method, self.uncertainty_weight, self.iterations = 'beam', .35, 48
        self.last_plan, self.effect_cache = [], {}
        self.meta_evaluations = self.meta_accepted_worse = self.meta_proxy_savings = 0

    def measure(self, p, ch):
        result = super().measure(p, ch)
        if result in ('direction', 'near') and dist(p, (0., 0.)) <= 1800.:
            self.interior_detected.add(ch)
        return result

    def scan_station(self, station_id):
        super().scan_station(station_id)
        if dist(self.station_points[station_id], (0., 0.)) <= 1800.:
            self.interior_stations.add(station_id)

    def try_near_endpoint(self, ch):
        if self.endpoint_gate == 'off':
            return
        if self.endpoint_gate == 'evidence':
            # Weak discovery inside the disk is observable evidence for an
            # outward-facing population; never read a scenario label or truth.
            sparse_inside = len(self.interior_stations) >= 3 and len(self.interior_detected) <= 2
            self.endpoint_evidence_seen |= sparse_inside
            # A compact estimate may also justify a bounded local optical probe.
            compact = ch in self.tracks and self.estimate(ch)[1] <= 40.
            if not (sparse_inside or compact):
                self.endpoint_gate_skips += 1
                return
        return super().try_near_endpoint(ch)

    def predicted_station_effects(self, ch, center):
        return MetaheuristicPolicy.predicted_station_effects(self, ch, center)

    def plan(self):
        if self.order_mode == 'beam':
            return MetaheuristicPolicy.plan(self)
        return super().plan()

    def active_point(self, ch):
        t = self.tracks[ch]
        center, radius = self.estimate(ch)
        last, bearing = t.observations[-1]
        length = dist(last, center)
        toward = mul(sub(last, center), 1/length) if length else unit(bearing+180)
        lateral = (-toward[1], toward[0])
        back = min(100., max(25., radius*.35))
        offset = min(150., max(45., radius*.4))
        points = []
        for rear in (back, max(60., back)):
            base = add(center, mul(toward, rear))
            for side in (0., offset/2, -offset/2, 75., -75.):
                points.append(add(base, mul(lateral, side)))
        candidates = []
        for p in dict.fromkeys(points):
            if self.redundant_reason(p, ch):
                continue
            if any(dist(p, q) <= 30. for q in self.negative.get(ch, [])):
                continue
            residual = predicted_radius(t.polygon, p, center)
            # Metres-equivalent surrogate: travel to observation, approach to
            # representative target, and penalty for remaining uncertainty.
            cost = dist(self.client.position, p)+dist(p, center)+3*max(0., residual-SAFE_RADIUS)
            candidates.append((cost, p))
            self.geometric_candidates += 1
        return min(candidates)[1] if candidates else None

    def localize_geometrically(self, ch, steps):
        t, tried = self.tracks[ch], []
        for _ in range(steps):
            if t.cleared:
                return
            if t.radius <= SAFE_RADIUS:
                self.certified_clear(ch)
                return
            center, radius = self.estimate(ch)
            if radius <= 80. and all(dist(center, p) > 5. for p in tried):
                tried.append(center)
                self.optimistic_clears += 1
                if self.clear(center, ch):
                    return
            p = self.active_point(ch)
            if p is None:
                break
            self.actual_geometric_queries += 1
            t.active_queries += 1
            if self.measure(p, ch) == 'no_signal':
                self.no_signal_localization += 1
        if not t.cleared and t.radius <= SAFE_RADIUS:
            self.certified_clear(ch)

    def bounded_scan_service(self, ch):
        if not self.geometric_service:
            return super().bounded_scan_service(ch)
        self.scan_attempted.add(ch)
        self.localize_geometrically(ch, self.scan_steps)
        if not self.tracks[ch].cleared:
            self.bounded_deferred += 1

    def service(self, ch):
        if not self.geometric_service:
            return super().service(ch)
        self.localize_geometrically(ch, 4)
        t = self.tracks[ch]
        if t.cleared:
            return
        self.fallback_channels += 1
        first, bearing = t.observations[0]
        candidates = clipped_optical_cover(first, bearing, t.polygon)
        while candidates:
            k = min(range(len(candidates)), key=lambda i: (dist(candidates[i], self.client.position), i))
            p = candidates.pop(k)
            t.fallback_attempts += 1
            if self.clear(p, ch):
                return
        raise ArithmeticError('Certified optical coverage exhausted')

    def run(self):
        result = super().run()
        result.update(policy_version='q4_adaptive_service_iteration4_20260913',
                      order_mode=self.order_mode, endpoint_gate=self.endpoint_gate,
                      geometric_service=self.geometric_service,
                      endpoint_gate_skips=self.endpoint_gate_skips,
                      endpoint_evidence_seen=self.endpoint_evidence_seen,
                      geometric_candidates=self.geometric_candidates,
                      actual_geometric_queries=self.actual_geometric_queries,
                      meta_evaluations=self.meta_evaluations)
        self.client.log('adaptive_service_summary', **result)
        return result
