"""Iteration 3: reuse service locations and reduce repeated local travel.

All early actions are bounded. Coverage replacements require a continuous
directional-coverage certificate; local clearing uses the actual 20 m rule.
"""
from geometry import dist, polygon_distance, dot, sub, add, mul
from planned_strategy import PlannedPolicy, PLANNED_CONFIG
from refined_geometry import source_domain, certify_layout
from strategy import Track

# Selected on the iteration-3 development scenes, before new held-out scenes.
EFFICIENT_CONFIG = dict(tight=False, local=True, stable=True, reuse=False, endpoint=True)


class EfficientPolicy(PlannedPolicy):
    def __init__(self, client, *, tight=False, local=False, stable=False, reuse=False, endpoint=False):
        super().__init__(client, **{**PLANNED_CONFIG, 'information':not stable})
        self.tight, self.local, self.stable, self.reuse = tight, local, stable, reuse
        self.endpoint = endpoint
        self.endpoint_tried = set()
        self.endpoint_probes = self.endpoint_successes = 0
        if tight:
            self.optical = 'polygon'
        self.local_busy = self.reuse_busy = False
        self.local_bursts = self.local_discoveries = self.local_clears = 0
        self.local_positions = []
        self.local_attempts = {}
        self.success_positions = {}
        self.search_positions = []
        self.coverage_replacements = []
        self.replacement_checks = 0

    def clear(self, p, ch, safe=False):
        success = super().clear(p, ch, safe)
        if success:
            self.success_positions[ch] = self.client.position
            if not self.in_station_scan and not self.local_busy and not self.reuse_busy:
                if self.local:
                    self.local_bundle()
                if self.reuse:
                    self.try_replacement()
        return success

    def local_bundle(self):
        """Clear nearby channels at the *current position*, without extra travel.

        Scan unknown channels optically only with at least three independent
        nearby source anchors. A failed optical attempt is not a no-signal
        observation and never creates a 1000 m exclusion.
        """
        if self.local_busy or self.local_bursts >= 6:
            return
        p = self.client.position
        if any(dist(p, q) <= 25. for q in self.local_positions):
            return
        anchors = []
        for ch, t in self.tracks.items():
            anchor = self.success_positions.get(ch)
            if anchor is None and not t.cleared and t.radius <= 60.:
                anchor = t.center
            if anchor is not None and dist(p, anchor) <= 80.:
                anchors.append(ch)
        clustered = len(anchors) >= 3
        candidates = []
        for ch in range(1, 21):
            if ch in self.cleared:
                continue
            attempted = self.local_attempts.get(ch, [])
            if len(attempted) >= 3 or any(dist(p,q) <= 20. for q in attempted):
                continue
            if ch in self.tracks:
                t = self.tracks[ch]
                if polygon_distance(p, t.polygon) > 20.+1e-6:
                    continue
                if not clustered and t.radius > 60.:
                    continue
                candidates.append((0, ch))
            elif clustered:
                candidates.append((1, ch))
        if not candidates:
            return
        self.local_busy = True
        self.local_bursts += 1
        self.local_positions.append(p)
        try:
            for was_unknown, ch in sorted(candidates):
                if ch in self.cleared:
                    continue
                self.local_attempts.setdefault(ch, []).append(p)
                self.optimistic_clears += 1
                reply = self.client.action('/clear', p, ch)
                success = reply['clear_result'] == 'success'
                if success:
                    if ch not in self.discovered:
                        self.local_discoveries += 1
                    self.discovered.add(ch)
                    self.cleared.add(ch)
                    if ch not in self.tracks:
                        self.tracks[ch] = Track(polygon=list(source_domain(self.circle_sides)))
                    self.tracks[ch].cleared = True
                    self.success_positions[ch] = p
                    self.local_clears += 1
                else:
                    self.failed_clear.setdefault(ch, []).append(p)
                self.hypothesis_cache.pop(ch, None)
                self.client.log('local_optical_result', channel=ch, position=p,
                                previously_unknown=bool(was_unknown), success=success)
        finally:
            self.local_busy = False

    def scan_station(self, station_id):
        super().scan_station(station_id)
        self.search_positions.append(self.station_points[station_id])
        if self.endpoint:
            for ch in sorted(self.discovered-self.cleared):
                self.try_near_endpoint(ch)

    def try_near_endpoint(self, ch):
        """A cheap optical attempt near an exterior positive observation.

        The nearest feasible endpoint is only a hypothesis. Its miss is paid
        and processed by the existing reactive measurement mechanism. This
        never declares clearing safe merely because the point lies in P.
        """
        if ch in self.endpoint_tried or ch in self.cleared:
            return
        t = self.tracks[ch]
        if t.radius <= 19.8 or not t.observations:
            return
        anchor = t.observations[-1][0]
        if dist(anchor,(0.,0.)) <= 1800. or dist(anchor,self.client.position) > 250.:
            return
        points = list(t.polygon)
        for a,b in zip(t.polygon,t.polygon[1:]+t.polygon[:1]):
            edge = sub(b,a)
            length2 = dot(edge,edge)
            f = max(0.,min(1.,dot(sub(anchor,a),edge)/length2)) if length2 else 0.
            points.append(add(a,mul(edge,f)))
        q = min(points,key=lambda p:dist(anchor,p))
        current = self.client.position
        center,_ = self.estimate(ch)
        extra = dist(current,q)+dist(q,center)-dist(current,center)
        if dist(current,q) > 250. or extra > 25.:
            return
        if any(dist(q,p) < 5. for p in self.failed_clear.get(ch,[])):
            return
        self.endpoint_tried.add(ch)
        self.endpoint_probes += 1
        self.optimistic_clears += 1
        before = self.client.virtual
        success = self.clear(q,ch)
        self.endpoint_successes += success
        self.client.log('near_endpoint_probe',channel=ch,position=q,success=success,
                        virtual_cost_s=self.client.virtual-before,additional_distance_to_estimate_m=extra)

    def try_replacement(self):
        """Replace a future station only after geometric certification.

        All still-unknown channels are queried now. The union of already
        scanned positions and future stations must retain the same sufficient
        directional coverage. Never infer source absence from a point sample.
        """
        if (self.coverage_complete or self.reuse_busy or len(self.discovered) >= 16
                or len(self.coverage_replacements) >= 4 or len(self.unvisited) < 2):
            return
        p = self.client.position
        if any(dist(p,q) < 5 for q in self.search_positions):
            return
        unknown = [ch for ch in range(1,21) if ch not in self.discovered]
        # At least four channels are absent. Charge all additional current
        # queries above those four, plus a 50 m margin, before paying for a
        # replacement. This ignores possible information loss on known sources.
        threshold = 30.*max(0,len(unknown)-4)+50.
        options = []
        for k,i in enumerate(self.unvisited):
            a = p if k == 0 else self.station_points[self.unvisited[k-1]]
            q = self.station_points[i]
            saving = dist(a,q)
            if k+1 < len(self.unvisited):
                b = self.station_points[self.unvisited[k+1]]
                saving += dist(q,b)-dist(a,b)
            if saving > threshold:
                options.append((-saving, i))
        for negative_saving, i in sorted(options)[:4]:
            remaining = [j for j in self.unvisited if j != i]
            points = self.search_positions+[p]+[self.station_points[j] for j in remaining]
            self.replacement_checks += 1
            certificate = certify_layout(points, max_depth=30)
            if not certificate['certified']:
                continue
            self.reuse_busy = True
            old_flag = self.in_station_scan
            self.in_station_scan = True
            try:
                if self.client.channel in unknown:
                    unknown.remove(self.client.channel)
                    unknown.insert(0,self.client.channel)
                for ch in unknown:
                    if ch not in self.discovered:
                        self.measure(p,ch)
                    if len(self.discovered) == 16:
                        break
            finally:
                self.in_station_scan = old_flag
                self.reuse_busy = False
            self.search_positions.append(p)
            self.unvisited.remove(i)
            event = dict(removed_station=i, scan_position=p, positions_for_certificate=points,
                         predicted_saved_distance_m=-negative_saving, certificate=certificate)
            self.coverage_replacements.append(event)
            self.client.log('coverage_replacement', **event)
            return

    def run(self):
        result = super().run()
        if self.coverage_replacements and len(self.discovered) < 16 and self.coverage_complete:
            result['discovery_certificate'] = 'adaptive_convex_cell_cover'
        result.update(policy_version='q4_efficiency_iteration3_20260913', tight_optical=self.tight,
                      local_bundle_enabled=self.local, stable_station_order=self.stable,
                      station_reuse_enabled=self.reuse, local_bursts=self.local_bursts,
                      near_endpoint_enabled=self.endpoint, endpoint_probes=self.endpoint_probes,
                      endpoint_successes=self.endpoint_successes,
                      local_discoveries=self.local_discoveries, local_clears=self.local_clears,
                      replacement_checks=self.replacement_checks,
                      dynamic_station_skips=len(self.coverage_replacements),
                      coverage_replacements=self.coverage_replacements)
        self.client.log('efficient_policy_summary', **result)
        return result
