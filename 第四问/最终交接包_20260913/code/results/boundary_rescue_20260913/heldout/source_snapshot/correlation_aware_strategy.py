"""Reject tiny-baseline follow-up bearings; retain bounded-error geometry."""
import math

from efficient_strategy import EfficientPolicy, EFFICIENT_CONFIG
from geometry import dist, sub, dot


CORRELATION_CONFIG = dict(min_baseline=30., min_parallax_deg=3., min_approach=80.)


class CorrelationAwarePolicy(EfficientPolicy):
    def __init__(self, client, *, min_baseline=30., min_parallax_deg=3., min_approach=80.):
        super().__init__(client, **EFFICIENT_CONFIG)
        self.min_baseline = min_baseline
        self.min_parallax_deg = min_parallax_deg
        self.min_approach = min_approach
        self.correlated_measure_skips = 0
        self.correlated_skip_reasons = {}

    def redundant_reason(self, p, ch):
        # Discovery/absence checks for unknown channels remain mandatory.
        t = self.tracks.get(ch)
        if t is None or not t.observations or t.cleared:
            return None
        positions = [q for q, _ in t.observations]
        if any(dist(p, q) <= self.min_baseline for q in positions):
            return 'short_baseline'
        # At a scheduled station the travel is already paid. This extra gate
        # applies only to optional service/after-failure/piggyback bearings.
        if self.in_station_scan:
            return None
        center, _ = self.estimate(ch)
        v = sub(p, center)
        new_range = dist(p, center)
        for q in positions:
            w = sub(q, center)
            old_range = dist(q, center)
            if old_range-new_range >= self.min_approach:
                return None
            if old_range*new_range > 1e-9:
                cosine = max(-1., min(1., dot(v, w)/(old_range*new_range)))
                # Bearings on the same line, including opposite directions,
                # give a weak intersection: use the angle between the lines.
                angle = math.degrees(math.acos(abs(cosine)))
                if angle >= self.min_parallax_deg:
                    return None
        return 'weak_geometric_change'

    def measure(self, p, ch):
        reason = self.redundant_reason(p, ch)
        if reason:
            self.correlated_measure_skips += 1
            self.correlated_skip_reasons[reason] = self.correlated_skip_reasons.get(reason, 0)+1
            self.client.log('correlated_measure_skipped', channel=ch, position=p, reason=reason)
            # Internal status only: no move, no fabricated sensor response,
            # no positive/negative observation, no narrowing of the polygon.
            return 'skipped_correlated'
        return super().measure(p, ch)

    def run(self):
        result = super().run()
        result.update(policy_version='q4_spatial_correlation_correction_20260913',
                      correlation_config=dict(min_baseline=self.min_baseline,
                                              min_parallax_deg=self.min_parallax_deg,
                                              min_approach=self.min_approach),
                      correlated_measure_skips=self.correlated_measure_skips,
                      correlated_skip_reasons=self.correlated_skip_reasons)
        self.client.log('correlation_policy_summary', **result)
        return result
