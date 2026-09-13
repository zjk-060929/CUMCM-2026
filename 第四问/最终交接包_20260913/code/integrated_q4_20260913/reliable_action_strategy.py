"""Bound optional sensing when its forecast has no reliable supporting samples.

Frozen ActionPolicy remains the first candidate and regression comparator.
"""
import math
from collections import Counter
from action_strategy import ActionPolicy

RELIABLE_CONFIG=dict(depth=3,width=4,branch=6,opportunities=True,cover_risk=True,
                     delay_weight=.08,channel_order=True)


class ReliableActionPolicy(ActionPolicy):
    def __init__(self,client,**kwargs):
        super().__init__(client,**{**RELIABLE_CONFIG,**kwargs})
        self.weak_probes=Counter();self.probe_vetoes=Counter()

    def valid_probe(self,ch,p):
        if not super().valid_probe(ch,p):return False
        if ch in self.covers:
            # The finite covering route is now the planned recovery action.
            # Reconsider other global tasks, but don't regenerate optional probes
            # for the same unresolved channel at every optical disk.
            self.probe_vetoes['optical_recovery_active']+=1;return False
        if self.weak_probes[ch]>=3:
            self.probe_vetoes['three_low_gain_probes']+=1;return False
        if self.forecaster(ch).depleted:
            # Restored particle samples are useful for a coarse location guess;
            # they cannot support a positive value-of-information assertion.
            self.probe_vetoes['depleted_belief']+=1;return False
        return True

    def execute(self,action):
        kind,ch,p,tag=action
        old=self.tracks[ch].radius if ch in self.tracks else math.inf
        old_n=len(self.tracks[ch].observations) if ch in self.tracks else 0
        super().execute(action)
        if kind=='measure' and ch in self.tracks:
            track=self.tracks[ch]
            improved=(ch in self.near or (len(track.observations)>old_n and track.radius<=.85*old))
            if improved:self.weak_probes[ch]=0
            elif tag in ('at_point_probe','enroute_probe','station_probe'):
                self.weak_probes[ch]+=1

    def run(self):
        result=super().run()
        result.update(policy_version='action_replanning_with_reliable_opportunities_20260913',
                      optional_probe_vetoes=dict(self.probe_vetoes),optional_probe_weak_limit=3)
        return result
