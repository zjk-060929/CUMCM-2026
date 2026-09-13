"""Spatially smooth error validation and a no-small-step bearing guard.

The spatial scales below are sensitivity assumptions, not measured properties
of the competition simulator. Interval geometry never assumes independent noise.
"""
import math
import random

from client import Client
from geometry import dist,enclosing_circle
from metaheuristic_strategy import MetaheuristicPolicy,METAHEURISTIC_CONFIG
from planned_strategy import PlannedPolicy,PLANNED_CONFIG
from refined_geometry import source_domain,incorporate_fine
from simulator import LocalSimulator
from strategy import Track


class SpatialErrorSimulator(LocalSimulator):
    def __init__(self,sources,seed=0,scale=300.,robot_id='SELF-Q4'):
        super().__init__(sources,seed,'smooth',robot_id)
        if scale<=0:
            raise ValueError('Positive spatial scale required')
        self.scale=scale
        rng=random.Random(seed+710319)
        self.rotation,self.phase1,self.phase2=[rng.uniform(0,2*math.pi) for _ in range(3)]

    def error(self,position):
        x,y=position
        c,s=math.cos(self.rotation),math.sin(self.rotation)
        u,v=c*x+s*y,-s*x+c*y
        return .6*math.sin(u/self.scale+self.phase1)+.4*math.sin(v/(1.7*self.scale)+self.phase2)


class SpatialBearingGuard:
    """Reject nearby angular refinements, without treating far errors as independent.

    25 m is an engineering exclusion radius against 3-5 m jitter, not an
    assertion of independence beyond 25 m. Unknown-channel certified scans
    remain actual measurements; near/optical detection still uses real replies.
    """
    def __init__(self,*args,min_bearing_baseline=25.,**kwargs):
        super().__init__(*args,**kwargs)
        self.min_bearing_baseline=min_bearing_baseline
        self.close_bearing_actions_skipped=0
        self.close_bearing_results_ignored=0

    def measure(self,p,ch):
        track=self.tracks.get(ch)
        close=bool(track and any(dist(p,q)<self.min_bearing_baseline for q,_ in track.observations))
        if close and not self.in_station_scan:
            self.close_bearing_actions_skipped+=1
            self.client.log('close_bearing_action_skipped',channel=ch,position=p,
                            min_baseline_m=self.min_bearing_baseline)
            # Explicit local skip status; never fabricate a sensor result.
            return 'skipped_close_bearing'
        reply=self.client.action('/measure',p,ch)
        result=reply['measure_result']
        if result!='no_signal':
            self.discovered.add(ch)
            if ch not in self.tracks:
                self.tracks[ch]=Track(polygon=list(source_domain()))
            t=self.tracks[ch]
            if result=='near':
                self.clear(p,ch,safe=True)
            elif close:
                self.close_bearing_results_ignored+=1
                self.client.log('close_bearing_result_not_fused',channel=ch,position=p)
            else:
                t.observations.append((p,reply['svd_deg']))
                t.polygon=incorporate_fine(t.polygon,p,reply['svd_deg'])
                t.center,t.radius=enclosing_circle(t.polygon)
        elif ch not in self.cleared:
            self.negative.setdefault(ch,[]).append(p)
        self.hypothesis_cache.pop(ch,None)
        if not self.in_station_scan:
            self.piggyback(self.client.position,exclude=ch)
        return result

    def predicted_station_effects(self,ch,center):
        effects=dict(super().predicted_station_effects(ch,center))
        t=self.tracks[ch]
        for i,p in enumerate(self.station_points):
            if any(dist(p,q)<self.min_bearing_baseline for q,_ in t.observations):
                effects[i]=(t.radius,effects[i][1])
        return effects

    def run(self):
        result=super().run()
        result.update(spatial_bearing_guard_m=self.min_bearing_baseline,
                      close_bearing_actions_skipped=self.close_bearing_actions_skipped,
                      close_bearing_results_ignored=self.close_bearing_results_ignored)
        self.client.log('spatial_guard_summary',**result)
        return result


class SpatialPlannedPolicy(SpatialBearingGuard,PlannedPolicy):
    def __init__(self,client):
        super().__init__(client,**PLANNED_CONFIG)


class SpatialBeamPolicy(SpatialBearingGuard,MetaheuristicPolicy):
    def __init__(self,client):
        super().__init__(client,**METAHEURISTIC_CONFIG)
