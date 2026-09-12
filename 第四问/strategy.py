"""Certified discovery, set-membership localization, finite optical fallback."""
from __future__ import annotations
from dataclasses import dataclass,field
import math
import time
from client import BudgetStop
from geometry import (DEFAULT_SIDE,add,sub,mul,unit,dist,dot,domain_polygon,incorporate,enclosing_circle,
                      polygon_distance,coverage_stations,open_route,clipped_optical_cover)

@dataclass
class Track:
    observations:list=field(default_factory=list)
    polygon:list=field(default_factory=lambda:list(domain_polygon()))
    center:tuple=(0.,0.)
    radius:float=float('inf')
    cleared:bool=False
    active_queries:int=0
    fallback_attempts:int=0
    def observe(self,p,bearing):
        self.observations.append((p,bearing))
        self.polygon=incorporate(self.polygon,p,bearing)
        self.center,self.radius=enclosing_circle(self.polygon)

class Policy:
    def __init__(self,client,grid='triangle',active=True,side=DEFAULT_SIDE):
        if not math.isfinite(side) or not 0 < side < 1000:
            raise ValueError('The directional-coverage certificate requires 0 < side < 1000')
        self.client,self.grid,self.active,self.side=client,grid,active,side
        self.tracks={}; self.discovered=set(); self.cleared=set()
        self.stations_visited=0; self.coverage_complete=False; self.fallback_channels=0
        self.no_signal_localization=0; self.safe_clears=0; self.optimistic_clears=0
        self.route=open_route(coverage_stations(grid,side))

    def clear(self,p,ch,safe=False):
        reply=self.client.action('/clear',p,ch)
        if reply['clear_result']=='success':
            self.cleared.add(ch); self.tracks[ch].cleared=True
            if safe: self.safe_clears+=1
            return True
        if safe: raise ArithmeticError(f'Certified clear failed on channel {ch}; inspect model/log')
        return False

    def measure(self,p,ch):
        reply=self.client.action('/measure',p,ch)
        result=reply['measure_result']
        if result!='no_signal':
            self.discovered.add(ch)
            t=self.tracks.setdefault(ch,Track())
            if result=='near': self.clear(p,ch,safe=True)
            else: t.observe(p,reply['svd_deg'])
        # Deliberately do not turn no_signal into a source-position exclusion.
        return result

    def scan(self):
        for station in self.route:
            unknown=[ch for ch in range(1,21) if ch not in self.discovered]
            known=[ch for ch,t in self.tracks.items() if not t.cleared and t.radius>19.8
                   and polygon_distance(station,t.polygon)<=1500.] if self.active else []
            # Unknown channels first. Same-channel first saves a switch when possible.
            if self.client.channel in unknown:
                unknown.remove(self.client.channel); unknown.insert(0,self.client.channel)
            for ch in unknown:
                self.measure(station,ch)
                if len(self.discovered)==16: break
            if len(self.discovered)==16:
                self.stations_visited+=1
                self.coverage_complete=True  # cardinality certificate, not full scan
                self.client.log('discovery_certificate',kind='count_upper_bound',channels=sorted(self.discovered))
                return
            for ch in known:
                if self.measure(station,ch)=='no_signal': self.no_signal_localization+=1
            self.stations_visited+=1
        self.coverage_complete=True
        self.client.log('discovery_certificate',kind='full_mesh',stations=self.stations_visited,
                        absent_channels=sorted(set(range(1,21))-self.discovered))
        if len(self.discovered)<10: raise ArithmeticError('Certified scan found fewer than the problem minimum of 10')

    def service(self,ch):
        t=self.tracks[ch]
        if t.radius<=19.8:
            self.clear(t.center,ch,safe=True); return
        if self.active:
            # Bounded heuristic: attempt short-range optical clear, then a transverse
            # observation. The proof of termination relies only on the fallback.
            tried=[]
            for iteration in range(4):
                if t.radius<=19.8:
                    self.clear(t.center,ch,safe=True); return
                if t.radius<=80 and all(dist(t.center,p)>5 for p in tried):
                    tried.append(t.center); self.optimistic_clears+=1
                    if self.clear(t.center,ch): return
                _,bearing=t.observations[-1]
                v=unit(bearing+90)
                offset=min(150.,max(45.,t.radius*.4))
                choices=[t.center,add(t.center,mul(v,offset)),add(t.center,mul(v,-offset))]
                # Avoid repeated fixed-error observations at an already measured point.
                previous=[p for p,_ in t.observations]+tried
                choices=[p for p in choices if all(dist(p,q)>5 for q in previous)]
                if not choices: break
                # Prefer a transverse baseline; alternate the side after a lost signal.
                p=choices[-1] if iteration%2 else choices[min(1,len(choices)-1)]
                tried.append(p); t.active_queries+=1
                if self.measure(p,ch)=='no_signal': self.no_signal_localization+=1
                if t.cleared: return
            if t.radius<=19.8:
                self.clear(t.center,ch,safe=True); return
        self.fallback_channels+=1
        first,bearing=t.observations[0]
        candidates=clipped_optical_cover(first,bearing,t.polygon)
        # NN route with no revisits gives a finite <=152-attempt fallback.
        while candidates:
            k=min(range(len(candidates)),key=lambda i:dist(candidates[i],self.client.position))
            p=candidates.pop(k); t.fallback_attempts+=1
            if self.clear(p,ch): return
        raise ArithmeticError(f'Optical coverage exhausted without clearing channel {ch}')

    def run(self):
        started=time.perf_counter(); self.client.action('/enter')
        completed,reason=False,'not_started'
        try:
            self.scan()
            while self.discovered-self.cleared:
                pending=self.discovered-self.cleared
                ch=min(pending,key=lambda c:(dist(self.client.position,self.tracks[c].center),c))
                self.service(ch)
            completed=self.coverage_complete and self.discovered==self.cleared
            reason='all_sources_cleared_with_certificate'
        except BudgetStop as e: reason=str(e)
        except ArithmeticError as e:
            reason='model_or_geometry_error: '+str(e)
            self.client.log('model_or_geometry_error',detail=str(e))
        self.client.action('/exit')
        result={'completed':completed,'stop_reason':reason,'grid':self.grid,'active':self.active,'side':self.side,
                'stations_total':len(self.route),'stations_visited':self.stations_visited,
                'discovered_channels':sorted(self.discovered),'cleared_channels':sorted(self.cleared),
                'discovery_certificate':'count_upper_bound' if len(self.discovered)==16 else ('full_mesh' if self.coverage_complete else None),
                'virtual_time_s':self.client.virtual,'runtime_s':time.perf_counter()-started,
                'fallback_channels':self.fallback_channels,'no_signal_localization':self.no_signal_localization,
                'safe_clears':self.safe_clears,'optimistic_clears':self.optimistic_clears,
                'active_queries':sum(t.active_queries for t in self.tracks.values()),
                'fallback_attempts':sum(t.fallback_attempts for t in self.tracks.values())}
        self.client.log('policy_summary',**result)
        return result
