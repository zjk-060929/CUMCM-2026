"""Integrate observed boundary rescue, single-action control and guarded routing.

All beliefs rank actions only. Continuous angle bounds and paid real feedback
remain authoritative. No policy reads the scene type, error scale or truth.
"""
import math
import random
import time
from collections import Counter

from action_strategy import ActionRouteModel, rolling_beam
from reliable_action_strategy import ReliableActionPolicy
from bayesian_belief import detection_mask, elementary
from boundary_geometry import arc_probe_points
from geometry import dist, sub, add, mul, dot, unit, clip, ALPHA_DEG, polygon_distance
from optimized_geometry import SAFE_RADIUS, safe_detour_point, polygon_optical_cover
from heuristic_search import neighbour


def radius_bound(poly, ceiling):
    if not poly:
        return ceiling
    return min(ceiling, math.hypot(max(x for x,y in poly)-min(x for x,y in poly),
                                  max(y for x,y in poly)-min(y for x,y in poly))/2)


def bearing_clip(poly, p, g, error):
    angle = math.degrees(math.atan2(g[1]-p[1], g[0]-p[0]))+error
    lo,hi = unit(angle-ALPHA_DEG),unit(angle+ALPHA_DEG)
    for normal in ((lo[1],-lo[0]),(-hi[1],hi[0])):
        poly = clip(poly,normal,dot(normal,p))
    return poly


class IntegratedModel(ActionRouteModel):
    def __init__(self, *args, entries, exits, station_origin, station_prefix, cap, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries,self.exits = entries,exits
        self.distance = [[dist(a,b)/5 for b in entries] for a in exits]
        self.station_origin,self.station_prefix,self.cap = station_origin,station_prefix,cap
        self.limit_enabled = False
        self.detail_cache = {}

    def station_length(self, route):
        points = [self.entries[n] for n in route if n in self.stations]
        return self.station_prefix+sum(dist(a,b) for a,b in zip([self.station_origin]+points,points))

    def details(self, route):
        key = tuple(route)
        if key in self.detail_cache:
            return self.detail_cache[key]
        elapsed=latency=known=0.; visited=0; radii=dict(self.task_radii); last=None; events=[]
        for node in route:
            elapsed += self.from_start[node] if last is None else self.distance[last][node]
            if node in self.stations:
                scan,gain,_ = self.scan_prediction(visited,node)
                latency += gain*(elapsed+.5*scan); elapsed += scan; visited |= 1<<node
                events.append((elapsed,{c:self.masks[node]&m for c,m in self.p['masks'].items()
                                        if c in self.pending[node]}))
                for task,r,possible in self.effects.get(node,[]):
                    if task not in radii or not possible:
                        continue
                    old=radii[task]; new=min(old,r)
                    benefit=max(0.,self.services[task]-5)*(old-new)/max(self.task_radii[task],1.)
                    if benefit>8.:
                        elapsed+=6.;known+=6.;radii[task]=new
            else:
                if node in self.blocked and visited!=self.allstations:
                    return (1e8+elapsed,1e8,latency,events)
                ratio=min(1.,radii.pop(node)/max(self.task_radii[node],1.))
                cost=5+max(0.,self.services[node]-5)*ratio
                elapsed+=cost;known+=cost
            last=node
        result=(elapsed,known,latency/max(self.p['expected_unknown'],1.),events)
        self.detail_cache[key]=result
        return result

    def cost(self, route):
        self.calls+=1
        if self.limit_enabled and self.station_length(route)>self.cap+1e-5:
            return 1e9+self.station_length(route)
        paid,_,latency,_=self.details(route)
        return paid+self.delay_weight*latency

    def last_discovery_charge(self, old, new):
        """Positive last-discovery delay under a finite per-channel N=16 model.

        Unlike an average signed delay, earlier discoveries cannot cancel the
        late last source. This is an explicit planning model, not a guarantee.
        """
        p=self.p; k=16-(20-len(p['unknown'])); p16=p['probability_n'].get(16,0.)
        if k<=0 or p16<1e-8:
            return 0.
        denom=p['coefficients'][k]
        if denom<=0:
            return 0.
        events=sorted([(t,side,m) for side,r in enumerate((old,new))
                       for t,m in self.details(r)[3]],key=lambda x:(x[0],x[1]))
        seen=[{c:0 for c in p['unknown']} for _ in range(2)]
        before=risk=0.
        for t,side,masks in events:
            aa=[p['likelihood'][c]*seen[0][c].bit_count()/self.sizes[c] for c in p['unknown']]
            ab=[p['likelihood'][c]*(seen[0][c]&seen[1][c]).bit_count()/self.sizes[c] for c in p['unknown']]
            fa=elementary(aa,k)[k]/denom; fab=elementary(ab,k)[k]/denom
            risk+=(t-before)*max(0.,fa-fab)
            for c,m in masks.items():seen[side][c]|=m
            before=t
        return p16*risk


class IntegratedPolicy(ReliableActionPolicy):
    def __init__(self, client, *, search='standard', patrol='gated', sensing=True,
                 boundary=True, actual_points=True):
        depth,width,branch=(5,12,8) if search=='expanded' else (3,4,6)
        super().__init__(client,depth=depth,width=width,branch=branch,opportunities=sensing)
        self.search,self.patrol,self.boundary,self.actual_points=search,patrol,boundary,actual_points
        self.station_reference=list(self.unvisited)
        self.station_origin=(0.,0.);self.station_prefix=0.
        self.base_cap=sum(dist(a,b) for a,b in zip([(0.,0.)]+[self.station_points[i] for i in self.unvisited],
                                                   [self.station_points[i] for i in self.unvisited]))+500.
        self.cap=self.base_cap
        self.integration_counts=Counter();self.guard_records=[]
        self.boundary_channels=set();self.rescues={};self.boundary_done=set();self.boundary_offers={}
        self.selected_actions={};self.action_exits={};self.gain_cache={};self.optional_count=0

    def boundary_action(self,ch):
        if not self.boundary or ch in self.boundary_done or ch in self.cleared:
            return None
        t=self.tracks[ch]
        if t.radius<=SAFE_RADIUS:
            return None
        state=self.rescues.get(ch)
        if state is None:
            interior=[i for i in self.visited_ids if dist(self.station_points[i],(0.,0.))<=1800.]
            if len(interior)<3 or len(self.boundary_channels)>2:
                return None
            if self.integration_counts['boundary_failed']>=2 and not self.integration_counts['boundary_success']:
                return None
            exterior=[p for p,_ in t.observations if dist(p,(0.,0.))>1800.]
            if not exterior:
                return None
            points=arc_probe_points(t.polygon,exterior[-1],1)
            if not points:
                return None
            q=points[0]; n=mul(q,1/1800.); side=(-n[1],n[0])
            probes=sorted([add(add(q,mul(n,60.)),mul(side,s*90.)) for s in (-1,1)],
                          key=lambda p:dist(p,self.client.position))
            state=dict(q=q,probes=probes,next=0)
        for stage in range(state['next'],5):
            if stage==0:
                kind,p,tag='clear',state['q'],'boundary_arc'
            elif stage in (1,3):
                kind,p,tag='measure',state['probes'][(stage-1)//2],'boundary_measure'
                if any(dist(p,x)<50. for x,_ in t.observations) or any(dist(p,x)<30. for x in self.negative.get(ch,[])):
                    continue
            else:
                center,radius=self.estimate(ch)
                if radius>80.:
                    continue
                kind,p,tag='clear',center,'boundary_center'
            if kind=='clear' and any(dist(p,x)<5. for x in self.failed_clear.get(ch,[])):
                continue
            action=(kind,ch,p,tag)
            self.boundary_offers[action]=dict(state,next=stage+1)
            return action
        return None

    def service_action(self,ch,endpoint=None):
        t=self.tracks[ch]; start=self.client.position
        if t.radius<=SAFE_RADIUS:
            p=safe_detour_point(t.center,t.radius,start,endpoint or t.center)
            return ('clear',ch,p,'safe_clear'),5.,p
        rescue=self.boundary_action(ch)
        if rescue:
            state=self.boundary_offers[rescue]
            # Finite rescue trajectory plus a residual recovery allowance.
            # The boundary hypothesis is used only after the observation gate.
            residual=12.+(2*math.hypot(60,90)+180)/5
            residual+=.15*self.forecaster(ch).clear_prediction(state['q'])['operations']
            return rescue,residual,state['q']
        if self.local_actions[ch]<8:
            options=self.action_candidates(ch,self.tried.get(ch,[]))
            if options:
                def price(a):
                    # Use the complete next connection, without a 25% multiplier.
                    return a[0]+(dist(a[2],endpoint)-dist(t.center,endpoint))/5 if endpoint else a[0]
                _,kind,p,pred=min(options,key=price)
                operation=pred['operations'] if kind=='clear' else 5.+(ch!=self.client.channel)+pred['remaining']
                return (kind,ch,p,'local_service'),operation,(p if kind=='clear' else t.center)
        candidates=self.covers.get(ch,polygon_optical_cover(t.polygon))
        candidates=[p for p in candidates if polygon_distance(p,t.polygon)<=SAFE_RADIUS+1e-6]
        if not candidates:
            raise ArithmeticError('Continuous optical cover exhausted')
        p=min(candidates,key=lambda p:dist(start,p))
        return ('clear',ch,p,'optical_cover'),self.forecaster(ch).cover_cost(p)[0],p

    def _search(self,model,warm,reference):
        candidates=[reference,model.repair([],range(model.n),2),model.nearest_completion([],range(model.n))]
        if warm:candidates.append(model.repair(warm,set(range(model.n))-set(warm),2))
        seed=8137+self.decisions
        for candidate in list(candidates):
            order,_=rolling_beam(model,candidate,self.depth,self.width,self.branch,seed)
            candidates.append(order)
            # A single beam pass per state keeps the enlarged budget finite.
            break
        best=min(candidates,key=model.cost);value=model.cost(best)
        rng=random.Random(seed+31)
        rounds,probes=(4,20) if self.search=='expanded' else (2,10)
        for _ in range(rounds):
            for kind in range(4):
                for _ in range(probes):
                    trial=neighbour(best,rng,kind);score=model.cost(trial)
                    if score<value-1e-8:best,value=trial,score
        return best

    def plan(self):
        started=time.perf_counter();self.selected_actions={};self.boundary_offers={}
        labels=[('station',i) for i in self.unvisited]+[('task',c) for c in sorted(self.discovered-self.cleared)]
        entries=[];exits=[];radii={};services={};effects={};pending={};masks={};blocked=set()
        for j,(kind,k) in enumerate(labels):
            if kind=='station':
                p=self.station_points[k];pending[j]=self.pending_channels(k);masks[j]=detection_mask(p);q=p
            else:
                action,cost,q=self.service_action(k);self.selected_actions[k]=action
                p=action[2] if self.actual_points else self.forecaster(k).center
                q=q if self.actual_points else p
                radii[j]=self.tracks[k].radius;services[j]=cost
                if radii[j]>SAFE_RADIUS and self.early_actions[k]>=4 and self.unvisited and not action[3].startswith('boundary'):
                    blocked.add(j)
            entries.append(p);exits.append(q)
        for j,(kind,ch) in enumerate(labels):
            if kind!='task' or radii[j]<=SAFE_RADIUS or not self.opportunities:continue
            for s in pending:
                p=entries[s]
                if not self.valid_probe(ch,p):continue
                pred=self.forecaster(ch).measurement_prediction(p,50.)
                if pred['valid']:effects.setdefault(s,[]).append((j,pred['radius'],True))
        model=IntegratedModel(labels,entries,self.client.position,radii,effects,blocked,
            entries=entries,exits=exits,station_origin=self.station_origin,station_prefix=self.station_prefix,cap=self.cap,
            posterior=self.belief.posterior(self.discovered),pending=pending,masks=masks,services=services,delay_weight=self.delay_weight)
        index={label:i for i,label in enumerate(labels)}
        stations=[index[('station',i)] for i in self.station_reference if ('station',i) in index]
        reference=model.repair(stations,[i for i in radii],2)
        warm=[index[k] for k in self.last_plan if k in index]
        # Station scans can be interrupted. Charge every real arrival and restore
        # feasibility explicitly if a partial scan makes the old cap impossible.
        if model.station_length(reference)>self.cap+1e-5 and self.patrol!='off':
            self.integration_counts['partial_scan_cap_repairs']+=1
            self.cap=model.cap=model.station_length(reference)
        model.limit_enabled=self.patrol!='off'
        capped=self._search(model,warm,reference)
        chosen=capped
        if self.patrol=='gated' and radii:
            model.limit_enabled=False
            free=self._search(model,capped,reference)
            needed=model.station_length(free)>self.cap+1e-5
            paid_gain=model.details(capped)[0]-model.details(free)[0]
            service_gain=model.details(capped)[1]-model.details(free)[1]
            if needed and paid_gain>=20. and service_gain>1e-6:
                charge=model.last_discovery_charge(capped,free)
                accepted=paid_gain-charge>=20.
                event=dict(paid_gain_s=paid_gain,known_service_gain_s=service_gain,discovery_risk_s=charge,
                           old_cap_m=self.cap,candidate_station_m=model.station_length(free),accepted=accepted)
                self.guard_records.append(event)
                if accepted:
                    chosen=free;self.cap=model.station_length(free);self.integration_counts['guard_relaxations']+=1
                else:self.integration_counts['discovery_risk_rejections']+=1
        self.station_reference=[labels[i][1] for i in chosen if i in model.stations]
        self.last_plan=[labels[i] for i in chosen]
        self.schedule_replans+=1;self.meta_evaluations+=model.calls
        self.schedule_planning_s+=time.perf_counter()-started
        self.budget_counts[f'{self.depth}x{self.width}']+=1
        self.client.log('integrated_global_plan',decision=self.decisions,order=self.last_plan,
                        cost_evaluations=model.calls,cap_m=self.cap)
        return self.last_plan,{label:p for label,p in zip(labels,entries)}

    def incremental_gain(self,ch,p,endpoint=None):
        if not self.valid_probe(ch,p):return 0.
        t=self.tracks[ch];bank=self.forecaster(ch)
        if bank.depleted:return 0.
        key=(ch,len(t.observations),len(self.negative.get(ch,[])),len(self.failed_clear.get(ch,[])),p,endpoint)
        if key in self.gain_cache:return self.gain_cache[key]
        count=len(bank.points)
        indices=sorted({round(i*(count-1)/min(4,count-1)) for i in range(min(5,count))}) if count>1 else list(range(count))
        values=[]
        for i in indices:
            g=bank.points[i];states=bank.states[i]
            if not states:continue
            first=[bearing_clip(t.polygon,p,g,e) for e in (-ALPHA_DEG,0.,ALPHA_DEG)]
            r1=max(radius_bound(poly,t.radius) for poly in first)
            if endpoint is not None:
                r2=max(radius_bound(bearing_clip(t.polygon,endpoint,g,e),t.radius) for e in (-ALPHA_DEG,0.,ALPHA_DEG))
                both=max(radius_bound(bearing_clip(poly,endpoint,g,e),t.radius) for poly in first for e in (-ALPHA_DEG,0.,ALPHA_DEG))
            def receives(q,r,n):return dist(g,q)<=r and (n is None or dot(n,sub(q,g))>=0.)
            gain=0.
            for reach,normal in states:
                a=receives(p,reach,normal);b=endpoint is not None and receives(endpoint,reach,normal)
                before=r2 if b else t.radius
                after=(both if b else r1) if a else before
                gain+=max(0.,before-after)
            values.append(gain/len(states))
        value=.35*sum(values)/len(values) if values else 0.
        self.gain_cache[key]=value
        return value

    def information_benefit(self,ch,p):
        if not self.opportunities or self.optional_count>=40:return -math.inf
        return self.incremental_gain(ch,p)

    def opportunistic(self,destination):
        if not self.opportunities or self.optional_count>=40:return None
        start=self.client.position;edge=sub(destination,start);l2=dot(edge,edge);choices=[]
        for ch in sorted(self.discovered-self.cleared):
            t=self.tracks[ch]
            if t.radius<=SAFE_RADIUS:
                p=safe_detour_point(t.center,t.radius,start,destination)
                extra=dist(start,p)+dist(p,destination)-dist(start,destination)
                if extra<1e-5:return 'clear',ch,p,'safe_clear'
                continue
            points=[start]
            if l2>=160**2:
                f=max(0.,min(1.,dot(sub(self.forecaster(ch).center,start),edge)/l2))
                points += [add(start,mul(edge,x)) for x in dict.fromkeys((.25,.5,.75,f)) if .02<x<.98]
            endpoint=destination if self.valid_probe(ch,destination) else None
            for p in points:
                gain=self.incremental_gain(ch,p,endpoint if p!=destination else None)
                cost=5.+(ch!=self.client.channel)+(dist(start,p)+dist(p,destination)-dist(start,destination))/5
                if gain>max(12.,cost+4.):choices.append((cost-gain,ch,p))
        if choices:
            _,ch,p=min(choices)
            return 'measure',ch,p,'at_point_probe' if p==start else 'enroute_probe'
        return None

    def next_action(self,route,locations):
        for ch,p in sorted(self.near.items()):
            if ch not in self.cleared and dist(self.client.position,p)<1e-7:return 'clear',ch,p,'safe_clear'
        opportunity=self.opportunistic(locations[route[0]])
        if opportunity:return opportunity
        kind,k=route[0]
        if kind=='task':
            return self.selected_actions[k]
        destination=locations[route[0]];posterior=self.belief.posterior(self.discovered)
        choices=self.pending_channels(k)
        ch=max(choices,key=lambda c:(self.belief.discovery_probability(c,destination,posterior)/(5+(c!=self.client.channel)),c==self.client.channel,-c))
        known=[(self.information_benefit(c,destination),c) for c in sorted(self.discovered-self.cleared)]
        if known and max(known)[0]>12.:
            return 'measure',max(known)[1],destination,'station_probe'
        return 'measure',ch,destination,'station_unknown'

    def execute(self,action):
        kind,ch,p,tag=action
        old_positive=len(self.tracks[ch].observations) if ch in self.tracks else 0
        old_radius=self.tracks[ch].radius if ch in self.tracks else math.inf
        if tag.startswith('boundary'):
            if ch not in self.rescues:self.integration_counts['boundary_episodes']+=1
            self.rescues[ch]=self.boundary_offers[action]
        if tag=='optical_cover':
            if ch not in self.covers:
                self.covers[ch]=polygon_optical_cover(self.tracks[ch].polygon);self.fallback_channels+=1
            self.covers[ch].remove(p)
        if any(dist(p,q)<1e-7 for q in self.station_points):
            self.station_prefix+=dist(self.station_origin,p);self.station_origin=p
        super().execute(action)
        if tag in ('at_point_probe','enroute_probe','station_probe'):self.optional_count+=1
        if ch in self.tracks:
            t=self.tracks[ch]
            positive=len(t.observations)>old_positive or ch in self.near
            if positive and dist(p,(0.,0.))<=1800. and any(dist(p,q)<1e-7 for q in self.station_points):
                self.boundary_channels.add(ch)
            if ch in self.rescues and ch not in self.boundary_done:
                exhausted=t.radius>SAFE_RADIUS and (self.rescues[ch]['next']>=5 or (not t.cleared and self.boundary_action(ch) is None))
                if t.cleared or exhausted:
                    self.boundary_done.add(ch)
                    self.integration_counts['boundary_success' if t.cleared else 'boundary_failed']+=1
            if positive and t.radius<.85*old_radius:self.gain_cache={k:v for k,v in self.gain_cache.items() if k[0]!=ch}

    def run(self):
        result=super().run()
        result.update(policy_version='integrated_q4_20260913',circle_sides=1024,search=self.search,patrol=self.patrol,
                      integration_counts=dict(self.integration_counts),guard_records=self.guard_records,
                      optional_count=self.optional_count,station_prefix_m=self.station_prefix,base_cap_m=self.base_cap,
                      final_cap_m=self.cap)
        return result
