"""One accepted sensor/clear request per globally reconsidered decision.

All geometry and the simulator are the frozen previous-turn versions. Forecasts
rank actions only; neither particles nor spatial spacing certify localization.
"""
import math
import random
import time
from collections import Counter

from bayesian_belief import detection_mask
from client import BudgetStop
from geometry import dist, add, sub, mul, dot, unit, enclosing_circle, polygon_distance
from heuristic_search import RouteModel, neighbour
from optimized_geometry import SAFE_RADIUS, polygon_optical_cover, safe_clear_point, safe_detour_point
from predictive_strategy_q4 import PredictivePolicy, PREDICTIVE_CONFIG
from predictive_models_q4 import TargetForecast
from refined_geometry import source_domain, incorporate_fine
from strategy import Track


def optical_tour(points, start):
    """Finite NN route through every certified disk; no representative truncation."""
    pending=list(points); route=[]; p=start; elapsed=0.; times=[]
    while pending:
        j=min(range(len(pending)), key=lambda j:(dist(p,pending[j]),j))
        q=pending.pop(j); elapsed+=dist(p,q)/5+3.; times.append(elapsed+2.)
        route.append(q); p=q
    return route, sum(times)/max(1,len(times)), (times[-1] if times else 5.)


class CoverForecast(TargetForecast):
    """Mix a sample estimate with continuous-cover workload, including its tail.

    Mean disk-index time is an explicit heuristic prior, not a calibrated target
    probability. The last disk time is a genuine bound for this particular tour.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cover=polygon_optical_cover(self.poly)
        self.tail_weight=min(.85,.2+.12*len(self.failed)+.35*self.depleted)
        self.cover_cache={}
        self.initial_recovery=max(self.initial_recovery,self.cover_cost(self.center)[0])

    def cover_cost(self,q):
        q=tuple(q)
        if q not in self.cover_cache:
            # Remove only disks fully contained in an already failed clear disk.
            remaining=[c for c in self.cover if not any(dist(c,f)+SAFE_RADIUS<=20.-1e-7 for f in self.failed)]
            _,mean,worst=optical_tour(remaining,q)
            self.cover_cache[q]=(mean,worst)
        return self.cover_cache[q]

    def clear_prediction(self,q):
        pred=dict(super().clear_prediction(q))
        if pred['success']==1.:return pred
        mean,worst=self.cover_cost(q)
        sampled=pred['failure_recovery']
        recovery=max(sampled,(1-self.tail_weight)*sampled+self.tail_weight*mean)
        cover_hit=sum(dist(q,c)<=20 for c in self.cover)/max(1,len(self.cover))
        success=min(pred['success'],(1-self.tail_weight)*pred['success']+self.tail_weight*cover_hit)
        pred.update(success=success,failure_recovery=recovery,cover_mean_s=mean,
                    cover_worst_s=worst,cover_disks=len(self.cover),tail_weight=self.tail_weight,
                    operations=success*5+(1-success)*(3+recovery))
        return pred

    def measurement_prediction(self,q,min_baseline=25.):
        pred=dict(super().measurement_prediction(q,min_baseline))
        if not pred['valid']:return pred
        mean,worst=self.cover_cost(q)
        no_signal=max(pred['no_signal_remaining'],self.tail_weight*mean+(1-self.tail_weight)*pred['no_signal_remaining'])
        pred.update(no_signal_remaining=no_signal,
                    remaining=pred['reception']*pred['received_remaining']+(1-pred['reception'])*no_signal)
        return pred


class ActionRouteModel(RouteModel):
    """Price full travel, unfinished channel scans, service, and first reception."""
    def __init__(self,*args,posterior,pending,masks,services,delay_weight,**kwargs):
        super().__init__(*args,**kwargs)
        self.p=posterior;self.pending=pending;self.masks=masks;self.services=services
        self.delay_weight=delay_weight;self.prefix_cache={};self.union_cache={0:0}
        self.sizes={c:m.bit_count() for c,m in posterior['masks'].items()}
        self.allstations=sum(1<<s for s in self.stations)

    def scan_prediction(self,visited,node):
        key=visited,node
        if key in self.prefix_cache:return self.prefix_cache[key]
        covered=self.union_cache[visited];visible=self.masks[node]
        after=visited|(1<<node);self.union_cache[after]=covered|visible
        scans=float(len(self.pending[node]));gain=0.;found=0.
        for c,mask in self.p['masks'].items():
            mass=self.p['existence'][c]/self.sizes[c]
            before=(mask&covered).bit_count()*mass
            if c in self.pending[node]:scans-=before
            found+=before
            gain+=(mask&visible&~covered).bit_count()*mass
        self.prefix_cache[key]=(max(0.,scans)*6.,gain,found)
        return self.prefix_cache[key]

    def cost(self,route):
        self.calls+=1
        if not route:return 0.
        elapsed=latency=0.;visited=0;radii=dict(self.task_radii);last=None
        for node in route:
            elapsed+=self.from_start[node] if last is None else self.distance[last][node]
            if node in self.stations:
                scan,gain,_=self.scan_prediction(visited,node)
                latency+=gain*(elapsed+.5*scan);elapsed+=scan;visited|=1<<node
                for task,r,possible in self.effects.get(node,[]):
                    if task not in radii or not possible:continue
                    old=radii[task];new=min(old,r)
                    benefit=max(0.,self.services[task]-5)*(old-new)/max(self.task_radii[task],1.)
                    if benefit>6.:
                        elapsed+=6.;radii[task]=new
            else:
                if node in self.blocked and visited!=self.allstations:return 1e8+elapsed
                ratio=min(1.,radii.pop(node)/max(self.task_radii[node],1.))
                elapsed+=5+max(0.,self.services[node]-5)*ratio
            last=node
        return elapsed+self.delay_weight*latency/max(self.p['expected_unknown'],1.)


def rolling_beam(model,warm,depth,width,branch,seed):
    """Warm incumbent, local route exchanges, then bounded prefix rollouts."""
    rng=random.Random(seed)
    initial=model.nearest_completion([],range(model.n))
    candidates=[initial]
    if warm:
        candidates.append(model.repair(warm,set(range(model.n))-set(warm),2))
    best=min(candidates,key=model.cost);value=model.cost(best)
    initial_cost=value
    for _ in range(2):
        for kind in range(4):
            for j in range(8):
                route=neighbour(best,rng,kind);cost=model.cost(route)
                if cost<value-1e-8:best,value=route,cost
    beam=[([],set(range(model.n)))]
    for _ in range(min(depth,model.n)):
        expanded=[]
        for prefix,remaining in beam:
            available=[k for k in remaining if k not in model.blocked or not (remaining&model.stations)]
            nearest=sorted(available,key=lambda k:(model.distance[prefix[-1]][k] if prefix else model.from_start[k],k))[:branch]
            choices=list(dict.fromkeys(nearest+[k for k in best if k in available][:2]))
            for k in choices:
                new=prefix+[k];rest=remaining-{k};route=model.nearest_completion(new,rest)
                cost=model.cost(route);expanded.append((cost,new,rest))
                if cost<value-1e-8:best,value=route,cost
        expanded.sort(key=lambda x:(x[0],x[1]));beam=[(p,r) for _,p,r in expanded[:width]]
    return best,dict(initial_cost=initial_cost,final_cost=value,cost_evaluations=model.calls)


class ActionPolicy(PredictivePolicy):
    def __init__(self,client,*,depth=3,width=4,branch=6,opportunities=True,
                 cover_risk=True,delay_weight=.08,channel_order=True,adaptive_budget=False,
                 unknown_probes=False,conservative_reception=False):
        super().__init__(client,**{**PREDICTIVE_CONFIG,'delay_weight':delay_weight})
        self.depth,self.width,self.branch=depth,width,branch
        self.opportunities,self.cover_risk,self.channel_order=opportunities,cover_risk,channel_order
        self.adaptive_budget,self.unknown_probes=adaptive_budget,unknown_probes
        self.conservative_reception=conservative_reception
        self.last_changed=True;self.budget_counts=Counter();self.probed_unknown={}
        self.cost_parts={};self.failure_wait={};self.recovery_samples=[]
        self.tested={i:set() for i in self.unvisited};self.near={}
        self.early_actions=Counter();self.local_actions=Counter();self.tried={};self.covers={}
        self.action_kinds=Counter();self.decisions=0;self.interrupted_stations=0
        self.opportunity_cache={};self.effect_cache={};self.last_station=None

    def forecaster(self,ch):
        if not self.cover_risk:return super().forecaster(ch)
        t=self.tracks[ch];key=ch,len(t.observations),len(self.negative.get(ch,[])),len(self.failed_clear.get(ch,[]))
        if key not in self.forecast_cache:
            for k in list(self.forecast_cache):
                if k[0]==ch:del self.forecast_cache[k]
            bank=CoverForecast(t,self.negative.get(ch,[]),self.failed_clear.get(ch,[]),self.estimate(ch)[0],self.recovery_scale)
            if self.conservative_reception and bank.depleted:
                base_prediction=bank.measurement_prediction
                def cautious(q,min_baseline=25.,base=base_prediction,bank=bank):
                    pred=dict(base(q,min_baseline))
                    if pred['valid']:
                        p=pred['reception'];received=(pred['radius']-(1-p)*bank.radius)/max(p,1e-8)
                        p*=.35
                        pred.update(reception=p,radius=(1-p)*bank.radius+p*received,
                                    remaining=p*pred['received_remaining']+(1-p)*pred['no_signal_remaining'])
                    return pred
                bank.measurement_prediction=cautious
            self.forecast_states_depleted+=int(bank.depleted);self.forecast_cache[key]=bank
        return self.forecast_cache[key]

    def before_action(self,path,p,ch):
        # A near reply gives its own safe disk and needs no bearing bank.
        if ch in self.near:return None
        return super().before_action(path,p,ch)

    def execute(self,action):
        """Exactly one request. Automatic near/clear reactions are future decisions."""
        kind,ch,p,tag=action;before=self.client.counter
        old_position=self.client.position;old_time=self.client.virtual
        self.decisions+=1;self.action_kinds[tag]+=1
        self.client.log('action_decision',decision=self.decisions,kind=kind,channel=ch,position=p,reason=tag)
        if kind=='measure':
            was_known=ch in self.discovered
            reply=self.client.action('/measure',p,ch);result=reply['measure_result']
            self.last_changed=result!='no_signal' or was_known or dist(old_position,p)>1e-7
            if tag=='unknown_opportunity':self.probed_unknown.setdefault(tuple(p),set()).add(ch)
            for i,q in enumerate(self.station_points):
                if dist(p,q)<1e-7:self.tested[i].add(ch)
            if result!='no_signal':
                self.discovered.add(ch);t=self.tracks.setdefault(ch,Track(polygon=list(source_domain())))
                if result=='near':
                    self.near[ch]=p;t.center=p;t.radius=5.
                    # Circumscribed square contains the 5 m near disk.
                    t.polygon=[add(p,(x,y)) for x,y in ((-5,-5),(5,-5),(5,5),(-5,5))]
                    t.center,t.radius=enclosing_circle(t.polygon)
                elif any(dist(p,q)<50. for q,_ in t.observations):
                    self.close_bearing_results_ignored+=1
                else:
                    t.observations.append((p,reply['svd_deg']))
                    t.polygon=incorporate_fine(t.polygon,p,reply['svd_deg']);t.center,t.radius=enclosing_circle(t.polygon)
            else:
                self.negative.setdefault(ch,[]).append(p)
                if was_known:self.no_signal_localization+=1
            if was_known:self.tracks[ch].active_queries+=1
        else:
            t=self.tracks[ch]
            prediction=None if t.radius<=SAFE_RADIUS else self.forecaster(ch).clear_prediction(p)
            reply=self.client.action('/clear',p,ch);t=self.tracks[ch]
            success=reply['clear_result']=='success'
            self.last_changed=True
            if success:
                self.cleared.add(ch);t.cleared=True
                if tag=='safe_clear':self.safe_clears+=1
            else:
                if tag=='safe_clear':raise ArithmeticError('Certified clear failed')
                self.failed_clear.setdefault(ch,[]).append(p)
                self.failure_wait.setdefault(ch,(self.client.virtual,prediction['failure_recovery']))
            if tag=='optical_cover':t.fallback_attempts+=1
            else:self.optimistic_clears+=int(tag!='safe_clear')
            if success and ch in self.failure_wait:
                start,forecast=self.failure_wait.pop(ch)
                self.recovery_samples.append(dict(channel=ch,predicted_recovery_s=forecast,
                    actual_elapsed_to_success_s=self.client.virtual-start))
        self.hypothesis_cache.pop(ch,None)
        if tag=='local_service':
            self.early_actions[ch]+=1;self.local_actions[ch]+=1;self.tried.setdefault(ch,[]).append(p)
        ledger=self.cost_parts.setdefault(tag,dict(actions=0,move_s=0.,operation_s=0.))
        move=dist(old_position,p)/5
        ledger['actions']+=1;ledger['move_s']+=move;ledger['operation_s']+=self.client.virtual-old_time-move
        assert self.client.counter-before==1,'Hidden multi-action service is forbidden'

    def pending_channels(self,i):
        return set(range(1,21))-self.discovered-self.tested[i]

    def refresh_certificate(self):
        for i in list(self.unvisited):
            if not self.pending_channels(i):
                self.unvisited.remove(i);self.visited_ids.append(i)
        self.stations_visited=len(self.visited_ids)
        if len(self.discovered)==16:self.unvisited=[]
        self.coverage_complete=not self.unvisited
        if self.coverage_complete and len(self.discovered)<10:raise ArithmeticError('Certified scan found fewer than 10')

    def plan(self):
        started=time.perf_counter()
        labels=[('station',i) for i in self.unvisited]+[('task',c) for c in sorted(self.discovered-self.cleared)]
        points=[];radii={};services={};effects={};blocked=set();pending={};masks={}
        for j,(kind,k) in enumerate(labels):
            if kind=='station':
                q=self.station_points[k];pending[j]=self.pending_channels(k);masks[j]=detection_mask(q)
            else:
                t=self.tracks[k];q=t.center if t.radius<=SAFE_RADIUS else self.forecaster(k).center
                if k in self.covers:
                    # Keep the original covering family, pruning only disks disjoint
                    # from the updated region. Every possible source remains covered.
                    self.covers[k]=[c for c in self.covers[k] if polygon_distance(c,t.polygon)<=SAFE_RADIUS+1e-6]
                    if self.covers[k]:q=min(self.covers[k],key=lambda c:dist(self.client.position,c))
                radii[j]=t.radius
                services[j]=5. if t.radius<=SAFE_RADIUS else self.forecaster(k).clear_prediction(q)['operations']
                if t.radius>SAFE_RADIUS and self.early_actions[k]>=4 and self.unvisited:blocked.add(j)
            points.append(q)
        for j,(kind,k) in enumerate(labels):
            if kind!='task' or radii[j]<=SAFE_RADIUS:continue
            bank=self.forecaster(k)
            for s,(sk,si) in enumerate(labels):
                if sk!='station':continue
                q=points[s]
                if not self.valid_probe(k,q):continue
                pred=bank.measurement_prediction(q,50.)
                if pred['valid']:effects.setdefault(s,[]).append((j,pred['radius'],True))
        posterior=self.belief.posterior(self.discovered)
        model=ActionRouteModel(labels,points,self.client.position,radii,effects,blocked,
              posterior=posterior,pending=pending,masks=masks,services=services,delay_weight=self.delay_weight)
        index={k:i for i,k in enumerate(labels)}
        warm=[index[k] for k in self.last_plan if k in index and index[k] not in blocked]
        warm += [index[k] for k in self.last_plan if k in index and index[k] in blocked]
        depth,width,branch=self.depth,self.width,self.branch
        if self.adaptive_budget and not self.last_changed:depth,width,branch=3,4,6
        self.budget_counts[f'{depth}x{width}']+=1
        order,stats=rolling_beam(model,warm,depth,width,branch,8137+self.decisions)
        self.last_plan=[labels[i] for i in order]
        self.schedule_replans+=1;self.meta_evaluations+=stats['cost_evaluations']
        self.schedule_planning_s+=time.perf_counter()-started
        self.client.log('single_action_global_plan',order=self.last_plan,depth=depth,width=width,**stats)
        return self.last_plan,{k:p for k,p in zip(labels,points)}

    def valid_probe(self,ch,p):
        t=self.tracks[ch]
        return (t.radius>SAFE_RADIUS and polygon_distance(p,t.polygon)<=1500.
                and all(dist(p,q)>=50. for q,_ in t.observations)
                and all(dist(p,q)>=5. for q in self.negative.get(ch,[])))

    def information_benefit(self,ch,p):
        if not self.valid_probe(ch,p):return -math.inf
        bank=self.forecaster(ch);pred=bank.measurement_prediction(p,50.)
        if not pred['valid']:return -math.inf
        before=bank.clear_prediction(bank.center)['operations']
        return max(0.,before-5.)*max(0.,1-pred['radius']/max(self.tracks[ch].radius,1.))

    def opportunistic(self,destination):
        if not self.opportunities:return None
        start=self.client.position;edge=sub(destination,start);length2=dot(edge,edge);options=[]
        for ch in sorted(self.discovered-self.cleared):
            t=self.tracks[ch]
            if t.radius<=SAFE_RADIUS:
                q=safe_detour_point(t.center,t.radius,start,destination)
                extra=dist(start,q)+dist(q,destination)-dist(start,destination)
                if extra<1e-5:options.append((-100.,'clear',ch,q,'safe_clear'))
                continue
            center=self.forecaster(ch).center
            f=max(0.,min(1.,dot(sub(center,start),edge)/length2)) if length2 else 0.
            points=[start]
            if .02<f<.98:points.append(add(start,mul(edge,f)))
            for p in points:
                benefit=self.information_benefit(ch,p)
                cost=5.+(ch!=self.client.channel)+(dist(start,p)+dist(p,destination)-dist(start,destination))/5
                if benefit>cost+2.:
                    options.append((cost-benefit,'measure',ch,p,'at_point_probe' if p==start else 'enroute_probe'))
        if options:
            _,kind,ch,p,tag=min(options);return kind,ch,p,tag
        if self.unknown_probes and len(self.probed_unknown.get(tuple(start),set()))<2 and self.unvisited:
            # At a source-service point, try only channels whose expected avoided
            # future station queries exceed this actual query's time cost.
            at_station=any(dist(start,q)<1e-7 for q in self.station_points)
            at_source=any(dist(start,self.tracks[c].center)<25. for c in self.cleared)
            if not at_station and at_source:
                posterior=self.belief.posterior(self.discovered);unknown=[]
                for ch in posterior['unknown']:
                    if any(dist(start,q)<5. for q in self.negative.get(ch,[])):continue
                    p=self.belief.discovery_probability(ch,start,posterior)
                    saving=p*(3.*len(self.unvisited)+20.)-(5.+(ch!=self.client.channel))
                    if saving>2.:unknown.append((saving,-ch,ch))
                if unknown:return 'measure',max(unknown)[2],start,'unknown_opportunity'
        return None

    def next_action(self,route,locations):
        # Near sources have a cheap, certified clear and are globally visible.
        for ch,p in sorted(self.near.items()):
            if ch not in self.cleared and dist(self.client.position,p)<1e-7:return 'clear',ch,p,'safe_clear'
        label=route[0];destination=locations[label]
        opportunity=self.opportunistic(destination)
        if opportunity:return opportunity
        kind,k=label
        if kind=='station':
            choices=self.pending_channels(k);posterior=self.belief.posterior(self.discovered)
            if self.channel_order:
                ch=max(choices,key=lambda c:(self.belief.discovery_probability(c,destination,posterior)/(5+(c!=self.client.channel)),c==self.client.channel,-c))
            else:ch=min(choices,key=lambda c:(c!=self.client.channel,c))
            # A known channel at the same station competes by its saved service time.
            known=[]
            for c in sorted(self.discovered-self.cleared):
                b=self.information_benefit(c,destination)
                if b>8.:known.append((b,c))
            if known:
                _,c=max(known);return 'measure',c,destination,'station_probe'
            return 'measure',ch,destination,'station_unknown'
        ch=k;t=self.tracks[ch]
        end=locations[route[1]] if len(route)>1 else t.center
        if t.radius<=SAFE_RADIUS:
            p=safe_detour_point(t.center,t.radius,self.client.position,end)
            return 'clear',ch,p,'safe_clear'
        if self.local_actions[ch]<8:
            options=self.action_candidates(ch,self.tried.get(ch,[]))
            if options:
                # Price how this observation/clear point changes the next route leg.
                best=min(options,key=lambda a:a[0]+.25*(dist(a[2],end)-dist(destination,end))/5)
                return best[1],ch,best[2],'local_service'
        if ch not in self.covers:
            self.covers[ch]=polygon_optical_cover(t.polygon);self.fallback_channels+=1
        candidates=self.covers[ch]
        if not candidates:raise ArithmeticError('Continuous optical cover exhausted')
        j=min(range(len(candidates)),key=lambda i:dist(self.client.position,candidates[i]))
        return 'clear',ch,candidates.pop(j),'optical_cover'

    def run(self):
        started=time.perf_counter();self.client.action('/enter');completed=False;reason='not_started'
        try:
            while True:
                self.refresh_certificate()
                if self.coverage_complete and self.discovered==self.cleared:
                    completed=True;reason='all_sources_cleared_with_certificate';break
                route,locations=self.plan()
                action=self.next_action(route,locations)
                # Leaving an unfinished station is allowed, but its channel ledger persists.
                here=next((i for i in self.unvisited if dist(self.station_points[i],self.client.position)<1e-7),None)
                if here is not None and dist(action[2],self.client.position)>1e-7:self.interrupted_stations+=1
                self.execute(action)
                if self.decisions>2500:raise ArithmeticError('Finite action safety bound exceeded')
        except (BudgetStop,ArithmeticError) as e:reason=str(e)
        self.client.action('/exit')
        result=dict(completed=completed,stop_reason=reason,virtual_time_s=self.client.virtual,
            runtime_s=time.perf_counter()-started,policy_version='action_replanning_20260913',
            stations_visited=self.stations_visited,visited_station_ids=self.visited_ids,
            discovery_certificate=('count_upper_bound' if len(self.discovered)==16 else 'convex_cell_cover') if self.coverage_complete else None,
            discovered_channels=sorted(self.discovered),cleared_channels=sorted(self.cleared),
            schedule_replans=self.schedule_replans,decisions=self.decisions,schedule_planning_s=self.schedule_planning_s,
            meta_evaluations=self.meta_evaluations,action_kinds=dict(self.action_kinds),
            interrupted_stations=self.interrupted_stations,fallback_channels=self.fallback_channels,
            budget_counts=dict(self.budget_counts),action_cost_parts=self.cost_parts,
            failure_recovery_samples=self.recovery_samples,
            fallback_attempts=sum(t.fallback_attempts for t in self.tracks.values()),
            prediction_metrics=self.prediction_metrics,depleted_target_banks=self.forecast_states_depleted,
            depth=self.depth,width=self.width,branch=self.branch)
        assert self.schedule_replans==self.decisions
        self.client.log('action_summary',**result);return result
