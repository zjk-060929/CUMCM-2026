"""Predict discovery latency, sensing outcomes, and failed-clear recovery."""
import math
import time

from bayesian_belief import DiscoveryBelief,detection_mask
from client import Client
from geometry import dist,add,sub,mul,unit,polygon_distance,clipped_optical_cover,enclosing_circle
from heuristic_search import optimize
from optimized_geometry import SAFE_RADIUS,polygon_optical_cover
from planned_strategy import insert_tasks
from predictive_models_q4 import TargetForecast,ForecastRouteModel,cap_samples
from refined_geometry import incorporate_fine
from spatial_error_q4 import SpatialBeamPolicy

# Selected on development layouts only; frozen before independent validation.
PREDICTIVE_CONFIG=dict(discovery=True,sensing=True,failure=True,delay_weight=.08,
                       recovery_scale=1.5,clear_probability_floor=0.)


class PredictiveClient(Client):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.before_action=self.after_action=None
        self.first_discovery={}
    def action(self,path,p=None,ch=None):
        context=self.before_action(path,p,ch) if self.before_action else None
        reply=super().action(path,p,ch)
        positive=(path=='/measure' and reply.get('measure_result') in ('direction','near'))
        positive|=path=='/clear' and reply.get('clear_result')=='success'
        if positive:self.first_discovery.setdefault(ch,self.virtual)
        if self.after_action:self.after_action(path,p,ch,reply,context)
        return reply


class PredictivePolicy(SpatialBeamPolicy):
    def __init__(self,client,*,discovery=True,sensing=True,failure=True,delay_weight=.08,
                 recovery_scale=1.,clear_probability_floor=0.):
        super().__init__(client)
        self.use_discovery,self.use_sensing,self.use_failure=discovery,sensing,failure
        self.delay_weight=delay_weight
        self.recovery_scale,self.clear_probability_floor=recovery_scale,clear_probability_floor
        self.belief=DiscoveryBelief();self.forecast_cache={}
        self.forecast_actions=self.forecast_states_depleted=0
        self.prediction_metrics={}
        self.discovery_forecasts=[]
        self.current_episode=None;self.episode_count=0;self.recovery_pending={}
        client.before_action=self.before_action
        client.after_action=self.after_action

    def record(self,key,predicted,actual):
        item=self.prediction_metrics.setdefault(key,dict(n=0,predicted_sum=0.,actual_sum=0.,squared_error=0.,absolute_error=0.))
        item['n']+=1;item['predicted_sum']+=predicted;item['actual_sum']+=actual
        item['squared_error']+=(predicted-actual)**2;item['absolute_error']+=abs(predicted-actual)

    def forecaster(self,ch):
        t=self.tracks[ch]
        key=ch,len(t.observations),len(self.negative.get(ch,[])),len(self.failed_clear.get(ch,[]))
        if key not in self.forecast_cache:
            # Keep one bank per channel; each version is invalidated by feedback.
            old=[k for k in self.forecast_cache if k[0]==ch]
            for k in old:del self.forecast_cache[k]
            bank=TargetForecast(t,self.negative.get(ch,[]),self.failed_clear.get(ch,[]),self.estimate(ch)[0],self.recovery_scale)
            self.forecast_states_depleted+=int(bank.depleted)
            self.forecast_cache[key]=bank
        return self.forecast_cache[key]

    def before_action(self,path,p,ch):
        if path=='/measure' and ch not in self.discovered:
            posterior=self.belief.posterior(self.discovered)
            return dict(unknown=True,probability=self.belief.discovery_probability(ch,p,posterior))
        if path in ('/measure','/clear') and ch in self.tracks and ch not in self.cleared:
            bank=self.forecaster(ch)
            if path=='/measure':
                prediction=bank.measurement_prediction(p,self.min_bearing_baseline)
                if prediction['valid']:
                    return dict(unknown=False,prediction=prediction,old_radius=self.tracks[ch].radius,
                                old_polygon=list(self.tracks[ch].polygon))
            elif self.tracks[ch].radius>SAFE_RADIUS:
                return dict(prediction=bank.clear_prediction(p),episode=self.current_episode)
        return None

    def after_action(self,path,p,ch,reply,context):
        if path=='/measure' and context:
            result=reply['measure_result']
            received=result!='no_signal'
            if context.get('unknown'):
                self.record('unknown_reception_probability',context['probability'],float(received))
                if not received:self.belief.negative(ch,p)
            else:
                forecast=context['prediction'];old=context['old_radius']
                self.record('known_reception_probability',forecast['reception'],float(received))
                actual=old
                if result=='direction':
                    _,actual=enclosing_circle(incorporate_fine(context['old_polygon'],p,reply['svd_deg']))
                elif result=='near':actual=0.
                self.record('remaining_radius_fraction',forecast['radius']/max(old,1.),actual/max(old,1.))
        elif path=='/clear':
            success=reply['clear_result']=='success'
            if context:
                forecast=context['prediction']
                self.record('clear_success_probability',forecast['success'],float(success))
                if not success and context['episode'] is not None:
                    self.recovery_pending.setdefault(ch,(context['episode'],self.client.virtual,forecast['failure_recovery']))
            if success and ch in self.recovery_pending:
                episode,start,prediction=self.recovery_pending.pop(ch)
                if self.current_episode==episode:
                    self.record('same_episode_failure_recovery_s',prediction,self.client.virtual-start)

    def predicted_station_effects(self,ch,center):
        if not self.use_sensing:return super().predicted_station_effects(ch,center)
        bank=self.forecaster(ch)
        return {i:(bank.measurement_prediction(q,self.min_bearing_baseline)['radius'],
                   polygon_distance(q,self.tracks[ch].polygon)<=1500.) for i,q in enumerate(self.station_points)}

    def plan(self):
        started=time.perf_counter()
        stations=[('station',i) for i in self.unvisited]
        tasks=[('task',ch) for ch in sorted(self.discovered-self.cleared)]
        labels=stations+tasks
        positions={k:self.station_points[k[1]] for k in stations};blocked=set()
        for k in tasks:
            t=self.tracks[k[1]]
            positions[k]=t.center if t.radius<=SAFE_RADIUS else self.estimate(k[1])[0]
            if t.radius>SAFE_RADIUS and k[1] in self.scan_attempted:blocked.add(k)
        initial=insert_tasks(stations,[k for k in tasks if k not in blocked],positions,self.client.position)+sorted(blocked)
        index={k:i for i,k in enumerate(labels)};effects={};service_costs={}
        for k in tasks:
            future=self.predicted_station_effects(k[1],positions[k])
            for station in stations:
                radius,possible=future[station[1]]
                effects.setdefault(index[station],[]).append((index[k],radius,possible))
            if self.use_failure:
                service_costs[index[k]]=self.forecaster(k[1]).clear_prediction(positions[k])['operations']
        posterior=self.belief.posterior(self.discovered) if self.use_discovery else None
        model=ForecastRouteModel(labels,[positions[k] for k in labels],self.client.position,
            {index[k]:self.tracks[k[1]].radius for k in tasks},effects,{index[k] for k in blocked},
            uncertainty_weight=.35,unknown_channels=20-len(self.discovered),posterior=posterior,
            station_masks={index[k]:detection_mask(positions[k]) for k in stations},
            delay_weight=self.delay_weight,service_costs=service_costs)
        warm=[index[k] for k in self.last_plan if k in index and k not in blocked]
        warm += [index[k] for k in self.last_plan if k in blocked]
        order,stats=optimize(model,[index[k] for k in initial],method='beam',seed=8137+self.schedule_replans,warm=warm)
        result=[labels[i] for i in order]
        self.unvisited=[k[1] for k in result if k[0]=='station'];self.last_plan=result
        self.schedule_replans+=1;self.schedule_planning_s+=time.perf_counter()-started
        self.meta_evaluations+=stats['cost_evaluations'];self.meta_proxy_savings+=stats['initial_cost']-stats['final_cost']
        model.cost(order)
        if model.last_forecast:
            self.discovery_forecasts.append(dict(time_s=self.client.virtual,
                unknown_channels=sorted(set(range(1,21))-self.discovered),**model.last_forecast))
        self.client.log('predictive_route',order=result,forecast=model.last_forecast,**stats)
        return result

    def action_candidates(self,ch,tried,allow_measure=True,allow_clear=True):
        t=self.tracks[ch];bank=self.forecaster(ch);center=bank.center
        last,bearing=t.observations[-1]
        length=dist(last,center);toward=mul(sub(last,center),1/length) if length else unit(bearing+180)
        v=unit(bearing+90);base=add(center,mul(toward,min(120.,max(40.,self.estimate(ch)[1]*.35))))
        measure_points=[base]+[add(base,mul(v,s)) for s in (50.,-50.,100.,-100.,180.,-180.)]
        measure_points += [add(center,mul(v,s)) for s in (80.,-80.)]
        if self.unvisited:
            measure_points += sorted([self.station_points[i] for i in self.unvisited],key=lambda p:dist(p,center))[:2]
        clear_points=[center]
        if bank.points:
            clear_points += [min(bank.points,key=lambda p:dist(p,self.client.position)),
                             max(bank.points,key=lambda p:sum(dist(p,q)<=20 for q in bank.points))]
        actions=[]
        if allow_clear:
            for p in dict.fromkeys(clear_points):
                if any(dist(p,q)<=20 for q in self.failed_clear.get(ch,[])) or any(dist(p,q)<5 for q in tried):continue
                prediction=bank.clear_prediction(p)
                if self.use_failure:
                    if prediction['success']<self.clear_probability_floor:continue
                    score=dist(self.client.position,p)/5+prediction['operations']
                elif p==center and self.estimate(ch)[1]<=80.:
                    score=dist(self.client.position,p)/5+5.
                else:continue
                actions.append((score,'clear',p,prediction))
        if allow_measure:
            previous=[p for p,_ in t.observations]+self.negative.get(ch,[])+tried
            for p in dict.fromkeys(measure_points):
                # Deliberate active probes use >=50m, still with the full angle bound.
                if any(dist(p,q)<50 for q,_ in t.observations) or any(dist(p,q)<5 for q in previous):continue
                prediction=bank.measurement_prediction(p,self.min_bearing_baseline)
                if not prediction['valid']:continue
                score=dist(self.client.position,p)/5+6.+prediction['remaining']
                actions.append((score,'measure',p,prediction))
        return sorted(actions,key=lambda a:(a[0],a[1],a[2]))

    def predictive_service(self,ch,steps,early):
        t=self.tracks[ch];tried=[];measures=clears=0
        self.episode_count+=1;self.current_episode=self.episode_count
        start=self.client.virtual;first_prediction=None
        try:
            for _ in range(2*steps):
                if t.cleared:break
                if t.radius<=SAFE_RADIUS:
                    self.certified_clear(ch);break
                options=self.action_candidates(ch,tried,measures<steps,clears<steps)
                if not options:break
                score,kind,p,prediction=options[0]
                if first_prediction is None:first_prediction=score
                self.forecast_actions+=1
                self.client.log('predictive_local_action',channel=ch,kind=kind,position=p,
                                expected_remaining_s=score,prediction=prediction)
                tried.append(p)
                if kind=='clear':
                    clears+=1;self.optimistic_clears+=1
                    self.clear(p,ch)
                else:
                    measures+=1;t.active_queries+=1
                    if self.measure(p,ch)=='no_signal':self.no_signal_localization+=1
            if not t.cleared and t.radius<=SAFE_RADIUS:self.certified_clear(ch)
            if not t.cleared and not early:
                self.fallback_channels+=1
                first,bearing=t.observations[0]
                candidates=(polygon_optical_cover(t.polygon) if self.optical=='polygon' else clipped_optical_cover(first,bearing,t.polygon))
                while candidates:
                    k=min(range(len(candidates)),key=lambda i:(dist(candidates[i],self.client.position),i))
                    p=candidates.pop(k);t.fallback_attempts+=1
                    if self.clear(p,ch):break
                if not t.cleared:raise ArithmeticError('Certified optical coverage exhausted')
            if first_prediction is not None and t.cleared:
                self.record('completed_service_episode_s',first_prediction,self.client.virtual-start)
        finally:
            self.current_episode=None
            self.recovery_pending.pop(ch,None)
        return t.cleared

    def bounded_scan_service(self,ch):
        if not (self.use_sensing or self.use_failure):return super().bounded_scan_service(ch)
        self.scan_attempted.add(ch)
        if not self.predictive_service(ch,self.scan_steps,True):self.bounded_deferred+=1

    def service(self,ch):
        if not (self.use_sensing or self.use_failure):return super().service(ch)
        self.predictive_service(ch,4,False)

    def run(self):
        result=super().run()
        result.update(policy_version='q4_three_cost_predictions',prediction_discovery=self.use_discovery,
                      prediction_sensing=self.use_sensing,prediction_failure=self.use_failure,
                      recovery_scale=self.recovery_scale,clear_probability_floor=self.clear_probability_floor,
                      delay_weight=self.delay_weight,prediction_actions=self.forecast_actions,
                      depleted_target_banks=self.forecast_states_depleted,prediction_metrics=self.prediction_metrics)
        self.client.log('prediction_summary',**result)
        return result
