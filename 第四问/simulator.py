"""Independent local physical emulator, NOT the official competition simulator.

The strategy receives only request(path, body) -> (status, JSON).
Truth is exported separately by the experiment harness after /exit.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

@dataclass(frozen=True)
class Source:
    channel: int
    x: float
    y: float
    radius: float
    orientation: float | None

def make_case(seed, mode='mixed'):
    rng = random.Random(seed)
    n = 10 + seed % 7
    if mode == 'edge_outward': n = 16
    if mode == 'cluster': n = 10
    channels = rng.sample(range(1,21),n)
    directional_count = rng.randint(1,n-1)
    ans = []
    for k,ch in enumerate(channels):
        angle = rng.uniform(0,2*math.pi)
        radius = 1800*math.sqrt(rng.random())
        x,y = radius*math.cos(angle), radius*math.sin(angle)
        reach = rng.uniform(1000,1500)
        ori = rng.uniform(0,360) if k < directional_count else None
        if mode in ('edge_outward','all_directional'):
            ori = math.degrees(angle) % 360
            reach = 1000.
            if mode == 'edge_outward':
                radius = 1800. if k%2==0 else 1799.
                x,y = radius*math.cos(angle),radius*math.sin(angle)
                if k == n-1: ori = None  # still a legal mixed Q4 case
        if mode == 'all_omni': ori = None
        if mode == 'cluster':
            x,y = 1100+rng.uniform(-25,25),500+rng.uniform(-25,25)
            reach = 1000.
        if mode == 'minimum_range': reach = 1000.
        if mode == 'near_origin' and k == 0: x,y,ori = 2.,0.,180.
        ans.append(Source(ch,x,y,reach,ori))
    return ans

class LocalSimulator:
    def __init__(self, sources, seed=0, error_mode='fixed_hash', robot_id='SELF-Q4', capture=False):
        self.__sources = {s.channel:s for s in sources}
        self.seed, self.error_mode, self.robot_id = seed,error_mode,robot_id
        self.capture, self.log = capture,[]
        self.position, self.channel, self.virtual = (0.,0.),1,0.
        self.entered, self.exited = False,False
        self.__cleared, self.__cache = set(),{}
        self.distance = 0.
        self.measures = self.switches = self.clear_success = self.clear_fail = 0

    def error(self, position):
        x,y = position
        if self.error_mode == 'plus_one': return 1.
        if self.error_mode == 'minus_one': return -1.
        if self.error_mode == 'smooth': return math.sin(x/170+y/230+self.seed)
        if self.error_mode == 'checker': return 1. if (math.floor(x/40)+math.floor(y/40))%2 else -1.
        if self.error_mode == 'zero': return 0.
        # Deterministic per position across all channels, no repeated-read averaging.
        key = f'{self.seed}:{x:.9f}:{y:.9f}'.encode()
        value = int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'big')
        return 2*value/(2**64-1)-1

    def _reply(self, accepted, **kwargs):
        return {'accepted':accepted,'real_timestamp_ms':int(time.time()*1000),
                'virtual_time_s':round(self.virtual,6) if accepted else 0, **kwargs}

    def request(self, path, body):
        base = {'arena_id','robot_id','request_id'}
        if path not in ('/enter','/measure','/clear','/exit'): return 404,self._reply(False)
        expected = base | ({'position','channel'} if path in ('/measure','/clear') else set())
        if not isinstance(body,dict) or not expected <= set(body): return 400,self._reply(False)
        if set(body)-expected: return 200,self._reply(False)
        for field,limit in (('arena_id',128),('robot_id',64),('request_id',128)):
            v = body[field]
            if not isinstance(v,str) or not v or len(v.encode())>limit or any(ord(c)<32 for c in v):
                return 400,self._reply(False)
        if body['arena_id']!='default' or body['robot_id']!=self.robot_id: return 200,self._reply(False)
        if path in ('/measure','/clear'):
            c,p = body['channel'],body['position']
            if isinstance(c,bool) or not isinstance(c,(int,float)) or not math.isfinite(c) or int(c)!=c or not 1<=c<=20:
                return 400,self._reply(False)
            if not isinstance(p,dict) or not {'x','y'}<=set(p): return 400,self._reply(False)
            if set(p)-{'x','y'}: return 200,self._reply(False)
            if any(isinstance(p[k],bool) or not isinstance(p[k],(int,float)) or not math.isfinite(p[k]) or abs(p[k])>2e6 for k in ('x','y')):
                return 400,self._reply(False)
        rid = body['request_id']
        signature = (path,json.dumps(body,sort_keys=True,allow_nan=False))
        if rid in self.__cache:
            sig,response = self.__cache[rid]
            return (200,copy.deepcopy(response)) if sig==signature else (409,self._reply(False))
        if path=='/enter':
            if self.entered or self.exited: return 200,self._reply(False)
            self.entered = True
            response = self._reply(True,max_virtual_duration_s=360000,max_real_duration_s=1200,remaining_real_duration_s=1200)
        elif not self.entered or self.exited: return 200,self._reply(False)
        elif path=='/exit':
            self.exited=True; response=self._reply(True,exit_reason='user_exit')
        else:
            target = (float(body['position']['x']),float(body['position']['y']))
            distance = math.dist(target,self.position)
            self.distance += distance
            self.virtual += distance/5
            self.position = target
            source = self.__sources.get(int(body['channel']))
            if body['channel'] in self.__cleared: source=None
            d = math.hypot(target[0]-source.x,target[1]-source.y) if source else float('inf')
            if path=='/clear':
                success = d <= 20. + 1e-9
                self.virtual += 5 if success else 3
                if success: self.__cleared.add(source.channel); self.clear_success+=1
                else: self.clear_fail+=1
                response = self._reply(True,clear_result='success' if success else 'no_target_in_range')
            else:
                switch = self.channel != body['channel']
                self.switches += int(switch); self.measures+=1
                self.virtual += 5 + switch
                self.channel = int(body['channel'])
                detectable = source is not None and d <= source.radius+1e-9
                if detectable and source.orientation is not None:
                    a=math.radians(source.orientation)
                    detectable = math.cos(a)*(target[0]-source.x)+math.sin(a)*(target[1]-source.y) >= -1e-9
                if not detectable: response=self._reply(True,measure_result='no_signal')
                elif d <= 5.+1e-9: response=self._reply(True,measure_result='near')
                else:
                    angle = math.degrees(math.atan2(source.y-target[1],source.x-target[0]))
                    response=self._reply(True,measure_result='direction',svd_deg=round((angle+self.error(target))%360,2)%360)
        self.__cache[rid] = signature,copy.deepcopy(response)
        if self.capture: self.log.append({'path':path,'request':copy.deepcopy(body),'response':copy.deepcopy(response)})
        return 200,response

    def evaluation(self):
        if not self.exited: raise RuntimeError('Truth may only be exported after exit')
        n=len(self.__sources)
        return {'source_count':n,'directional_count':sum(s.orientation is not None for s in self.__sources.values()),
                'cleared':self.clear_success,'clear_ratio':self.clear_success/n,
                'virtual_time_s':self.virtual,'time_per_source_s':self.virtual/self.clear_success if self.clear_success else None,
                'distance_m':self.distance,'measures':self.measures,'switches':self.switches,
                'clear_failures':self.clear_fail,'sources':[asdict(s) for s in self.__sources.values()]}

def serve(seed=0, port=2027):
    sim = LocalSimulator(make_case(seed),seed,capture=True)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args): pass
        def do_POST(self):
            try:
                data=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
                status,reply=sim.request(self.path,data)
            except (ValueError,TypeError): status,reply=400,sim._reply(False)
            payload=json.dumps(reply).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
    print(f'SELF SIMULATOR ONLY: http://127.0.0.1:{port}, robot_id=SELF-Q4',flush=True)
    HTTPServer(('127.0.0.1',port),Handler).serve_forever()

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--port',type=int,default=2027)
    a=ap.parse_args(); serve(a.seed,a.port)
