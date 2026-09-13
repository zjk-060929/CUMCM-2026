"""Serial JSON client. Offline transport and HTTP transport share the policy API."""
from __future__ import annotations
import json
import math
import socket
import time
import urllib.error
import urllib.request
import uuid

class ProtocolError(RuntimeError): pass
class BudgetStop(RuntimeError): pass

class HttpTransport:
    def __init__(self,url='http://127.0.0.1:2027',timeout=3.,attempts=2):
        from urllib.parse import urlparse
        p=urlparse(url)
        if p.scheme!='http' or p.hostname not in ('127.0.0.1','localhost','::1') or p.path not in ('','/') or p.query or p.fragment:
            raise ValueError('Only a local simulator root URL is supported')
        self.url,self.timeout,self.attempts=url.rstrip('/'),timeout,attempts

    def request(self,path,body):
        # Serialize once: retries preserve the exact action and request ID.
        payload=json.dumps(body,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
        for attempt in range(self.attempts):
            req=urllib.request.Request(self.url+path,data=payload,headers={'Content-Type':'application/json'},method='POST')
            try:
                with urllib.request.urlopen(req,timeout=self.timeout) as r:
                    return r.status,json.loads(r.read())
            except urllib.error.HTTPError as e:
                try: reply=json.loads(e.read())
                except ValueError: reply={'accepted':False}
                return e.code,reply
            except (urllib.error.URLError,TimeoutError,socket.timeout,ConnectionError,OSError) as e:
                if attempt+1 == self.attempts:
                    raise ProtocolError(f'Connection uncertain after identical retries; stop new actions: {e}') from e
        raise AssertionError('Unreachable')

class Client:
    def __init__(self,transport,robot_id='SELF-Q4',log_path=None):
        # Store a callable only; the policy never receives simulator truth.
        self._request=transport.request
        self.robot_id,self.prefix=robot_id,uuid.uuid4().hex[:12]
        self.counter=0; self.position=(0.,0.); self.channel=1; self.virtual=0.
        self.deadline=float('inf'); self.virtual_limit=360000.
        self.log_file=open(log_path,'w',encoding='utf8') if log_path else None
        self.entered=self.exited=False

    def log(self,event,**data):
        if self.log_file:
            self.log_file.write(json.dumps({'event':event,**data},ensure_ascii=False,allow_nan=False)+'\n')
            self.log_file.flush()

    def action(self,path,position=None,channel=None):
        if position is not None:
            position=tuple(float(x) for x in position)
            if len(position)!=2 or any(not math.isfinite(x) or abs(x)>2e6 for x in position): raise ValueError('Invalid position')
            predicted=self.virtual+math.dist(self.position,position)/5+5+(path=='/measure' and channel!=self.channel)
            if predicted>=self.virtual_limit-1: raise BudgetStop('virtual_time_reserve')
            if time.monotonic()>=self.deadline-30: raise BudgetStop('real_time_reserve')
        self.counter+=1
        body={'arena_id':'default','robot_id':self.robot_id,'request_id':f'{self.prefix}-{self.counter}'}
        if position is not None: body.update(position={'x':position[0],'y':position[1]},channel=channel)
        started=time.monotonic()
        self.log('request',path=path,body=body)
        status,reply=self._request(path,body)
        self.log('response',path=path,status=status,body=reply)
        if status!=200 or reply.get('accepted') is not True:
            raise ProtocolError(f'{path}: HTTP {status}, accepted={reply.get("accepted")}')
        vt=reply.get('virtual_time_s')
        if isinstance(vt,bool) or not isinstance(vt,(int,float)) or not math.isfinite(vt) or vt+1e-5<self.virtual:
            raise ProtocolError('Invalid accepted virtual clock')
        if path=='/measure':
            result=reply.get('measure_result')
            if result not in ('direction','near','no_signal'): raise ProtocolError('Invalid measurement result')
            if result=='direction':
                angle=reply.get('svd_deg')
                if isinstance(angle,bool) or not isinstance(angle,(int,float)) or not math.isfinite(angle) or not 0<=angle<360:
                    raise ProtocolError('Invalid bearing')
            self.channel=channel
        if path=='/clear' and reply.get('clear_result') not in ('success','no_target_in_range'):
            raise ProtocolError('Invalid clear result')
        self.virtual=float(vt)
        if position is not None: self.position=position
        if path=='/enter':
            remaining=reply.get('remaining_real_duration_s')
            limit=reply.get('max_virtual_duration_s')
            if not isinstance(remaining,(int,float)) or not math.isfinite(remaining) or not 0<=remaining<=1200:
                raise ProtocolError('Missing/invalid remaining real duration')
            if not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit<=0:
                raise ProtocolError('Missing/invalid virtual limit')
            self.deadline=started+remaining; self.virtual_limit=limit; self.entered=True
        if path=='/exit': self.exited=True
        return reply

    def close(self):
        if self.log_file: self.log_file.close(); self.log_file=None
