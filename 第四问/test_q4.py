"""Physical-rule, geometric-certificate and end-to-end regressions."""
import io
import json
import math
import random
import time
import unittest
from unittest.mock import patch
from client import Client,HttpTransport,ProtocolError
from simulator import LocalSimulator,Source,make_case
from strategy import Policy
from geometry import (ALPHA_DEG,DEFAULT_SIDE,coverage_stations,triangular_mesh,polygon_distance,
                      domain_polygon,incorporate,enclosing_circle,clip,optical_cover)

class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.sim=LocalSimulator([Source(2,300.,400.,1000.,None)])
        self.client=Client(self.sim); self.client.action('/enter')
    def test_time_and_clear_does_not_switch(self):
        self.client.action('/measure',(0.,0.),2) # 6
        self.client.action('/clear',(300.,400.),2) # 100+5
        self.assertEqual(self.client.virtual,111.)
        self.assertEqual(self.client.channel,2)
        self.client.action('/measure',(300.,400.),2) # +5
        self.assertEqual(self.client.virtual,116.)
    def test_clear_other_channel_does_not_switch(self):
        self.client.action('/clear',(0.,0.),20)
        self.assertEqual(self.client.channel,1)
        self.client.action('/measure',(0.,0.),1)
        self.assertEqual(self.client.virtual,8.)
    def test_same_location_same_error(self):
        a=self.client.action('/measure',(0.,0.),2)
        b=self.client.action('/measure',(0.,0.),2)
        self.assertEqual(a['svd_deg'],b['svd_deg'])
    def test_cleared_source_disappears(self):
        self.client.action('/clear',(300.,400.),2)
        self.assertEqual(self.client.action('/measure',(300.,400.),2)['measure_result'],'no_signal')
    def test_truth_hidden_until_exit(self):
        with self.assertRaises(RuntimeError): self.sim.evaluation()
    def test_idempotent_and_conflict(self):
        b={'arena_id':'default','robot_id':'SELF-Q4','request_id':'fixed','position':{'x':100,'y':100},'channel':3}
        first=self.sim.request('/measure',b)
        vt=self.sim.virtual
        self.assertEqual(first,self.sim.request('/measure',b))
        self.assertEqual(vt,self.sim.virtual)
        self.assertEqual(self.sim.request('/clear',b)[0],409)
    def test_bad_field_and_nan_do_not_move(self):
        b={'arena_id':'default','robot_id':'SELF-Q4','request_id':'bad','position':{'x':0,'y':0,'z':0},'channel':3}
        self.assertFalse(self.sim.request('/measure',b)[1]['accepted'])
        b['position']={'x':float('nan'),'y':0}
        self.assertEqual(self.sim.request('/measure',b)[0],400)
        self.assertEqual(self.sim.virtual,0)
    def test_closed_directional_boundary_and_backside_clear(self):
        s=LocalSimulator([Source(1,0.,0.,1000.,0.)])
        c=Client(s); c.action('/enter')
        self.assertEqual(c.action('/measure',(0.,1000.),1)['measure_result'],'direction')
        self.assertEqual(c.action('/measure',(-4.,0.),1)['measure_result'],'no_signal')
        self.assertEqual(c.action('/clear',(-20.,0.),1)['clear_result'],'success')
    def test_near_boundary_and_range(self):
        s=LocalSimulator([Source(1,0.,0.,1000.,None)]); c=Client(s); c.action('/enter')
        self.assertEqual(c.action('/measure',(5.,0.),1)['measure_result'],'near')
        self.assertEqual(c.action('/measure',(1000.001,0.),1)['measure_result'],'no_signal')
    def test_rejected_clock_not_adopted(self):
        self.client.action('/measure',(0.,0.),2)
        self.client.robot_id='wrong'
        with self.assertRaises(ProtocolError): self.client.action('/measure',(0.,0.),2)
        self.assertEqual(self.client.virtual,6.)

class GeometryTests(unittest.TestCase):
    def test_triangle_sizes(self):
        points,triangles=triangular_mesh()
        self.assertEqual(len(points),31); self.assertEqual(len(triangles),42)
        self.assertTrue(any(math.hypot(*p)>1800 for p in points))
        for tri in triangles:
            self.assertLessEqual(max(math.dist(a,b) for a in tri for b in tri),DEFAULT_SIDE+1e-6)
    def test_domain_covered_by_retained_triangles(self):
        _,triangles=triangular_mesh()
        for radius in (0,500,1200,1799,1800):
            for degree in range(360):
                a=math.radians(degree); p=(radius*math.cos(a),radius*math.sin(a))
                self.assertTrue(any(polygon_distance(p,list(t))<1e-5 for t in triangles))
    def test_all_orientations_via_angular_gap(self):
        # A largest circular gap <= pi proves nearby samples are not in one open halfplane.
        points=coverage_stations()
        for k in range(721):
            a=k*2*math.pi/721; g=(1800*math.cos(a),1800*math.sin(a))
            angles=sorted(math.atan2(p[1]-g[1],p[0]-g[0]) for p in points if math.dist(p,g)<=1000)
            gaps=[angles[(i+1)%len(angles)]-angles[i] for i in range(len(angles)-1)]
            gaps.append(angles[0]+2*math.pi-angles[-1])
            self.assertLessEqual(max(gaps),math.pi+1e-7)
    def test_optical_rectangle_cover(self):
        points=optical_cover((0.,0.),0.)
        h=1500*math.sin(math.radians(ALPHA_DEG))
        self.assertEqual(len(points),152)
        for x in range(0,1501,5):
            for y in (-h,-h/2,0,h/2,h):
                self.assertLess(min(math.dist((x,y),p) for p in points),20)
    def test_enclosing_circle_equilateral_diameter_is_not_enough(self):
        tri=[(0.,0.),(40.,0.),(20.,20*math.sqrt(3))]
        center,r=enclosing_circle(tri)
        self.assertAlmostEqual(r,40/math.sqrt(3),places=5)
        self.assertGreater(r,20.)
    def test_point_segment_empty_clipping(self):
        box=[(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)]
        line=clip(clip(box,(1.,0.),0.),(-1.,0.),0.)
        self.assertEqual(len(line),2)
        self.assertAlmostEqual(enclosing_circle(line)[1],1.,places=5)
        point=clip(clip(line,(0.,1.),0.),(0.,-1.),0.)
        self.assertEqual(len(point),1)
        self.assertEqual(clip(box,(1.,0.),-2.),[])
    def test_true_source_never_excluded_with_extreme_rounded_errors(self):
        rng=random.Random(712)
        for _ in range(150):
            a=rng.random()*2*math.pi; r=1800*math.sqrt(rng.random()); g=(r*math.cos(a),r*math.sin(a))
            poly=list(domain_polygon())
            for j in range(4):
                a=rng.random()*2*math.pi; r=rng.uniform(6,1500); p=(g[0]+r*math.cos(a),g[1]+r*math.sin(a))
                truth=math.degrees(math.atan2(g[1]-p[1],g[0]-p[0]))
                angle=round((truth+(-1 if j%2 else 1))%360,2)%360
                poly=incorporate(poly,p,angle)
                self.assertLess(polygon_distance(g,poly),1e-5)

class IntegrationTests(unittest.TestCase):
    def test_generators_respect_mixed_case_definition(self):
        for mode in ('mixed','edge_outward','minimum_range','cluster','near_origin'):
            for seed in range(10000,10050):
                sources=make_case(seed,mode)
                count=sum(s.orientation is not None for s in sources)
                self.assertTrue(1<=count<len(sources))
                self.assertTrue(10<=len(sources)<=16)
                self.assertEqual(len({s.channel for s in sources}),len(sources))
                self.assertTrue(all(math.hypot(s.x,s.y)<=1800.000001 and 1000<=s.radius<=1500 for s in sources))
    def test_complete_adversarial_cases_and_clock(self):
        for mode,error in [('mixed','fixed_hash'),('edge_outward','plus_one'),('all_directional','minus_one'),
                           ('cluster','checker'),('minimum_range','smooth'),('near_origin','zero')]:
            with self.subTest(mode=mode,error=error):
                s=LocalSimulator(make_case(55,mode),55,error)
                r=Policy(Client(s)).run(); e=s.evaluation()
                self.assertTrue(r['completed']); self.assertEqual(e['clear_ratio'],1.)
                reconstructed=e['distance_m']/5+5*e['measures']+e['switches']+5*e['cleared']+3*e['clear_failures']
                self.assertAlmostEqual(e['virtual_time_s'],reconstructed,places=6)
    def test_fallback_alone_clears(self):
        s=LocalSimulator(make_case(62,'edge_outward'),62,'plus_one')
        r=Policy(Client(s),active=False).run()
        self.assertTrue(r['completed']); self.assertEqual(s.evaluation()['clear_ratio'],1.)
        self.assertLessEqual(r['fallback_attempts'],152*16)
    def test_budget_stop_is_not_success(self):
        s=LocalSimulator(make_case(1),1)
        original=s.request
        class ShortBudget:
            def request(self,path,body):
                status,reply=original(path,body)
                if path=='/enter': reply['remaining_real_duration_s']=0
                return status,reply
        r=Policy(Client(ShortBudget())).run()
        self.assertFalse(r['completed']); self.assertEqual(r['stop_reason'],'real_time_reserve')
    def test_invalid_mesh_cannot_claim_coverage(self):
        with self.assertRaises(ValueError): Policy(Client(LocalSimulator(make_case(0))),side=1001.)
    def test_model_failure_exits_without_claiming_success(self):
        s=LocalSimulator([])
        r=Policy(Client(s)).run()
        self.assertFalse(r['completed']); self.assertTrue(r['stop_reason'].startswith('model_or_geometry_error'))
        self.assertTrue(s.exited)
    def test_http_retry_preserves_payload(self):
        calls=[]
        class Reply:
            status=200
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def read(self): return b'{"accepted":true}'
        def fake(req,timeout):
            calls.append(req.data)
            if len(calls)==1: raise TimeoutError('lost response')
            return Reply()
        with patch('urllib.request.urlopen',fake):
            self.assertEqual(HttpTransport().request('/enter',{'request_id':'same'})[0],200)
        self.assertEqual(calls[0],calls[1])

if __name__=='__main__': unittest.main(verbosity=2)
