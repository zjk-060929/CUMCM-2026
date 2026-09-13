"""Geometry and control-flow contracts, including adversarial interrupted scans."""
import math
import unittest
from action_strategy import ActionPolicy,CoverForecast,optical_tour
from benchmark_action import run_one
from geometry import dist,enclosing_circle
from optimized_geometry import polygon_optical_cover
from predictive_strategy_q4 import PredictiveClient
from simulator import Source
from spatial_error_q4 import SpatialErrorSimulator
from strategy import Track


class Contracts(unittest.TestCase):
    def test_near_requires_separate_decision(self):
        sim=SpatialErrorSimulator([Source(1,2.,0.,1000.,None)],seed=12)
        client=PredictiveClient(sim);policy=ActionPolicy(client)
        client.action('/enter')
        policy.execute(('measure',1,(0.,0.),'station_unknown'))
        self.assertEqual(client.counter,2)
        self.assertNotIn(1,policy.cleared)
        action=policy.next_action([('task',1)],{('task',1):(0.,0.)})
        self.assertEqual(action,('clear',1,(0.,0.),'safe_clear'))
        policy.execute(action)
        self.assertEqual(client.counter,3)
        self.assertEqual(policy.decisions,2)
        client.action('/exit');client.close()

    def test_partial_station_is_not_coverage(self):
        sim=SpatialErrorSimulator([]);client=PredictiveClient(sim);policy=ActionPolicy(client)
        i=policy.unvisited[0];policy.tested[i]=set(range(1,10))
        policy.refresh_certificate()
        self.assertIn(i,policy.unvisited)
        self.assertEqual(policy.pending_channels(i),set(range(10,21)))
        # A discovered channel no longer needs an absence query at this station.
        policy.discovered.update((10,11));policy.refresh_certificate()
        self.assertEqual(policy.pending_channels(i),set(range(12,21)))
        policy.tested[i].update(range(12,21));policy.refresh_certificate()
        self.assertNotIn(i,policy.unvisited)

    def test_no_small_step_bearing_candidate(self):
        sim=SpatialErrorSimulator([]);client=PredictiveClient(sim);policy=ActionPolicy(client)
        t=Track(polygon=[(100.,-10.),(500.,-10.),(500.,10.),(100.,10.)],observations=[((0.,0.),0.)])
        t.center,t.radius=enclosing_circle(t.polygon);policy.tracks[1]=t
        for p in ((0.,0.),(3.,0.),(5.,0.),(49.,0.)):
            self.assertFalse(policy.valid_probe(1,p))
        self.assertTrue(policy.valid_probe(1,(0.,100.)))

    def test_unknown_channel_can_compete_at_a_clear_point(self):
        client=PredictiveClient(SpatialErrorSimulator([]))
        policy=ActionPolicy(client,unknown_probes=True)
        client.position=(100.,0.);policy.cleared.add(1);policy.discovered.add(1)
        policy.tracks[1]=Track(center=(100.,0.),radius=10.,cleared=True)
        action=policy.opportunistic((900.,0.))
        self.assertIsNotNone(action)
        self.assertEqual(action[0],'measure');self.assertEqual(action[3],'unknown_opportunity')
        self.assertEqual(action[2],client.position)
        self.assertNotEqual(action[1],1)

    def test_continuous_cover_contains_long_thin_and_rotated_regions(self):
        # Grid testing supplements the rectangle half-diagonal proof in the code.
        for width,height,angle in ((1500.,.001,0.),(620.,46.,.7),(38.,38.,1.4)):
            c,s=math.cos(angle),math.sin(angle)
            rotate=lambda x,y:(c*x-s*y,s*x+c*y)
            poly=[rotate(x,y) for x,y in ((0.,0.),(width,0.),(width,height),(0.,height))]
            cover=polygon_optical_cover(poly)
            for i in range(201):
                for j in range(5):
                    p=rotate(width*i/200,height*j/4)
                    self.assertLessEqual(min(dist(p,q) for q in cover),19.8+1e-7)
            route,mean,worst=optical_tour(cover,(-100.,0.))
            self.assertEqual(len(set(route)),len(cover));self.assertLessEqual(mean,worst)

    def test_tail_risk_does_not_collapse_to_24_samples(self):
        t=Track(polygon=[(200.,-15.),(1450.,-15.),(1450.,15.),(200.,15.)],observations=[((0.,0.),0.)])
        t.center,t.radius=enclosing_circle(t.polygon)
        bank=CoverForecast(t,[],[],t.center,1.5)
        self.assertGreater(len(bank.cover),len(bank.points))
        p=bank.clear_prediction(t.center)
        self.assertGreaterEqual(p['cover_worst_s'],p['cover_mean_s'])
        self.assertGreater(p['operations'],5.)


if __name__=='__main__':unittest.main()
