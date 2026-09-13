import math
from types import SimpleNamespace
import unittest

from integrated_strategy import IntegratedPolicy, IntegratedModel
from refined_geometry import source_domain, incorporate_fine
from geometry import polygon_distance
from boundary_geometry import arc_probe_points


class IntegrationTests(unittest.TestCase):
    def test_1024_circles_contain_their_disks(self):
        poly=source_domain()
        self.assertEqual(len(poly),1024)
        self.assertLessEqual(max(math.hypot(*p) for p in poly)-1800,.008472)
        for angle in range(0,360,7):
            a=math.radians(angle);g=(1800*math.cos(a),1800*math.sin(a))
            self.assertLess(polygon_distance(g,poly),1e-6)
        p=(-700.,0.);g=(799.99,0.)
        located=incorporate_fine(list(poly),p,0.)
        self.assertLess(polygon_distance(g,located),1e-6)
        self.assertLessEqual(max(math.dist(x,p) for x in located)-1500,.00706)

    def test_boundary_candidate_preserves_input(self):
        poly=[(1700.,-100.),(1850.,-100.),(1850.,100.),(1700.,100.)]
        before=list(poly);points=arc_probe_points(poly,(1864.,0.),1)
        self.assertEqual(poly,before)
        self.assertEqual(len(points),1)
        self.assertAlmostEqual(math.hypot(*points[0]),1800.)
        self.assertLess(polygon_distance(points[0],poly),1e-6)

    def test_planning_does_not_consume_or_activate_optical_cover(self):
        policy=object.__new__(IntegratedPolicy)
        policy.boundary=False;policy.boundary_done=set();policy.cleared=set()
        policy.client=SimpleNamespace(position=(0.,0.))
        policy.tracks={1:SimpleNamespace(radius=100.,center=(100.,0.),polygon=[(0.,-5.),(200.,-5.),(200.,5.),(0.,5.)])}
        policy.local_actions={1:8};policy.covers={};policy.fallback_channels=0
        policy.forecaster=lambda ch:SimpleNamespace(cover_cost=lambda p:(50.,100.))
        first=policy.service_action(1);second=policy.service_action(1)
        self.assertEqual(first,second)
        self.assertEqual(policy.covers,{})
        self.assertEqual(policy.fallback_channels,0)

    def test_last_discovery_risk_cannot_cancel_late_completion(self):
        model=object.__new__(IntegratedModel)
        model.p=dict(unknown=[1,2,3,4,5],probability_n={16:1.},coefficients=[1.,1.],
                     likelihood={1:1.,2:0.,3:0.,4:0.,5:0.})
        model.sizes={c:1 for c in model.p['unknown']}
        model.details=lambda route:(0.,0.,0.,[(route,{c:1 for c in model.p['unknown']})])
        self.assertAlmostEqual(model.last_discovery_charge(10.,20.),10.)
        self.assertAlmostEqual(model.last_discovery_charge(20.,10.),0.)

    def test_no_sensing_disables_its_route_value(self):
        policy=object.__new__(IntegratedPolicy)
        policy.opportunities=False;policy.optional_count=0
        self.assertEqual(policy.information_benefit(1,(0.,0.)),-math.inf)


if __name__=='__main__':unittest.main()
