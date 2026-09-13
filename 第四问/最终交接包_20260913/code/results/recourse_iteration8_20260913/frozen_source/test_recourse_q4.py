"""Retry insertion, bounded penalties and local station-order invariants."""
import unittest

from benchmark_recourse_q4 import run_one
from recourse_strategy import RecourseModel,head_candidates


def example_model(**options):
    return RecourseModel([('task',1),('station',0),('station',1),('task',2)],
             [(0.,0.),(100.,0.),(-100.,0.),(1000.,0.)],(0.,0.),{0:100.,3:1.},{},set(),
             station_origin=(0.,0.),station_prefix_m=0.,station_limit_m=None,
             service_profiles={0:dict(blind_fraction=.5)},return_weight=1.,**options)


class RecourseTests(unittest.TestCase):
    def test_retry_can_be_inserted_with_zero_extra_travel(self):
        model=example_model()
        route=[0,1,2,3]
        # The source is exactly on the edge from station 0 to station 1.
        expected_service=17.+.35*(100.-19.8)
        self.assertAlmostEqual(model.retry_insertion_cost(route,0,100.),expected_service)
        self.assertLess(model.retry_insertion_cost(route,0,100.),model.distance[3][0])
        self.assertAlmostEqual(model.failure_return_charge(route),.5*expected_service)
        # Useful intervening information makes a later certified clear possible
        # in the ranking model; no actual polygon is modified by this forecast.
        model.effects={1:[(0,10.,True)]}
        self.assertAlmostEqual(model.retry_insertion_cost(route,0,100.),5.)

    def test_cap_limits_only_the_added_failure_penalty(self):
        model=example_model(cap_s=3.)
        route=[0,1,2,3]
        self.assertEqual(model.failure_return_charge(route),3.)
        self.assertGreater(model.cost(route),3.)
        model.risk_enabled=False
        self.assertEqual(model.failure_return_charge(route),0.)

    def test_local_candidates_preserve_stations_and_deferred_tasks(self):
        route=[0,1,2,3,4,5]
        stations={1,3,4};blocked={5}
        candidates=head_candidates(route,stations,blocked)
        self.assertIn(route,candidates)
        self.assertTrue(any(c[0]==1 for c in candidates))
        for c in candidates:
            self.assertEqual(sorted(c),route)
            self.assertEqual([k for k in c if k in stations],[1,3,4])
            self.assertGreater(c.index(5),c.index(4))

    def test_complete_missions_keep_geometry_and_coverage(self):
        for seed,mode,scale,variant in [(349100,'mixed',300.,'local_capped'),(349101,'edge_outward',900.,'insert')]:
            r=run_one(seed,mode,scale,variant)
            self.assertTrue(r['completed'])
            self.assertEqual(r['containment_violations'],0)
            self.assertEqual(r['nearby_bearings_le30m'],0)


if __name__=='__main__':unittest.main()
