"""Whole-route regressions and end-to-end invariant audits for iteration 6."""
import unittest

from benchmark_mission_cost_q4 import run_one
from mission_cost_strategy import MissionCostModel


class MissionCostTests(unittest.TestCase):
    def test_longer_station_skeleton_can_have_shorter_actual_route(self):
        # Services have moved the receiver to x=99. The old skeleton anchor is 0.
        model = MissionCostModel([('station',1),('station',2),('task',3)],
                    [(10.,0.),(20.,0.),(19.,0.)],(99.,0.),{2:1.},{},set(),
                    station_origin=(0.,0.),station_prefix_m=80.,station_limit_m=None,
                    travel_discount=0.)
        old = [0,2,1]
        new = [1,2,0]
        self.assertGreater(model.station_path_s(new),model.station_path_s(old))
        self.assertLess(model.path_seconds(new),model.path_seconds(old))
        self.assertLess(model.cost(new),model.cost(old))
        self.assertAlmostEqual(model.path_seconds(new)*5,89.)

    def test_optional_future_station_travel_is_fully_charged(self):
        args = ([('station',0),('station',1)],[(0.,0.),(100.,0.)],(0.,0.),{}, {},set())
        kwargs = dict(station_origin=(0.,0.),station_prefix_m=0.,station_limit_m=None,
                      unknown_channels=5,discovery=({0:1,1:2},3))
        optimistic = MissionCostModel(*args,travel_discount=1.,**kwargs)
        full = MissionCostModel(*args,travel_discount=0.,**kwargs)
        self.assertAlmostEqual(full.cost([0,1])-optimistic.cost([0,1]),5.)

    def test_full_route_detour_charge_is_soft(self):
        args = ([('station',0),('task',1)],[(10.,0.),(100.,0.)],(0.,0.),{1:1.},{},set())
        kwargs = dict(station_origin=(0.,0.),station_prefix_m=0.,station_limit_m=None)
        plain = MissionCostModel(*args,**kwargs)
        soft = MissionCostModel(*args,detour_weight=.5,geometric_reference_s=10.,
                                 detour_allowance_m=0.,**kwargs)
        self.assertAlmostEqual(soft.cost([0,1])-plain.cost([0,1]),5.)
        self.assertLess(soft.cost([1,0]),1e6)

    def test_complete_missions_preserve_localization_and_discovery(self):
        for seed,mode,scale,variant in [(309100,'mixed',300.,'soft'),(309101,'edge_outward',900.,'fulltravel')]:
            row = run_one(seed,mode,scale,variant)
            self.assertTrue(row['completed'])
            self.assertEqual(row['containment_violations'],0)
            self.assertEqual(row['nearby_bearings_le30m'],0)
            self.assertIsNone(row['station_limit_m'])


if __name__=='__main__':unittest.main()
