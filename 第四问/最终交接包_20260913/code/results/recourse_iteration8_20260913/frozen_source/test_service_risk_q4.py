"""Dependence, return-cost and temporary-waiting regression checks."""
import unittest

from benchmark_service_risk_q4 import run_one
from service_risk_strategy import joint_blind_fraction, should_wait, ServiceRiskModel


class ServiceRiskTests(unittest.TestCase):
    def test_two_queries_share_the_same_transmitter_hypothesis(self):
        hypotheses=[((0.,0.),100.,(1.,0.),1.),((0.,0.),100.,(-1.,0.),1.)]
        self.assertEqual(joint_blind_fraction(hypotheses,[(10.,0.),(20.,0.)]),.5)
        self.assertEqual(joint_blind_fraction(hypotheses,[(10.,0.),(-20.,0.)]),0.)
        self.assertEqual(joint_blind_fraction(hypotheses,[(30.,0.)],clear_center=(1.,0.)),0.)
        self.assertIsNone(joint_blind_fraction([],[(1.,0.)]))

    def test_waiting_can_end_after_new_information(self):
        self.assertTrue(should_wait('weak',1,True,300.,.8))
        self.assertFalse(should_wait('weak',2,True,300.,.8))
        self.assertFalse(should_wait('weak',1,True,79.,.8))
        self.assertFalse(should_wait('weak',1,False,300.,.8))
        self.assertTrue(should_wait('risk',1,True,300.,.61))
        self.assertFalse(should_wait('risk',1,True,300.,.59))

    def test_failed_service_keeps_a_cost_to_return(self):
        model=ServiceRiskModel([('task',1),('station',0),('station',1)],
                 [(0.,0.),(100.,0.),(1000.,0.)],(0.,0.),{0:100.},{},set(),
                 station_origin=(0.,0.),station_prefix_m=0.,station_limit_m=None,
                 service_profiles={0:dict(blind_fraction=.5)},return_weight=1.)
        self.assertAlmostEqual(model.failure_return_charge([0,1,2]),.5*(200.+12.+.35*(100.-19.8)))
        self.assertEqual(model.failure_return_charge([1,2,0]),0.)
        # A preceding station forecast that certifies a compact region removes
        # this particular early-service failure surcharge, but not travel costs.
        model.effects={1:[(0,10.,True)]}
        self.assertEqual(model.failure_return_charge([1,0,2]),0.)

    def test_complete_missions_keep_truth_containment_and_coverage(self):
        for seed,mode,scale,variant in [(329100,'mixed',300.,'combined'),(329101,'edge_outward',900.,'riskgate')]:
            row=run_one(seed,mode,scale,variant)
            self.assertTrue(row['completed'])
            self.assertEqual(row['containment_violations'],0)
            self.assertEqual(row['nearby_bearings_le30m'],0)


if __name__=='__main__':unittest.main()
