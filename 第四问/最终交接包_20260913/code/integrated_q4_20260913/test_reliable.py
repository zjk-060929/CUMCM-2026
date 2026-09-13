import unittest
from reliable_action_strategy import ReliableActionPolicy
from predictive_strategy_q4 import PredictiveClient
from spatial_error_q4 import SpatialErrorSimulator
from strategy import Track
from geometry import enclosing_circle


class ReliableContracts(unittest.TestCase):
    def make_policy(self):
        p=ReliableActionPolicy(PredictiveClient(SpatialErrorSimulator([])))
        t=Track(polygon=[(100.,-10.),(500.,-10.),(500.,10.),(100.,10.)],observations=[((0.,0.),0.)])
        t.center,t.radius=enclosing_circle(t.polygon);p.tracks[1]=t;p.discovered.add(1)
        return p

    def test_three_low_gain_probes_suspend_optional_forecasts(self):
        p=self.make_policy();self.assertTrue(p.valid_probe(1,(0.,100.)))
        p.weak_probes[1]=3
        self.assertFalse(p.valid_probe(1,(0.,100.)))
        self.assertFalse(p.valid_probe(1,(0.,200.)))

    def test_optical_recovery_does_not_recommend_optional_probe(self):
        p=self.make_policy();p.covers[1]=[(250.,0.)]
        self.assertFalse(p.valid_probe(1,(0.,100.)))

    def test_empty_feasible_bank_is_not_information_gain(self):
        p=self.make_policy();p.forecaster(1).depleted=True
        self.assertFalse(p.valid_probe(1,(0.,100.)))


if __name__=='__main__':unittest.main()
