"""Smooth, bounded spatial bias for sensitivity tests, not an official model."""
import math
import random

from simulator import LocalSimulator


class CorrelatedSimulator(LocalSimulator):
    def __init__(self, sources, seed=0, length_scale=100., robot_id='SELF-Q4', capture=False):
        if not math.isfinite(length_scale) or length_scale <= 0:
            raise ValueError('length_scale must be finite and positive')
        super().__init__(sources, seed, 'smooth_spatial', robot_id, capture)
        self.length_scale = length_scale
        rng = random.Random(seed+970133)
        self.waves = tuple((rng.uniform(0, 2*math.pi), rng.uniform(0, 2*math.pi))
                           for _ in range(4))

    def error(self, position):
        # Same field at the same position for every channel and every reading.
        # |e| <= 1 degree and |e(p)-e(q)| <= ||p-q|| / length_scale degrees.
        # The spatial scale is a sensitivity parameter, not a known real value.
        x, y = position
        return sum(math.sin((math.cos(a)*x+math.sin(a)*y)/self.length_scale+b)
                   for a, b in self.waves)/len(self.waves)
