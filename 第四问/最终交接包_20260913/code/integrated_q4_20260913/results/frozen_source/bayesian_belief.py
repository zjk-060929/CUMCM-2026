"""Finite directional/omnidirectional Bayesian discovery model, not a certificate."""
from functools import lru_cache
import math

POSITIONS = tuple((1800*math.sqrt((j+.5)/192)*math.cos(j*math.pi*(3-math.sqrt(5))),
                   1800*math.sqrt((j+.5)/192)*math.sin(j*math.pi*(3-math.sqrt(5)))) for j in range(192))
NORMALS = tuple((math.cos(k*math.pi/4), math.sin(k*math.pi/4)) for k in range(8))
PARTICLE_COUNT = 192*3*16
FULL_MASK = (1 << PARTICLE_COUNT)-1


def detects(g, radius, normal, q):
    dx, dy = q[0]-g[0], q[1]-g[1]
    return dx*dx+dy*dy <= (radius+1e-9)**2 and (normal is None or normal[0]*dx+normal[1]*dy >= -1e-9)


@lru_cache(maxsize=2048)
def detection_mask(q):
    """192 area-uniform positions, 3 radii, 8 directions + equal omni weight."""
    mask = 0
    for j, g in enumerate(POSITIONS):
        dx, dy = q[0]-g[0], q[1]-g[1]
        d2 = dx*dx+dy*dy
        direction_bits = sum(1 << k for k, n in enumerate(NORMALS) if n[0]*dx+n[1]*dy >= -1e-9)
        for k, radius in enumerate((1000., 1250., 1500.)):
            if d2 <= (radius+1e-9)**2:
                mask |= (direction_bits | (255 << 8)) << ((j*3+k)*16)
    return mask


def elementary(values, max_degree=16):
    coefficients = [1.]+[0.]*max_degree
    for value in values:
        for k in range(max_degree, 0, -1):
            coefficients[k] += value*coefficients[k-1]
    return coefficients


class DiscoveryBelief:
    def __init__(self):
        self.masks = {c:FULL_MASK for c in range(1, 21)}
        self.negative_positions = {c:set() for c in range(1, 21)}
        self.version = 0
        self._cache = None
        self.depleted_channels = set()

    def negative(self, channel, q):
        q = tuple(q)
        if q not in self.negative_positions[channel]:
            self.negative_positions[channel].add(q)
            self.masks[channel] &= ~detection_mask(q)
            self.version += 1
            if not self.masks[channel]:
                self.depleted_channels.add(channel)

    def posterior(self, discovered):
        key = self.version, tuple(sorted(discovered))
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        unknown = [c for c in range(1, 21) if c not in discovered]
        d = len(discovered)
        # A finite bank can miss a real thin component. A small likelihood
        # floor avoids declaring impossibility from this sampling artifact.
        likelihood = {c:max(1/PARTICLE_COUNT, self.masks[c].bit_count()/PARTICLE_COUNT) for c in unknown}
        coefficients = elementary(list(likelihood.values()))
        raw = {n:coefficients[n-d]/math.comb(20, n) for n in range(max(10, d), 17)}
        total = sum(raw.values())
        probability_n = {n:w/total for n,w in raw.items()}
        existence = {}
        for c in unknown:
            without = elementary([likelihood[k] for k in unknown if k != c])
            existence[c] = sum(probability_n[n]*likelihood[c]*without[n-d-1]/coefficients[n-d]
                               for n in probability_n if n > d and coefficients[n-d] > 0)
        mean_n = sum(n*w for n,w in probability_n.items())
        result = dict(unknown=unknown, probability_n=probability_n, existence=existence,
                      expected_total=mean_n, expected_unknown=mean_n-d,
                      likelihood=likelihood, coefficients=coefficients,
                      masks={c:self.masks[c] or FULL_MASK for c in unknown})
        self._cache = key, result
        return result

    def gain(self, q, posterior, missed_at=None):
        visible = detection_mask(tuple(q))
        missed = ~detection_mask(tuple(missed_at)) if missed_at is not None else FULL_MASK
        return sum(posterior['existence'][c]*(mask & visible & missed).bit_count()/mask.bit_count()
                   for c,mask in posterior['masks'].items())

    def discovery_probability(self, channel, q, posterior):
        mask = posterior['masks'][channel]
        return posterior['existence'][channel]*(mask & detection_mask(tuple(q))).bit_count()/mask.bit_count()
