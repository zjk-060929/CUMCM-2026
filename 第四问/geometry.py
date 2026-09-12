"""Q4 conservative geometry. All distances are metres, angles are degrees."""
from __future__ import annotations
import math
from functools import lru_cache
from itertools import combinations

ALPHA_DEG = 1.005
ALPHA = math.radians(ALPHA_DEG)
EPS = 1e-7
DEFAULT_SIDE = 905.

def add(a, b): return (a[0] + b[0], a[1] + b[1])
def sub(a, b): return (a[0] - b[0], a[1] - b[1])
def mul(a, t): return (a[0] * t, a[1] * t)
def dot(a, b): return a[0] * b[0] + a[1] * b[1]
def cross(a, b): return a[0] * b[1] - a[1] * b[0]
def dist(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
def unit(deg): return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))

def clip(poly, normal, bound):
    """Sutherland-Hodgman clipping by normal·x <= bound; keeps degeneracies."""
    if not poly: return []
    out = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        da, db = dot(normal, a) - bound, dot(normal, b) - bound
        ia, ib = da <= EPS, db <= EPS
        if ia: out.append(a)
        if ia != ib:
            den = da - db
            if abs(den) > 1e-15:
                t = max(0.0, min(1.0, da / den))
                out.append(add(a, mul(sub(b, a), t)))
    clean = []
    for p in out:
        if not clean or dist(p, clean[-1]) > 1e-8: clean.append(p)
    if len(clean) > 1 and dist(clean[0], clean[-1]) < 1e-8: clean.pop()
    return clean

@lru_cache(None)
def domain_polygon():
    # Circumscribed (never inscribed) polygon, so no allowed source is excluded.
    p = [(-1800., -1800.), (1800., -1800.), (1800., 1800.), (-1800., 1800.)]
    for k in range(64): p = clip(p, unit(k * 360 / 64), 1800.)
    return tuple(p)

def incorporate(poly, position, bearing):
    lo, hi = unit(bearing - ALPHA_DEG), unit(bearing + ALPHA_DEG)
    for n in ((lo[1], -lo[0]), (-hi[1], hi[0])):
        poly = clip(poly, n, dot(n, position))
    # Positive reception proves distance <= 1500, regardless of emitter type.
    for k in range(24):
        n = unit(k * 15.)
        poly = clip(poly, n, dot(n, position) + 1500.)
    if not poly: raise ArithmeticError('Empty feasible set: inspect observations/rounding/model')
    return poly

def circumcircle(a, b, c):
    # Translate before computing to avoid cancellation at large coordinates.
    u, v = sub(b, a), sub(c, a)
    den = 2 * cross(u, v)
    if abs(den) < 1e-10: return None
    uu, vv = dot(u, u), dot(v, v)
    offset = ((uu * v[1] - vv * u[1]) / den, (u[0] * vv - v[0] * uu) / den)
    center = add(a, offset)
    return center, dist(center, a)

def enclosing_circle(poly):
    """Exact candidate enumeration; optimal circle supported by <=3 vertices."""
    if not poly: raise ArithmeticError('Empty polygon')
    if len(poly) == 1: return poly[0], 0.
    candidates = [(mul(add(a, b), .5), dist(a, b) / 2) for a, b in combinations(poly, 2)]
    candidates += [c for abc in combinations(poly, 3) if (c := circumcircle(*abc)) is not None]
    for center, radius in sorted(candidates, key=lambda c: c[1]):
        if all(dist(center, p) <= radius + EPS for p in poly):
            return center, max(dist(center, p) for p in poly) + EPS
    raise ArithmeticError('Failed enclosing-circle calculation')

def segment_distance(p, a, b):
    d = sub(b, a)
    t = max(0., min(1., dot(sub(p, a), d) / dot(d, d))) if dot(d, d) else 0.
    return dist(p, add(a, mul(d, t)))

def polygon_distance(p, poly):
    if len(poly) == 1: return dist(p, poly[0])
    edges = list(zip(poly, poly[1:] + poly[:1]))
    if len(poly) >= 3 and all(cross(sub(b, a), sub(p, a)) >= -EPS for a, b in edges): return 0.
    return min(segment_distance(p, a, b) for a, b in edges)

@lru_cache(None)
def triangular_mesh(side=DEFAULT_SIDE):
    """Retain every lattice triangle intersecting the closed source disk."""
    def xy(i, j): return (side * (i + j / 2), side * math.sqrt(3) * j / 2)
    extent = math.ceil(3600 / side) + 3
    triangles, indices = [], set()
    for i in range(-extent, extent):
        for j in range(-extent, extent):
            for ids in (((i,j),(i+1,j),(i,j+1)), ((i+1,j),(i+1,j+1),(i,j+1))):
                tri = [xy(*k) for k in ids]
                if polygon_distance((0., 0.), tri) <= 1800. + EPS:
                    triangles.append(tuple(tri))
                    indices.update(ids)
    points = tuple(xy(*k) for k in sorted(indices))
    return points, tuple(triangles)

def coverage_stations(kind='triangle', side=DEFAULT_SIDE):
    if kind == 'square': return [(float(x),float(y)) for x in range(-2000,2001,500) for y in range(-2000,2001,500)]
    if kind == 'triangle': return list(triangular_mesh(side)[0])
    raise ValueError(kind)

def open_route(points, start=(0.,0.)):
    """Nearest neighbour + deterministic open-path 2-opt, fixed start, free end."""
    remaining, route, p = list(points), [], start
    while remaining:
        k = min(range(len(remaining)), key=lambda i: (dist(p, remaining[i]), remaining[i]))
        p = remaining.pop(k); route.append(p)
    for _ in range(15):
        changed = False
        for i in range(len(route)):
            a = start if i == 0 else route[i-1]
            for j in range(i+1,len(route)):
                before, after = dist(a,route[i]), dist(a,route[j])
                if j+1 < len(route):
                    before += dist(route[j],route[j+1]); after += dist(route[i],route[j+1])
                if after < before - EPS:
                    route[i:j+1] = reversed(route[i:j+1]); changed = True
        if not changed: break
    return route

def optical_cover(position, bearing):
    """152 disks cover the full first-bearing uncertainty rectangle."""
    u, v = unit(bearing), unit(bearing+90)
    half_width = 1500 * math.sin(ALPHA) / 2
    return [add(position, add(mul(u,20*k), mul(v,sign*half_width)))
            for k in range(76) for sign in (-1,1)]

def clipped_optical_cover(position, bearing, poly):
    # Dropping only disks disjoint from the conservative feasible polygon is safe.
    return [p for p in optical_cover(position,bearing) if polygon_distance(p,poly) <= 20. + 1e-5]
