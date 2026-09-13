"""Certified geometry for the independent Q4 optimization entry point.

The 25-station construction is a finite-domain triangulation, not a claim of
globally optimal coverage. See 优化说明与实验结果.md for the proof.
"""
from __future__ import annotations
from array import array
from functools import lru_cache
import math

from geometry import add, sub, mul, dot, cross, dist, unit, polygon_distance, EPS

SAFE_RADIUS = 19.8


@lru_cache(None)
def ring_mesh(inner_radius=970., outer_radius=1870.):
    inner = tuple(mul(unit(15 + 30*k), inner_radius) for k in range(12))
    outer = tuple(mul(unit(30*k), outer_radius) for k in range(12))
    origin = (0., 0.)
    triangles = []
    for k in range(12):
        j = (k+1) % 12
        triangles.extend(((origin, inner[k], inner[j]),
                          (outer[k], outer[j], inner[k]),
                          (inner[k], outer[j], inner[j])))
    if outer_radius * math.cos(math.pi/12) < 1800.:
        raise ValueError('Outer polygon must contain the closed source disk')
    if not 0 < inner_radius < outer_radius * math.cos(math.pi/12):
        raise ValueError('Inner vertices must be strictly inside outer edges')
    if max(dist(a, b) for tri in triangles for a in tri for b in tri) >= 1000.:
        raise ValueError('Every triangle diameter must be strictly below 1000 m')
    if any(cross(sub(t[1], t[0]), sub(t[2], t[0])) <= 0 for t in triangles):
        raise ValueError('Invalid triangle orientation')
    return (origin,) + inner + outer, tuple(triangles)


def route_length(route, start=(0., 0.)):
    return sum(dist(a, b) for a, b in zip([start] + list(route), route))


def improve_route(route, start=(0., 0.)):
    """Open-path 2-opt with fixed start and free end; only improving exchanges."""
    route = list(route)
    for _ in range(30):
        changed = False
        for i in range(len(route)):
            a = start if i == 0 else route[i-1]
            for j in range(i+1, len(route)):
                before, after = dist(a, route[i]), dist(a, route[j])
                if j+1 < len(route):
                    before += dist(route[j], route[j+1])
                    after += dist(route[i], route[j+1])
                if after < before - 1e-6:
                    route[i:j+1] = reversed(route[i:j+1])
                    changed = True
        if not changed:
            break
    return route


@lru_cache(None)
def ring_route(inner_radius=970., outer_radius=1870., outer_first=False):
    points, _ = ring_mesh(inner_radius, outer_radius)
    inner, outer = list(points[1:13]), list(points[13:])
    initial = [points[0]] + (outer + inner if outer_first else inner + outer)
    return tuple(improve_route(initial))


def safe_clear_point(center, radius, start):
    """Nearest point in a sufficient safe-clear disk B(center, 19.8-radius)."""
    if radius > SAFE_RADIUS:
        raise ValueError('No certified clear disk')
    delta = sub(start, center)
    length = dist(start, center)
    slack = max(0., SAFE_RADIUS-radius)
    return start if length <= slack else add(center, mul(delta, slack/length))


def safe_detour_point(center, radius, start, end):
    """Pick a safe clear point toward the nearest point of the planned segment."""
    edge = sub(end, start)
    length2 = dot(edge, edge)
    t = max(0., min(1., dot(sub(center, start), edge)/length2)) if length2 else 0.
    return safe_clear_point(center, radius, add(start, mul(edge, t)))


def polygon_optical_cover(poly, cover_radius=SAFE_RADIUS):
    """Cover an oriented bounding rectangle of the entire feasible polygon.

    Each rectangular cell has half-diagonal <= cover_radius. Its centre's
    clear disk covers the entire cell. We only discard disks disjoint from P.
    No probability model or source estimate enters the coverage certificate.
    """
    if not poly:
        raise ArithmeticError('Cannot cover an empty feasible polygon')
    if len(poly) == 1:
        return list(poly)
    candidates = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        length = dist(a, b)
        if length <= 1e-8:
            continue
        u = mul(sub(b, a), 1/length)
        v = (-u[1], u[0])
        xx, yy = [dot(p, u) for p in poly], [dot(p, v) for p in poly]
        xmin, xmax, ymin, ymax = min(xx), max(xx), min(yy), max(yy)
        width, height = xmax-xmin, ymax-ymin
        # All row counts up to square-sized cells, plus a little slack.
        max_rows = max(1, math.ceil(height/(math.sqrt(2)*cover_radius)) + 1)
        for ny in range(1, max_rows+1):
            half_y = height/(2*ny)
            if half_y >= cover_radius:
                continue
            max_dx = 2*math.sqrt(cover_radius**2-half_y**2)
            nx = max(1, math.ceil(width/max_dx))
            candidates.append((nx*ny, width+height, nx, ny, u, v,
                               xmin, xmax, ymin, ymax))
    if not candidates:
        raise ArithmeticError('Could not construct optical cover')
    _, _, nx, ny, u, v, xmin, xmax, ymin, ymax = min(candidates)
    points = [add(mul(u, xmin+(i+.5)*(xmax-xmin)/nx),
                  mul(v, ymin+(j+.5)*(ymax-ymin)/ny))
              for i in range(nx) for j in range(ny)]
    return [p for p in points if polygon_distance(p, poly) <= cover_radius + 1e-6]


def held_karp_open(points, start):
    """Exact shortest open path for <=16 *fixed* representative points.

    dp[S,j] is minimum start-to-j length visiting exactly S. This does not
    solve joint sensing/clear-region optimization or predict future bearings.
    """
    n = len(points)
    if n > 16:
        raise ValueError('Exact service routing is limited to 16 sources')
    if not n:
        return []
    distance = [[dist(a, b) for b in points] for a in points]
    dp = array('d', [math.inf]) * ((1 << n)*n)
    parent = array('b', [-1]) * len(dp)
    for j in range(n):
        dp[(1 << j)*n+j] = dist(start, points[j])
    for mask in range(1, 1 << n):
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length()-1
            previous = mask ^ bit
            remaining ^= bit
            if not previous:
                continue
            kbits = previous
            best, pred = math.inf, -1
            base = previous*n
            while kbits:
                kbit = kbits & -kbits
                k = kbit.bit_length()-1
                cost = dp[base+k]+distance[k][j]
                if cost < best:
                    best, pred = cost, k
                kbits ^= kbit
            dp[mask*n+j], parent[mask*n+j] = best, pred
    mask = (1 << n)-1
    j = min(range(n), key=lambda k: dp[mask*n+k])
    route = []
    while j != -1:
        route.append(j)
        pred = parent[mask*n+j]
        mask ^= 1 << j
        j = pred
    return list(reversed(route))


def sample_polygon(poly, step=15.):
    """Interior cross-section samples for heuristic planning, never a certificate."""
    if len(poly) == 1:
        return list(poly)
    a, b = max(((a, b) for a in poly for b in poly), key=lambda pair: dist(*pair))
    length = dist(a, b)
    if length < 1e-8:
        return [a]
    u = mul(sub(b, a), 1/length)
    v = (-u[1], u[0])
    local = [(dot(p, u), dot(p, v)) for p in poly]
    xmin, xmax = min(p[0] for p in local), max(p[0] for p in local)
    nx = max(1, math.ceil((xmax-xmin)/step))
    result = []
    for i in range(nx):
        x = xmin+(i+.5)*(xmax-xmin)/nx
        ys = []
        for (ax, ay), (bx, by) in zip(local, local[1:]+local[:1]):
            if min(ax, bx)-1e-8 <= x <= max(ax, bx)+1e-8 and abs(ax-bx) > 1e-10:
                ys.append(ay+(by-ay)*(x-ax)/(bx-ax))
        if not ys:
            continue
        lo, hi = min(ys), max(ys)
        ny = max(1, math.ceil((hi-lo)/10.))
        result.extend(add(mul(u, x), mul(v, lo+(j+.5)*(hi-lo)/ny)) for j in range(ny))
    return result or list(poly)


def possible_emitter_at(g, positive_positions, negative_positions):
    """Conservative feasibility of an unknown radius and orientation at sample g.

    A positive reception forces R >= max(1000, distance to every positive).
    Every negative position within that radius must be behind a directional
    transmitter. A common half-plane normal exists iff the corresponding
    signed vectors fit in a closed semicircle. Relaxing strict negativity to
    closed halfplanes may retain extra hypotheses; it cannot justify removing
    a true position. The caller uses this only to score discrete hypotheses.
    """
    required_radius = max([1000.] + [dist(g, p) for p in positive_positions])
    forced_back = [q for q in negative_positions if dist(g, q) <= required_radius-1e-6]
    if not forced_back:
        return True  # An omnidirectional emitter is still possible.
    vectors = [sub(p, g) for p in positive_positions] + [sub(g, q) for q in forced_back]
    angles = sorted(math.atan2(v[1], v[0]) for v in vectors if dot(v, v) > 1e-12)
    if len(angles) <= 1:
        return True
    gaps = [b-a for a, b in zip(angles, angles[1:]+[angles[0]+2*math.pi])]
    return max(gaps) >= math.pi-1e-9
