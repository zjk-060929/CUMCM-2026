"""2026-09-13: directional coverage certificates and finer conservative circles."""
from functools import lru_cache
import math
from geometry import clip, unit, mul, dist, sub, cross, dot, ALPHA_DEG, polygon_distance
from optimized_geometry import improve_route


def hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return points
    def chain(items):
        result = []
        for p in items:
            while len(result) > 1 and cross(sub(result[-1], result[-2]), sub(p, result[-1])) <= 0:
                result.pop()
            result.append(p)
        return result
    return chain(points)[:-1]+chain(reversed(points))[:-1]


def hull_margin(poly, points):
    if len(poly) < 3:
        return -math.inf
    return min(cross(sub(b, a), sub(p, a))/dist(a, b)
               for a, b in zip(poly, poly[1:]+poly[:1]) for p in points)


@lru_cache(None)
def source_domain(sides=1024):
    poly = [(-1800., -1800.), (1800., -1800.), (1800., 1800.), (-1800., 1800.)]
    for k in range(sides):
        poly = clip(poly, unit(k*360/sides), 1800.)
    return tuple(poly)


def incorporate_fine(poly, position, bearing, sides=1024):
    lo, hi = unit(bearing-ALPHA_DEG), unit(bearing+ALPHA_DEG)
    for normal in ((lo[1], -lo[0]), (-hi[1], hi[0])):
        poly = clip(poly, normal, dot(normal, position))
    for k in range(sides):
        normal = unit(360*k/sides)
        poly = clip(poly, normal, dot(normal, position)+1500.)
    if not poly:
        raise ArithmeticError('Empty finer conservative feasible region')
    return poly


@lru_cache(None)
def sparse_rings(inner_count=7, outer_count=12, inner_radius=999., outer_radius=1870., phase=0.):
    points = [(0., 0.)]
    points += [mul(unit(phase+k*360/inner_count), inner_radius) for k in range(inner_count)]
    points += [mul(unit(k*360/outer_count), outer_radius) for k in range(outer_count)]
    return tuple(points)


def certify_layout(points, max_depth=12, keep_cells=False):
    """Finite sufficient certificate, NOT sampling of emitter orientations.

    Partition a circumscribed source polygon into convex cells C. For each
    cell choose stations with max_{vertex v of C}|s-v| <= 999.9. Certify that
    every vertex of C lies inside their convex hull with >=1e-5 m margin.
    Convexity proves both conditions for every g in C; the weighted-normal
    identity then proves reception for every emitter orientation.
    """
    cells, visits, worst_distance, min_margin = [], 0, 0., math.inf
    stack = [(list(source_domain(128)), 0, ())]
    while stack:
        poly, depth, path = stack.pop()
        visits += 1
        selected = [i for i, s in enumerate(points) if max(dist(s, v) for v in poly) <= 999.9]
        convex = hull([points[i] for i in selected])
        margin = hull_margin(convex, poly)
        if margin >= 1e-5:
            bound = max(dist(points[i], v) for i in selected for v in poly)
            worst_distance = max(worst_distance, bound)
            min_margin = min(min_margin, margin)
            cells.append({'path': path, 'vertices': poly, 'stations': selected} if keep_cells else None)
            continue
        center = (sum(p[0] for p in poly)/len(poly), sum(p[1] for p in poly)/len(poly))
        actual_nearby = [s for s in points if dist(s, center) <= 1000.]
        nearby_hull = hull(actual_nearby)
        outside = not nearby_hull or polygon_distance(center, nearby_hull) > 1e-6
        if dist(center, (0., 0.)) <= 1800. and outside:
            return dict(certified=False, witness=center, visits=visits, reason='uncovered_direction_at_witness')
        if depth == max_depth:
            return dict(certified=False, witness=center, visits=visits, reason='subdivision_limit')
        xmin, xmax = min(p[0] for p in poly), max(p[0] for p in poly)
        ymin, ymax = min(p[1] for p in poly), max(p[1] for p in poly)
        axis = 0 if xmax-xmin >= ymax-ymin else 1
        midpoint = (xmin+xmax)/2 if axis == 0 else (ymin+ymax)/2
        normal = (1., 0.) if axis == 0 else (0., 1.)
        children = [clip(poly, normal, midpoint), clip(poly, mul(normal, -1), -midpoint)]
        for k, child in enumerate(children):
            if child:
                stack.append((child, depth+1, path+(k,)))
    result = dict(certified=True, stations=len(points), cells=len(cells), visits=visits,
                  distance_bound_m=worst_distance, hull_margin_m=min_margin,
                  max_depth=max_depth, source_polygon_sides=128)
    if keep_cells:
        result.update(points=points, leaves=cells)
    return result


@lru_cache(None)
def certified_sparse_route(inner_count=7, outer_count=12, inner_radius=999., outer_radius=1870., phase=0.):
    points = sparse_rings(inner_count, outer_count, inner_radius, outer_radius, phase)
    certificate = certify_layout(points, max_depth=30)
    if not certificate['certified']:
        raise ValueError(f'Uncertified directional coverage: {certificate}')
    return tuple(improve_route(points))
