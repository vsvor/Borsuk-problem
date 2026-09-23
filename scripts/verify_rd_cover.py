#!/usr/bin/env python3
"""Verify the four convex hulls in rd2_0957.txt against the intended polyhedron.

Usage:
    python verify_rd_cover.py rd2_0957.txt --json rd2_cover_report.json
Dependencies: numpy, scipy, shapely.

All measurements use the input coordinates unchanged. A separately projected
copy is used ONLY to construct a covering triangulation of the target boundary
for the global distance bound. It is not substituted for the input hulls.

Numerical method:
* Volumes: halfspace intersections and inclusion-exclusion (all 15 subsets).
* Outward error: maximum distance of any input vertex from the target.
* Inward error: common-center reduction to the target boundary, followed by
  triangle subdivision. Convexity of distance to each convex hull supplies
  upper bounds, while evaluated points supply lower bounds.

This is double/extended-precision numerical verification, not an
interval-arithmetic or exact-arithmetic certificate.
"""
from __future__ import annotations
import argparse
from decimal import Decimal, localcontext
import ast
import heapq
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Polygon
from shapely.ops import unary_union


def load_input(path: Path):
    lines = path.read_text().splitlines()
    if len(lines) != 5:
        raise ValueError('Expected five lines: k, stored value, n, index lists, coordinates.')
    k, stored, n = int(lines[0]), float(lines[1]), int(lines[2])
    groups = ast.literal_eval(lines[3])
    points = np.asarray(ast.literal_eval(lines[4]), dtype=float)
    if k != 4 or points.shape != (n, 3) or len(groups) != 4:
        raise ValueError('This verifier expects four groups and three-dimensional points.')
    if n != 53 or not np.all(np.isfinite(points)):
        raise ValueError('Expected 53 finite points.')
    for g in groups:
        if len(g) != len(set(g)) or any(not isinstance(j, int) or j < 0 or j >= n for j in g):
            raise ValueError('Invalid or repeated vertex index.')
    return points, groups, stored


def target_normals():
    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            row = np.zeros(3)
            row[axis] = sign
            rows.append(row)
    for i, j in itertools.combinations(range(3), 2):
        for si, sj in itertools.product((-1, 1), repeat=2):
            row = np.zeros(3)
            row[i], row[j] = si / np.sqrt(2), sj / np.sqrt(2)
            rows.append(row)
    return np.asarray(rows)


def unique_planes(equations, tol=1e-12):
    out = []
    for row in equations:
        if not out or np.min(np.linalg.norm(np.asarray(out) - row, axis=1)) > tol:
            out.append(row)
    return np.asarray(out)


def intersect_halfspaces(eq):
    """Return volume, vertices, and a Chebyshev-center diagnostic."""
    A, b = eq[:, :3], -eq[:, 3]
    result = linprog(
        [0, 0, 0, -1],
        A_ub=np.column_stack((A, np.linalg.norm(A, axis=1))), b_ub=b,
        bounds=[(None, None)] * 3 + [(0, None)], method='highs',
        options={'primal_feasibility_tolerance': 1e-10,
                 'dual_feasibility_tolerance': 1e-10},
    )
    if not result.success:
        raise RuntimeError('Halfspace intersection LP failed: ' + result.message)
    radius = float(result.x[3])
    # In this dataset all triple and quadruple intersections are numerically
    # lower-dimensional; retain their radii in the report for inspection.
    if radius < 1e-12:
        return 0.0, np.empty((0, 3)), {'inscribed_radius': radius, 'lower_dimensional': True}
    vertices = HalfspaceIntersection(eq, result.x[:3]).intersections
    hull = ConvexHull(vertices)
    vertices = vertices[hull.vertices]
    return float(hull.volume), vertices, {
        'inscribed_radius': radius, 'vertices': len(vertices), 'lower_dimensional': False,
    }


class HullDistance:
    """Point-to-solid distance via orthogonal triangle and segment projections."""
    def __init__(self, points, hull):
        self.eq = hull.equations
        self.tri = points[hull.simplices]
        self.a, self.b, self.c = self.tri[:, 0], self.tri[:, 1], self.tri[:, 2]
        self.e, self.f = self.b - self.a, self.c - self.a
        normals = np.cross(self.e, self.f)
        self.normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        self.ee = np.einsum('ij,ij->i', self.e, self.e)
        self.ff = np.einsum('ij,ij->i', self.f, self.f)
        self.ef = np.einsum('ij,ij->i', self.e, self.f)
        self.den = self.ee * self.ff - self.ef ** 2

    def __call__(self, x, with_point=False):
        if np.max(self.eq[:, :3] @ x + self.eq[:, 3]) < -2e-16:
            return (0.0, x.copy()) if with_point else 0.0
        residual = np.einsum('ij,ij->i', x - self.a, self.normals)
        pp = x - residual[:, None] * self.normals
        ev = np.einsum('ij,ij->i', pp - self.a, self.e)
        fv = np.einsum('ij,ij->i', pp - self.a, self.f)
        ub = (self.ff * ev - self.ef * fv) / self.den
        uc = (self.ee * fv - self.ef * ev) / self.den
        inside = (ub >= -1e-14) & (uc >= -1e-14) & (ub + uc <= 1 + 1e-14)
        candidates = [pp[inside]]
        for a, b in ((self.a, self.b), (self.b, self.c), (self.c, self.a)):
            edge = b - a
            t = np.einsum('ij,ij->i', x - a, edge) / np.einsum('ij,ij->i', edge, edge)
            candidates.append(a + np.clip(t, 0, 1)[:, None] * edge)
        candidates = np.vstack(candidates)
        distances = np.linalg.norm(candidates - x, axis=1)
        j = int(np.argmin(distances))
        return (float(distances[j]), candidates[j]) if with_point else float(distances[j])


def project_nearby_planes(X, N):
    Y = X.copy()
    for j, x in enumerate(X):
        active = np.flatnonzero(np.abs(N @ x - .5) < 1e-6)
        if len(active):
            Y[j] += np.linalg.lstsq(N[active], .5 - N[active] @ x, rcond=None)[0]
    return Y


def boundary_cover(X, Y, groups, N):
    """Cover each target face by convex polygons; triangulate those polygons.

    Every polygon uses coplanar vertices of ONE projected hull and is clipped
    to the target face. Polygon union area is checked independently per face.
    """
    triangles, diagnostics = [], []
    for face_id, normal in enumerate(N):
        u = np.eye(3)[np.argmin(np.abs(normal))]
        u -= u.dot(normal) * normal
        u /= np.linalg.norm(u)
        basis = np.column_stack((u, np.cross(normal, u)))
        ids = np.flatnonzero(np.abs(X[:32] @ normal - .5) < 1e-12)
        p = Y[ids] @ basis
        face = Polygon(p[ConvexHull(p).vertices])
        patches = []
        for group in groups:
            ids = [j for j in group if abs(Y[j] @ normal - .5) < 1e-12]
            if len(ids) < 3:
                continue
            p = Y[ids] @ basis
            try:
                poly = Polygon(p[ConvexHull(p).vertices]).intersection(face)
            except QhullError:
                continue  # A collinear set contributes no area.
            if poly.geom_type != 'Polygon' or poly.area < 1e-20:
                continue
            patches.append(poly)
            coords = np.asarray(poly.exterior.coords)[:-1] @ basis.T + .5 * normal
            for j in range(1, len(coords) - 1):
                tri = coords[[0, j, j + 1]]
                if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-25:
                    triangles.append((face_id, tri))
        missing = float(face.difference(unary_union(patches)).area)
        diagnostics.append({'face_id': face_id, 'normal': normal.tolist(),
                            'area': float(face.area), 'uncovered_area_projected': missing})
        if missing > 1e-12:
            raise RuntimeError(f'Projected boundary does not cover target face {face_id}: {missing}.')
    return triangles, diagnostics


def chord_bound(D):
    """Max over a triangle of min_i sum_j lambda_j d(vertex_j, hull_i).

    Enumerate all vertices of this concave, piecewise-linear upper bound:
    triangle vertices; pairwise equalities on edges; triple equalities inside.
    """
    D = np.asarray(D, dtype=np.longdouble)
    candidates = [np.eye(3, dtype=np.longdouble)[i] for i in range(3)]
    for a, b in itertools.combinations(range(3), 2):
        for i, j in itertools.combinations(range(4), 2):
            u, v = D[a, i] - D[a, j], D[b, i] - D[b, j]
            if u == v:
                continue
            t = -v / (u - v)
            if 0 <= t <= 1:
                lam = np.zeros(3, dtype=np.longdouble)
                lam[a], lam[b] = t, 1 - t
                candidates.append(lam)
    for i, j, k in itertools.combinations(range(4), 3):
        lam = np.cross(D[:, i] - D[:, j], D[:, i] - D[:, k])
        total = lam.sum()
        if total == 0:
            continue
        lam /= total
        if np.all(lam >= 0):
            candidates.append(lam)
    candidates = np.asarray(candidates)
    values = (candidates @ D).min(axis=1)
    j = int(np.argmax(values))
    return float(values[j]), np.asarray(candidates[j], dtype=float)


def global_cover_distance(triangles, distances, tolerance, max_iterations=20000):
    cache, queue = {}, []
    serial, lower, witness, witness_face = 0, -1.0, None, None

    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            cache[key] = np.asarray([d(x) for d in distances])
        return cache[key]

    def insert(face_id, tri, D=None):
        nonlocal serial, lower, witness, witness_face
        if D is None:
            D = np.asarray([evaluate(x) for x in tri])
        d = D.min(axis=1)
        j = int(np.argmax(d))
        if d[j] > lower:
            lower, witness, witness_face = float(d[j]), tri[j].copy(), face_id
        upper, barycentric = chord_bound(D)
        heapq.heappush(queue, (-upper, serial, face_id, tri, D, barycentric))
        serial += 1

    for face_id, tri in triangles:
        insert(face_id, tri)
    iterations = 0
    while queue and -queue[0][0] > lower + tolerance:
        if iterations >= max_iterations:
            raise RuntimeError('Distance subdivision did not meet the requested tolerance.')
        _, _, face_id, tri, D, barycentric = heapq.heappop(queue)
        p = barycentric @ tri
        dp = evaluate(p)
        if dp.min() > lower:
            lower, witness, witness_face = float(dp.min()), p.copy(), face_id
        for j in range(3):
            if barycentric[j] < 1e-15:
                continue
            child, child_D = tri.copy(), D.copy()
            child[j], child_D[j] = p, dp
            insert(face_id, child, child_D)
        iterations += 1
    upper = max(lower, -queue[0][0]) if queue else lower
    return {'lower_numerical': lower, 'upper_numerical': upper,
            'witness': witness.tolist(), 'witness_face': witness_face,
            'distances_to_four_hulls_at_witness': evaluate(witness).tolist(),
            'initial_triangles': len(triangles), 'subdivisions': iterations,
            'requested_subdivision_tolerance': tolerance,
            'arithmetic_note': 'Floating-point bounds; no outward-rounded interval arithmetic.'}


def verify(path: Path, tolerance: float):
    X, groups, stored = load_input(path)
    N = target_normals()
    target_eq = np.column_stack((N, np.full(len(N), -.5)))
    target_hull = ConvexHull(X[:32])
    with localcontext() as ctx:
        ctx.prec = 60
        target_volume_decimal = Decimal(7) - Decimal(9) / 2 * Decimal(2).sqrt()
    target_volume = float(target_volume_decimal)
    hulls = [ConvexHull(X[g]) for g in groups]
    equations = [unique_planes(h.equations) for h in hulls]
    distances = [HullDistance(X[g], h) for g, h in zip(groups, hulls)]
    common = sorted(set.intersection(*(set(g) for g in groups)))
    if not common or np.max(N @ X[common[0]] - .5) > 1e-12:
        raise RuntimeError('No common input vertex inside the target; boundary reduction is invalid.')
    result: dict[str, Any] = {
        'input': path.name, 'indexing': 'zero-based', 'stored_value': stored,
        'target': {'description': '|x_i| <= 1/2; |x_i|+|x_j| <= 1/sqrt(2)',
                   'face_count': 18, 'vertex_count': 32,
                   'volume': target_volume, 'volume_formula': '7 - (9/2)*sqrt(2)',
                   'volume_decimal': str(target_volume_decimal),
                   'volume_from_first_32_points': float(target_hull.volume)},
        'common_vertex_indices': common,
        'hulls': [], 'intersections': [],
    }
    for k, (g, hull) in enumerate(zip(groups, hulls)):
        D = squareform(pdist(X[g]))
        a, b = np.unravel_index(np.argmax(D), D.shape)
        result['hulls'].append({'hull': k + 1, 'vertex_count': len(hull.vertices),
                                'volume': float(hull.volume), 'diameter': float(D[a, b]),
                                'diameter_pair': [g[a], g[b]]})
    raw_union, inside_union = 0.0, 0.0
    for count in range(1, 5):
        for ids in itertools.combinations(range(4), count):
            eq = np.vstack([equations[i] for i in ids])
            rv, _, rd = intersect_halfspaces(eq)
            iv, vertices, info = intersect_halfspaces(np.vstack((eq, target_eq)))
            raw_union += (-1) ** (count + 1) * rv
            inside_union += (-1) ** (count + 1) * iv
            result['intersections'].append({'hulls': [i + 1 for i in ids],
                                            'volume': rv, 'inside_target_volume': iv,
                                            'diagnostics': rd, 'inside_diagnostics': info})
            if count == 1:
                result['hulls'][ids[0]]['clipped_volume'] = iv
                if len(vertices) >= 2:
                    D = squareform(pdist(vertices))
                    a, b = np.unravel_index(np.argmax(D), D.shape)
                    result['hulls'][ids[0]]['clipped_diameter'] = float(D[a, b])
                    result['hulls'][ids[0]]['clipped_diameter_points'] = vertices[[a, b]].tolist()
    result['volumes'] = {
        'union': raw_union, 'union_inside_target': inside_union,
        'uncovered': target_volume - inside_union,
        'uncovered_fraction': (target_volume - inside_union) / target_volume,
        'outside_target': raw_union - inside_union,
        'outside_target_fraction': (raw_union - inside_union) / target_volume,
        'note': 'Inclusion-exclusion, including all six pairwise overlaps; not a sum-of-volumes test.',
    }
    # Project each input point onto the target by enumerating active sets of
    # one, two or three face planes. This avoids an optimization tolerance
    # comparable with the ~1e-8 outward errors.
    outside_records = []
    residuals = X @ N.T - .5
    for j, x in enumerate(X):
        if residuals[j].max() <= 1e-15:
            d, q = 0.0, x.copy()
        else:
            d, q = np.inf, None
            for count in (1, 2, 3):
                for ids in itertools.combinations(range(len(N)), count):
                    A = N[list(ids)]
                    if np.linalg.matrix_rank(A) < count:
                        continue
                    multipliers = np.linalg.solve(A @ A.T, A @ x - .5)
                    candidate = x - A.T @ multipliers
                    if multipliers.min() < -1e-13 or (N @ candidate - .5).max() > 1e-13:
                        continue
                    dd = np.linalg.norm(candidate - x)
                    if dd < d:
                        d, q = float(dd), candidate
            if q is None:
                raise RuntimeError(f'Projection failed at vertex {j}.')
        outside_records.append({'index': j, 'distance_to_target': d,
                                'closest_target_point': q.tolist(),
                                'max_normalized_plane_violation': float(residuals[j].max())})
    result['outward_error'] = max(outside_records, key=lambda row: row['distance_to_target'])
    result['vertex_target_distances'] = outside_records
    Y = project_nearby_planes(X, N)
    triangles, face_diagnostics = boundary_cover(X, Y, groups, N)
    corrections = np.linalg.norm(Y - X, axis=1)
    result['auxiliary_boundary_construction'] = {
        'largest_vertex_projection': float(corrections.max()),
        'largest_projection_index': int(corrections.argmax()),
        'face_checks': face_diagnostics,
        'note': 'Only a device to cover the target boundary; all distances above and below use original input hulls.',
    }
    result['cover_distance'] = global_cover_distance(triangles, distances, tolerance)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path)
    parser.add_argument('--json', type=Path, default=Path('rd2_cover_report.json'))
    parser.add_argument('--distance-tolerance', type=float, default=1e-13)
    args = parser.parse_args()
    if args.distance_tolerance <= 0:
        parser.error('--distance-tolerance must be positive')
    try:
        result = verify(args.input, args.distance_tolerance)
    except (ValueError, RuntimeError, OSError, QhullError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    args.json.write_text(json.dumps(result, indent=2) + '\n')
    print('Target volume:', result['target']['volume'])
    print('Uncovered volume:', result['volumes']['uncovered'])
    print('Uncovered fraction:', result['volumes']['uncovered_fraction'])
    print('Cover-distance numerical bracket:', result['cover_distance']['lower_numerical'],
          result['cover_distance']['upper_numerical'])
    print('Witness:', result['cover_distance']['witness'])
    print('Outward error:', result['outward_error'])
    print('Stored value:', result['stored_value'])
    for hull in result['hulls']:
        print('Hull', hull['hull'], 'diameter', hull['diameter'], 'pair', hull['diameter_pair'],
              'clipped diameter', hull.get('clipped_diameter'))
    print('Report:', args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
