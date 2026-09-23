#!/usr/bin/env python3
"""Independent exact 3-D convex-cover checker (Python >= 3.10, standard library).

Only the FINAL coordinates and membership lists are used. Decimal input is
interpreted exactly, not converted through binary floats. The fast certificate
checks every target face by exact polygon subtraction and verifies a point
common to all hulls. --full-volume instead uses exact 3-D inclusion-exclusion
and needs no common point. No initial mesh, optimizer, tolerance, or Qhull is used.

With --pad e, each exact hull inequality n.x <= h is REPLACED by
n.x <= h + e*||n||_1. These are different, enlarged hulls; their diameters are
recomputed exactly. --pad is NOT an arithmetic tolerance or a distance bound.

Analytic rhombic targets use a rational OUTER enclosure of 1/sqrt(2).
A pass proves coverage of the ideal target; a failure need not disprove it.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools as it
import json
import math
from fractions import Fraction as F
from pathlib import Path
import sys
import time

Point = tuple[int, int, int, int]   # (X,Y,Z,W), W>0
Plane = tuple[int, int, int, int]   # A*x+B*y+C*z <= D


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def primitive(v):
    g = math.gcd(*v)
    return tuple(x//g for x in v) if g else tuple(v)


def point(v) -> Point:
    if not v[3]:
        raise ValueError('Point at infinity.')
    return primitive(tuple(-x for x in v) if v[3] < 0 else tuple(v))


def from_xyz(xyz) -> Point:
    fs = [F(x) for x in xyz]
    if len(fs) != 3:
        raise ValueError('Every point must have three coordinates.')
    d = math.lcm(*(x.denominator for x in fs))
    return point(tuple(x.numerator*(d//x.denominator) for x in fs)+(d,))


def plane(coefficients) -> Plane:
    fs = list(map(F, coefficients))
    d = math.lcm(*(x.denominator for x in fs))
    h = primitive(tuple(x.numerator*(d//x.denominator) for x in fs))
    if not any(h[:3]):
        raise ValueError('Zero plane normal.')
    return h


def val(h: Plane, p: Point) -> int:
    return dot(h[:3], p[:3])-h[3]*p[3]


def mean_point(points) -> Point:
    return from_xyz([sum((F(p[j], p[3]) for p in points), F(0))/len(points) for j in range(3)])


def full_dimension(ps) -> bool:
    if len(ps) < 4:
        return False
    p = ps[0]
    ds = [tuple(q[j]*p[3]-p[j]*q[3] for j in range(3)) for q in ps[1:]]
    u = next((v for v in ds if any(v)), None)
    if u is None:
        return False
    n = next((cross(u, v) for v in ds if any(cross(u, v))), None)
    return n is not None and any(dot(n, v) for v in ds)


def hull_planes(ps) -> list[Plane]:
    """All supporting planes, by exhaustive exact enumeration of triples."""
    ps = list(dict.fromkeys(ps))
    if not full_dimension(ps):
        raise ValueError('Every input cell/target must be full-dimensional.')
    d = math.lcm(*(p[3] for p in ps))
    v = [tuple(p[j]*(d//p[3]) for j in range(3)) for p in ps]
    found = set()
    for a, b, c in it.combinations(v, 3):
        n = cross(tuple(b[j]-a[j] for j in range(3)), tuple(c[j]-a[j] for j in range(3)))
        if not any(n):
            continue
        rhs = dot(n, a)
        h = primitive(tuple(x*d for x in n)+(rhs,))
        if h in found or tuple(-x for x in h) in found:
            continue
        positive = negative = False
        for x in v:
            residual = dot(n, x)-rhs
            positive |= residual > 0
            negative |= residual < 0
            if positive and negative:
                break
        if positive and negative:
            continue
        if positive:
            h = tuple(-x for x in h)
        found.add(h)
    return sorted(found)


def poly_vertices(hs) -> list[Point]:
    """All vertices of a bounded intersection; empty/lower-dimensional allowed."""
    hs = list(dict.fromkeys(hs))
    out = set()
    for a, b, c in it.combinations(hs, 3):
        bc, ca, ab = cross(b[:3], c[:3]), cross(c[:3], a[:3]), cross(a[:3], b[:3])
        det = dot(a[:3], bc)
        if not det:
            continue
        xyz = tuple(a[3]*bc[j]+b[3]*ca[j]+c[3]*ab[j] for j in range(3))
        p = xyz+(det,)
        if det < 0:
            p = tuple(-x for x in p)
        if all(val(h, p) <= 0 for h in hs):
            out.add(point(p))
    return sorted(out)


def orient(p, q, r, axes):
    i, j = axes
    return ((q[i]*p[3]-p[i]*q[3])*(r[j]*p[3]-p[j]*r[3])
            -(q[j]*p[3]-p[j]*q[3])*(r[i]*p[3]-p[i]*r[3]))


def face_polygon(ps, axes):
    """Exact planar convex hull by monotone chains, in a coordinate projection."""
    ps = sorted(set(ps), key=lambda p: tuple(F(p[j], p[3]) for j in axes))
    if len(ps) < 3:
        return []
    def half(seq):
        out = []
        for p in seq:
            while len(out) > 1 and orient(out[-2], out[-1], p, axes) <= 0:
                out.pop()
            out.append(p)
        return out
    out = half(ps)[:-1]+half(list(reversed(ps)))[:-1]
    return out if has_area(out, axes) else []


def has_area(poly, axes):
    return len(poly) >= 3 and any(orient(poly[0], poly[j], poly[j+1], axes)
                                  for j in range(1, len(poly)-1))


def clip(poly, h, keep_inside=True):
    """Sutherland-Hodgman clipping, with exact homogeneous intersections."""
    sign = 1 if keep_inside else -1
    out = []
    for p, q in zip(poly, poly[1:]+poly[:1]):
        sp, sq = sign*val(h, p), sign*val(h, q)
        if sp <= 0:
            out.append(p)
        if (sp < 0 < sq) or (sq < 0 < sp):
            out.append(point(tuple(sp*q[j]-sq*p[j] for j in range(4))))
    clean = []
    for p in out:
        if not clean or p != clean[-1]:
            clean.append(p)
    if len(clean) > 1 and clean[0] == clean[-1]:
        clean.pop()
    return clean


def subtract(poly, hs, axes):
    """Closures of the positive-area pieces of poly minus a CLOSED polytope.

    Boundary-only pieces are discarded. This is valid because a point missing
    from a finite union of closed polytopes has a relatively open neighborhood.
    A plane coincident with the whole target face does NOT count as outside.
    """
    outside = []
    for h in hs:
        signs = [val(h, p) for p in poly]
        if max(signs) <= 0:
            continue
        if min(signs) >= 0:
            outside.append(poly)
            return outside
        piece = clip(poly, h, False)
        if has_area(piece, axes):
            outside.append(piece)
        poly = clip(poly, h, True)
        if not has_area(poly, axes):
            return outside
    return outside


def boundary_cover(target_hs, target_vs, cells, max_pieces=100000):
    faces, tested, peak = [], 0, 1
    for f, h in enumerate(target_hs):
        drop = max(range(3), key=lambda j: abs(h[j]))
        axes = tuple(j for j in range(3) if j != drop)
        poly = face_polygon([p for p in target_vs if val(h, p) == 0], axes)
        if not poly:
            continue
        tested += 1
        todo = [poly]
        for cell in cells:
            todo = [q for p in todo for q in subtract(p, cell, axes)]
            peak = max(peak, len(todo))
            if len(todo) > max_pieces:
                raise RuntimeError('Polygon-piece limit exceeded; no coverage verdict.')
            if not todo:
                break
        faces.append({'constraint': f, 'remaining_pieces': len(todo)})
        if todo:
            witness = mean_point(todo[0])
            if not all(any(val(g, witness) > 0 for g in cell) for cell in cells):
                raise ArithmeticError('Internal error: invalid uncovered-point witness.')
            return dict(covered=False, faces_checked=tested, peak_pieces=peak,
                        witness=point_record(witness), face=f, faces=faces)
    return dict(covered=True, faces_checked=tested, peak_pieces=peak, faces=faces)


def volume(hs, vs=None):
    """Exact volume from oriented facet triangles (bounded polytopes only)."""
    hs = list(dict.fromkeys(hs))
    vs = poly_vertices(hs) if vs is None else vs
    if not full_dimension(vs):
        return F(0)
    total = F(0)
    for h in hs:
        drop = max(range(3), key=lambda j: abs(h[j]))
        axes = tuple(j for j in range(3) if j != drop)
        face = face_polygon([p for p in vs if val(h, p) == 0], axes)
        if not face:
            continue
        p, q, r = face[:3]
        u = tuple(q[j]*p[3]-p[j]*q[3] for j in range(3))
        v = tuple(r[j]*p[3]-p[j]*r[3] for j in range(3))
        sign = 1 if dot(h[:3], cross(u, v)) > 0 else -1
        for q, r in zip(face[1:-1], face[2:]):
            total += sign*F(dot(p[:3], cross(q[:3], r[:3])), 6*p[3]*q[3]*r[3])
    if total <= 0:
        raise ArithmeticError('Nonpositive volume of a full-dimensional polytope.')
    return total


def volume_cover(target_hs, target_vs, cells):
    """Exact inclusion-exclusion. For four cells, 15 convex intersections."""
    base = volume(target_hs, target_vs)
    covered, records = F(0), []
    for k in range(1, len(cells)+1):
        for ids in it.combinations(range(len(cells)), k):
            hs = list(dict.fromkeys(target_hs+[h for i in ids for h in cells[i]]))
            v = volume(hs)
            covered += (1 if k % 2 else -1)*v
            records.append({'cells': [i+1 for i in ids], 'volume_approx': float(v),
                            'exact_zero_volume': v == 0})
    missing = base-covered
    if missing < 0:
        raise ArithmeticError('Negative exact missing volume.')
    return {'covered': missing == 0, 'target_volume_approx': float(base),
            'missing_volume_approx': float(missing), 'missing_volume_exact_zero': missing == 0,
            'missing_volume_exact': {'numerator_hex': hex(missing.numerator),
                                     'denominator_hex': hex(missing.denominator)},
            'intersections': records}


def diameter_squared(ps):
    best, pair = F(0), None
    for i, j in it.combinations(range(len(ps)), 2):
        p, q = ps[i], ps[j]
        ds = sum((p[t]*q[3]-q[t]*p[3])**2 for t in range(3))
        den = (p[3]*q[3])**2
        if ds*best.denominator > best.numerator*den:
            best, pair = F(ds, den), [i, j]
    return best, pair


def sqrt_upper(x, places=15):
    """Decimal upper bound computed by integer sqrt; no floating-point rounding."""
    scale = 10**places
    n, d = x.numerator*scale*scale, x.denominator
    k = math.isqrt(n//d)
    if k*k*d < n:
        k += 1
    return f'{k//scale}.{k%scale:0{places}d}'


def point_record(p):
    return {'xyz_approx': [float(F(x, p[3])) for x in p[:3]],
            'homogeneous_exact': [str(x) for x in p]}


def pad_planes(hs, eps):
    return [primitive(tuple(x*eps.denominator for x in h[:3])+
                      (h[3]*eps.denominator+eps.numerator*sum(abs(x) for x in h[:3]),))
            for h in hs]


def load_partition(path):
    rows = [r.strip() for r in Path(path).read_text().splitlines() if r.strip()]
    if len(rows) != 5:
        raise ValueError('Expected the five-line partition format.')
    k, stored, n = int(rows[0]), F(rows[1]), int(rows[2])
    groups = json.loads(rows[3])
    ps = [from_xyz(x) for x in json.loads(rows[4], parse_float=F)]
    if len(ps) != n or len(groups) != k or k < 1:
        raise ValueError('Inconsistent point/cell counts.')
    for g in groups:
        if (not isinstance(g, list) or len(g) < 4 or len(set(g)) != len(g)
                or any(type(j) is not int or not 0 <= j < n for j in g)):
            raise ValueError('Invalid cell membership list.')
    return ps, groups, stored


def rhombic_target(a, b, case, places=30, enclosure="outer"):
    """A rigorously containing RATIONAL target for the ideal irrational one."""
    if a+b < 0 or 1+a-b <= 0:
        raise ValueError('Require a+b>=0 and a nonempty x slab.')
    scale = 10**places
    m = math.isqrt(scale*scale//2)+1
    t = F(m, scale)
    assert 2*t*t > 1 and 2*(t-F(1, scale))**2 <= 1
    if enclosure == "inner":
        t -= F(1, scale)
    elif enclosure != "outer":
        raise ValueError("Unknown enclosure direction.")
    hs = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for si, sj in it.product((-1, 1), repeat=2):
            v = [0, 0, 0]
            v[i], v[j] = si, sj
            hs.append(plane(v+[t]))
    hs += [plane([1, 0, 0, F(1, 2)+a]), plane([-1, 0, 0, F(1, 2)-b])]
    hs += ([plane([0, -1, 0, F(1, 2)]), plane([0, 0, -1, F(1, 2)])]
           if case == 'adjacent' else
           [plane([0, -1, 0, F(1, 2)]), plane([0, 1, 0, F(1, 2)])])
    vs = poly_vertices(hs)
    if not full_dimension(vs):
        raise ValueError('Target is empty or lower-dimensional.')
    meta = {'kind': f'rational_{enclosure}_enclosure_of_analytic_target', 'a': str(a), 'b': str(b),
            'case': case, 'sqrt_half_enclosure_exact': str(t), 'vertex_count': len(vs),
            'meaning': 'PASS implies coverage of the ideal target. FAIL for the outer enclosure alone need not disprove ideal-target coverage.'}
    return hs, vs, meta


def common_point(ps, groups, target_hs, cells, specified=None):
    candidates = ([from_xyz(specified)] if specified is not None else
                  [ps[j] for j in sorted(set.intersection(*(set(g) for g in groups)))])
    candidates += [mean_point(ps)] + ps
    for p in candidates:
        if all(val(h, p) <= 0 for hs in [target_hs]+cells for h in hs):
            return p
    return None


def verify(ps, groups, target_hs, target_vs, padding=F(0), bound=None,
           common=None, overlaps=False, max_pieces=100000, full=False):
    original = [hull_planes([ps[j] for j in g]) for g in groups]
    cells = [pad_planes(h, padding) for h in original]
    center = common_point(ps, groups, target_hs, cells, common)
    raw, checked_squared, expanded, records = [], [], [], []
    for k, g in enumerate(groups):
        d2, pair = diameter_squared([ps[j] for j in g])
        raw.append(d2)
        vertices = poly_vertices(cells[k]) if padding else [ps[j] for j in g]
        if not full_dimension(vertices):
            raise ValueError('Degenerate checked cell.')
        e2, _ = diameter_squared(vertices)
        expanded.append(vertices)
        checked_squared.append(e2)
        records.append({'cell': k+1, 'original_facet_count': len(original[k]),
                        'original_diameter_upper': sqrt_upper(d2),
                        'original_diameter_pair': [g[j] for j in pair],
                        'checked_diameter_upper': sqrt_upper(e2),
                        'checked_diameter_squared_exact': str(e2),
                        'checked_vertex_count': len(vertices),
                        'below_diameter_bound': None if bound is None else e2 <= bound*bound})
    boundary = boundary_cover(target_hs, target_vs, cells, max_pieces)
    covered = False if not boundary['covered'] else (True if center is not None else None)
    answer = {'arithmetic': 'Exact integer/rational decisions on decimal strings; floats only for display.',
              'padding': str(padding), 'covers_checked_target': covered,
              'checked_cell_halfspaces_exact': [[[str(x) for x in h] for h in hs] for hs in cells],
              'coverage_method': 'exact_final_boundary_subtraction_plus_verified_common_point',
              'common_point': None if center is None else point_record(center),
              'boundary': boundary, 'cells': records,
              'original_max_diameter_upper': sqrt_upper(max(raw)),
              'checked_max_diameter_upper': sqrt_upper(max(checked_squared)),
              'diameter_bound': None if bound is None else str(bound),
              'diameter_bound_proved': None if bound is None else all(r['below_diameter_bound'] for r in records),
              'original_hulls_contained_in_checked_target': all(val(h, p) <= 0 for g in groups for j in g for p in [ps[j]] for h in target_hs),
              'checked_hulls_contained_in_checked_target': all(val(h, p) <= 0 for vs in expanded for p in vs for h in target_hs)}
    if not boundary['covered']:
        w = tuple(map(int, boundary['witness']['homogeneous_exact']))
        separators = []
        for hs in cells:
            h = next(h for h in hs if val(h, w) > 0)
            separators.append({'plane_exact': [str(t) for t in h],
                               'positive_violation_exact': str(F(val(h, w), w[3]))})
        boundary['witness']['separating_halfspaces'] = separators
    if boundary['covered'] and center is None and not full:
        answer['message'] = 'INCONCLUSIVE: boundary covered, but no common point verified. Supply --common x y z, or use a full 3-D union test.'
    if full:
        check = volume_cover(target_hs, target_vs, cells)
        if covered is not None and covered != check['covered']:
            raise ArithmeticError('Boundary and volume certificates disagree.')
        covered = check['covered']
        answer['covers_checked_target'] = covered
        answer['full_volume_check'] = check
        answer['coverage_method'] = 'exact_full_3D_inclusion_exclusion'
    if overlaps:
        pairs = []
        for i, j in it.combinations(range(len(cells)), 2):
            hs = list(dict.fromkeys(target_hs+cells[i]+cells[j]))
            vs = poly_vertices(hs)
            overlap = full_dimension(vs)
            witness = mean_point(vs) if overlap else None
            if overlap and not all(val(h, witness) < 0 for h in hs):
                raise ArithmeticError('Invalid strict-interior overlap witness.')
            pairs.append({'cells': [i+1, j+1], 'interiors_overlap_inside_target': overlap,
                          'witness': point_record(witness) if witness else None})
        answer['pairwise_overlaps'] = pairs
        answer['clipped_cells_form_partition_up_to_shared_boundaries'] = (None if covered is None else covered and not any(p['interiors_overlap_inside_target'] for p in pairs))
    return answer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', type=Path)
    ap.add_argument('--fixed-count', type=int, help='Target = exact convex hull of this prefix of the input.')
    ap.add_argument('--target', type=Path, help='JSON vertex array or object with vertices; decimal target used exactly.')
    ap.add_argument('--a', type=F, help='Analytic rhombic target: upper x=1/2+a.')
    ap.add_argument('--b', type=F, help='Analytic rhombic target: lower x=-1/2+b.')
    ap.add_argument('--case', choices=['adjacent', 'opposite'], default='adjacent')
    ap.add_argument('--pad', type=F, default=F(0), help='Explicit facet expansion, NOT a tolerance (default 0).')
    ap.add_argument('--diameter-bound', type=F, help='Prove every checked cell diameter <= this exact decimal/rational.')
    ap.add_argument('--common', type=F, nargs=3, help='Optional common point to check, not assumed valid.')
    ap.add_argument('--full-volume', action='store_true', help='Exact 3-D inclusion-exclusion; no common point required (slower).')
    ap.add_argument('--overlaps', action='store_true', help='Exact pairwise interior-intersection tests (slower).')
    ap.add_argument('--max-pieces', type=int, default=100000)
    ap.add_argument('--report', type=Path)
    args = ap.parse_args(argv)
    try:
        begin = time.monotonic()
        if args.pad < 0 or (args.diameter_bound is not None and args.diameter_bound <= 0):
            raise ValueError('Require pad>=0 and a positive diameter bound.')
        ps, groups, stored = load_partition(args.input)
        sources = sum([args.fixed_count is not None, args.target is not None, args.a is not None or args.b is not None])
        if sources != 1:
            raise ValueError('Choose exactly one target: --fixed-count N, --target FILE, or --a A --b B.')
        if args.a is not None or args.b is not None:
            if args.a is None or args.b is None:
                raise ValueError('Both --a and --b are required.')
            th, tv, meta = rhombic_target(args.a, args.b, args.case)
        else:
            if args.fixed_count is not None:
                if not 4 <= args.fixed_count <= len(ps):
                    raise ValueError('Invalid fixed-count.')
                tv = ps[:args.fixed_count]
                meta = {'kind': 'exact_decimal_vertex_hull', 'fixed_count': args.fixed_count}
            else:
                obj = json.loads(args.target.read_text(), parse_float=F)
                tv = [from_xyz(x) for x in (obj['vertices'] if isinstance(obj, dict) else obj)]
                meta = {'kind': 'exact_decimal_vertex_hull', 'file': str(args.target)}
            th = hull_planes(tv)
        result = verify(ps, groups, th, tv, args.pad, args.diameter_bound, args.common, args.overlaps, args.max_pieces, args.full_volume)
        result.update(target=meta, input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                      stored_diameter=str(stored), elapsed_seconds=time.monotonic()-begin,
                      checked_target_halfspaces_exact=[[str(x) for x in h] for h in th])
        if meta['kind'] == 'rational_outer_enclosure_of_analytic_target':
            result['covers_ideal_target'] = True if result['covers_checked_target'] else None
            if result['covers_checked_target'] is False:
                ih, iv, _ = rhombic_target(args.a, args.b, args.case, enclosure='inner')
                cells = [[tuple(map(int, h)) for h in hs] for hs in result['checked_cell_halfspaces_exact']]
                inner = boundary_cover(ih, iv, cells, args.max_pieces)
                if not inner['covered']:
                    result['covers_ideal_target'] = False
                    result['ideal_target_counterexample'] = inner['witness']
        result['elapsed_seconds'] = time.monotonic()-begin
        report = args.report or args.input.with_name(args.input.stem+'_exact_check.json')
        if report.resolve() in [args.input.resolve(), args.target.resolve() if args.target else args.input.resolve()]:
            raise ValueError('The report must not overwrite an input file.')
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        verdict = {True: 'PASS', False: 'FAIL', None: 'INCONCLUSIVE'}[result['covers_checked_target']]
        print(f'Coverage of checked target: {verdict}')
        if 'covers_ideal_target' in result:
            print('Coverage of ideal analytic target:', {True: 'PASS', False: 'FAIL', None: 'INCONCLUSIVE'}[result['covers_ideal_target']])
        print(f'Facet expansion: {args.pad} (not an arithmetic tolerance)')
        print(f'Original maximum diameter <= {result["original_max_diameter_upper"]}')
        print(f'Checked maximum diameter  <= {result["checked_max_diameter_upper"]}')
        print(f'Exact diameter-bound comparison: {result["diameter_bound_proved"]}')
        if not result['boundary']['covered']:
            print('Uncovered witness (approximate display):', result['boundary']['witness']['xyz_approx'])
        if 'message' in result:
            print(result['message'])
        if args.overlaps:
            print('Overlapping cell pairs:', [p['cells'] for p in result['pairwise_overlaps'] if p['interiors_overlap_inside_target']])
        print(f'Report: {report}')
        return 0 if result['covers_checked_target'] is True and result['diameter_bound_proved'] is not False else 1
    except (ValueError, KeyError, OSError, ArithmeticError, RuntimeError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
