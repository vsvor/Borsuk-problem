#!/usr/bin/env python3
"""Warm-started SLSQP continuation and uncertainty-aware bisection.

A(delta): a=dummy_a, b=delta.  B(delta): a=delta, b=0.
Keep this file next to partitions_slsqp.py and parametric_truncation_search.py.
The bracket concerns the minimum over the supplied reference models, NOT every
possible partition. Numerical primal/dual bounds are not interval certificates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable

if __name__ == '__main__':
    for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_v] = '1'

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull
import partitions_slsqp as ps
import parametric_truncation_search as pt

FORMAT = 'truncation_balance_seed_v1'

class BalanceError(ValueError):
    pass


def json_write(path: Path, value: Any) -> None:
    ps._atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def family_parameters(family: str, delta: float, dummy_a: float) -> tuple[float, float]:
    return (dummy_a, delta) if family == 'dummy' else (delta, 0.)


def family_geometry(family: str, delta: float, case: str, dummy_a: float):
    a, b = family_parameters(family, delta, dummy_a)
    A, rhs, _ = pt.make_halfspaces(a, b, case)
    velocity = np.zeros(len(rhs))
    velocity[13 if family == 'dummy' else 12] = -1. if family == 'dummy' else 1.
    return A, rhs, velocity


def load_source(path: Path, family: str, case: str, dummy_a: float,
                txt_delta: float | None = None):
    """Validate an engine checkpoint, local-optimizer report, or five-line file."""
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'best.json'
    requested_points = None
    requested_groups = None
    if path.suffix.lower() == '.txt':
        requested_points, requested_groups, _ = ps.read_partition(path)
        sibling = path.with_suffix('.json')
        if sibling.exists():
            path = sibling
        else:
            if txt_delta is None:
                raise BalanceError('A bare .txt file needs --seed-delta (or a matching JSON report).')
            a, b = family_parameters(family, txt_delta, dummy_a)
            geom = pt.make_target(a, b, case)
            n = len(geom['vertices'])
            vv = np.asarray(geom['vertices'], float)
            if (len(requested_points) < n or
                    np.max(np.min(np.linalg.norm(requested_points[:n, None]-vv, axis=2), axis=1)) > 2e-9):
                raise BalanceError('The bare text target does not match --seed-delta and the selected family.')
            target = ps.Target.from_vertices(requested_points[:n])
            ref = ps.reference_from_input(target, requested_points, requested_groups,
                                          incidence_tol=1e-8)
            model = ps.Model(target, ref)
            z = model.coordinates(target.normalize(requested_points))
            rec = model.record(z)
            rec['target'] = target.record()
            return rec, None, str(path)
    rec = json.loads(path.read_text(encoding='utf-8'))
    warm = None
    if rec.get('format') == FORMAT:
        if (rec['family'] != family or rec['case'] != case or
                abs(float(rec.get('dummy_a', dummy_a))-dummy_a) > 1e-14):
            raise BalanceError(f'{path}: continuation checkpoint is for a different family/case.')
        warm = (float(rec['delta']), np.asarray(rec['points'], float), rec['groups'])
        rec = rec['source_record']
    if 'reference' not in rec and 'model' in rec:
        old = rec['model']
        reference = np.asarray(old['reference_points'], float)
        n = int(old['fixed_count'])
        groups = old['groups']
        target = ps.Target.from_vertices(reference[:n])
        if requested_points is None:
            txt = path.with_suffix('.txt')
            if not txt.exists():
                raise BalanceError('A local-optimizer report needs its matching .txt beside it.')
            requested_points, requested_groups, _ = ps.read_partition(txt)
        if groups != requested_groups:
            raise BalanceError('The text partition and its model have different memberships.')
        ref = ps.reference_from_input(target, reference, groups, old_model=old)
        model = ps.Model(target, ref)
        z = model.coordinates(target.normalize(requested_points))
        rec = model.record(z)
        rec['target'] = target.record()
    if 'reference' not in rec or 'target' not in rec:
        raise BalanceError(f'{path}: need a partitions_slsqp checkpoint, not just a summary.')
    target = ps.Target.from_vertices(np.asarray(rec['target']['vertices'], float))
    ps.model_from_record(target, rec)  # validates the ORIGINAL coverage reference
    if requested_points is not None:
        compared = warm[1] if warm is not None else np.asarray(rec['points'], float)
        groups = warm[2] if warm is not None else rec['groups']
        if (groups != requested_groups or compared.shape != requested_points.shape or
                not np.allclose(compared, requested_points, rtol=0, atol=2e-10)):
            raise BalanceError('The .txt file and adjacent JSON contain different solutions.')
    return rec, warm, str(path)


@dataclass
class Options:
    maxiter: int = 1200
    ftol: float = 1e-13
    diameter_tol: float = 3e-12
    bound_guard: float = 1e-12   # absolute squared-diameter arithmetic allowance
    upper_guard: float = 5e-13  # absolute diameter arithmetic allowance
    polish_attempts: int = 3


class Branch:
    """One fixed membership/incidence model over a stable target face lattice.

    Target vertices V(delta) are affine. Every reference point is represented by
    fixed convex weights on its permitted target face. Consequently the moving
    anchors X(delta) are feasible; their nullspace coordinates supply ALL motions
    on the original affine face/edge. This is not clipping the optimized points.
    """
    def __init__(self, path: Path, family: str, case: str = 'adjacent',
                 dummy_a: float = .5, txt_delta: float | None = None):
        self.family, self.case, self.dummy_a = family, case, dummy_a
        self.source, warm, self.path = load_source(path, family, case, dummy_a, txt_delta)
        rec = self.source
        old_target = ps.Target.from_vertices(np.asarray(rec['target']['vertices'], float))
        self.V0 = old_target.vertices.copy()
        self.nfixed = len(self.V0)
        self.delta0 = float(self.V0[:, 0].min() + .5 if family == 'dummy'
                            else self.V0[:, 0].max() - .5)
        if abs(self.delta0) < 5e-14:
            self.delta0 = 0.
        if self.delta0 < 0:
            raise BalanceError(f'{path}: inferred negative delta; wrong family or orientation.')
        self.Aall, self.rhs0, self.rhs_velocity = family_geometry(
            family, self.delta0, case, dummy_a)
        a, b = family_parameters(family, self.delta0, dummy_a)
        expected = np.asarray(pt.make_target(a, b, case)['vertices'])
        if (len(expected) != self.nfixed or
                np.max(np.min(np.linalg.norm(self.V0[:, None] - expected, axis=2), axis=1)) > 2e-9):
            raise BalanceError(f'{path}: target does not match the {family} family in this orientation.')
        old_rhs = old_target.b * old_target.scale + old_target.A @ old_target.center
        mapping = []
        for normal, rhs in zip(old_target.A, old_rhs):
            dist = np.linalg.norm(self.Aall - normal, axis=1) + abs(self.rhs0 - rhs)
            i = int(dist.argmin())
            if dist[i] > 2e-8:
                raise BalanceError('Cannot identify a saved target facet in the parameterized family.')
            mapping.append(i)
        self.facets = sorted(set(mapping))
        inverse = {f: i for i, f in enumerate(self.facets)}
        self.active_global = [[mapping[f] for f in ids] for ids in rec['reference']['active_faces']]
        self.active = [[inverse[f] for f in ids] for ids in self.active_global]
        self.groups = rec['groups']
        if self.groups != rec['reference']['groups']:
            raise BalanceError('Checkpoint memberships differ from their reference.')
        self.V_velocity = np.zeros_like(self.V0)
        self.fixed_active = []
        for j, v in enumerate(self.V0):
            ids = np.flatnonzero(abs(self.Aall @ v - self.rhs0) < 1e-9)
            M = self.Aall[ids]
            if np.linalg.matrix_rank(M, tol=1e-10) != 3:
                raise BalanceError('Target prefix includes a nonvertex.')
            w = np.linalg.lstsq(M, self.rhs_velocity[ids], rcond=None)[0]
            if np.max(abs(M @ w - self.rhs_velocity[ids])) > 1e-9:
                raise BalanceError('Source is at a changing target incidence; use a seed away from it.')
            self.fixed_active.append(ids.tolist())
            self.V_velocity[j] = w
        # Convex barycentric anchors, including for points not shared by all cells.
        rp = np.asarray(rec['reference']['points'], float)
        weights = np.zeros((len(rp), self.nfixed))
        weights[:self.nfixed] = np.eye(self.nfixed)
        for j in range(self.nfixed, len(rp)):
            faces = self.active_global[j]
            vertices = np.arange(self.nfixed)
            if faces:
                vertices = np.flatnonzero(np.max(abs(self.V0 @ self.Aall[faces].T - self.rhs0[faces]), axis=1) < 2e-9)
            if not len(vertices):
                raise BalanceError(f'Point {j} has an empty prescribed target face.')
            vv = self.V0[vertices]
            lp = linprog(np.sum((vv-rp[j])**2, axis=1),
                         A_eq=np.vstack((vv.T, np.ones(len(vv)))), b_eq=np.r_[rp[j], 1.],
                         bounds=(0, None), method='highs',
                         options={'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
            if not lp.success:
                raise BalanceError(f'Cannot construct a face-preserving anchor for point {j}.')
            w = np.maximum(lp.x, 0.)
            w /= w.sum()
            if np.linalg.norm(w @ vv - rp[j]) > 2e-8:
                raise BalanceError('Inaccurate barycentric reference reconstruction.')
            weights[j, vertices] = w
        self.anchors0 = weights @ self.V0
        self.anchor_velocity = weights @ self.V_velocity
        self.cache: dict[float, dict] = {}
        self.model_cache: dict[float, ps.Model] = {}
        initial = self.model(self.delta0)
        self.warm_delta = self.delta0
        self.warm_z = initial.coordinates(np.asarray(rec['points'], float))
        if warm is not None:
            wd, wp, wg = warm
            if wg != self.groups:
                raise BalanceError('Continuation memberships changed unexpectedly.')
            wm = self.model(wd)
            z = wm.coordinates(wp)
            if np.max(np.linalg.norm(wm.points(z)-wp, axis=1)) > 1e-8:
                raise BalanceError('Incompatible warm-start coordinates in continuation checkpoint.')
            self.warm_delta, self.warm_z = wd, z

    def validate_delta(self, delta: float, independent: bool = False) -> np.ndarray:
        if not math.isfinite(delta) or delta < 0:
            raise BalanceError('delta must be finite and nonnegative.')
        V = self.V0 + (delta-self.delta0) * self.V_velocity
        rhs = self.rhs0 + (delta-self.delta0) * self.rhs_velocity
        slack = rhs - V @ self.Aall.T
        if np.min(slack) < -2e-10:
            raise BalanceError(f'{self.path}: target topology changes before delta={delta:.12g}. Use a closer seed/bracket.')
        for j, ids in enumerate(self.fixed_active):
            extra = [i for i in range(len(rhs)) if i not in ids and slack[j, i] < 2e-10]
            if extra:
                raise BalanceError(f'Target reaches a new vertex/face incidence at delta={delta:.12g}; not crossed silently.')
        if independent:
            a, b = family_parameters(self.family, delta, self.dummy_a)
            actual = np.asarray(pt.make_target(a, b, self.case)['vertices'])
            if len(actual) != len(V) or np.max(np.min(np.linalg.norm(V[:, None]-actual, axis=2), axis=1)) > 2e-9:
                raise BalanceError('Independent target reconstruction disagrees with continuation.')
        return V

    def model(self, delta: float) -> ps.Model:
        delta = float(delta)
        if delta in self.model_cache:
            return self.model_cache[delta]
        V = self.validate_delta(delta)
        rhs = self.rhs0 + (delta-self.delta0) * self.rhs_velocity
        target = ps.Target(V, np.zeros(3), 1., V, self.Aall[self.facets], rhs[self.facets],
                           float(ConvexHull(V).volume))
        anchors = self.anchors0 + (delta-self.delta0) * self.anchor_velocity
        anchors[:self.nfixed] = V
        reference = dict(points=anchors, groups=self.groups, active=self.active,
                         kind='transported_reference_complex')
        # 2 exceeds the base body's diameter sqrt(2), so this box does not
        # remove any permitted positions in the orthonormal local coordinates.
        model = ps.Model(target, reference, box=2.)
        self.model_cache[delta] = model
        return model

    def evaluate(self, delta: float, options: Options) -> dict:
        delta = float(delta)
        if delta in self.cache:
            return self.cache[delta]
        m = self.model(delta)
        if self.cache:
            near = min(self.cache, key=lambda x: abs(x-delta))
            z0 = self.cache[near]['z'].copy()
        else:
            z0 = self.warm_z.copy()
        result = optimize(m, z0, options)
        result.update(delta=delta, path=self.path, family=self.family)
        self.cache[delta] = result
        return result

    def checkpoint(self, evaluation: dict) -> dict:
        delta, z = evaluation['delta'], evaluation['z']
        model = self.model(delta)
        self.validate_delta(delta, independent=True)
        return dict(format=FORMAT, family=self.family, case=self.case, dummy_a=self.dummy_a,
                    delta=delta, points=model.points(z).tolist(), groups=self.groups,
                    diameter=evaluation['diameter'], lower=evaluation['lower'], upper=evaluation['upper'],
                    fixed_count=self.nfixed, target_vertices=model.target.V.tolist(),
                    geometry=evaluation['geometry'], source_record=self.source,
                    scope='Continuation of the original certified membership/incidence model; not a global search.')


def dual_lower_bound(model: ps.Model, z: np.ndarray, guard: float) -> tuple[float, dict]:
    """Residual-corrected convex dual lower bound, in physical squared units.

    Also minimize the quadratic Lagrangian on its positive eigenspace, then
    bound its remaining gradient over the explicit box. Thus no assumption of
    exact stationarity, exact rank, or positive-definite Hessian is necessary.
    """
    values, gradients = model.pair_values(z)
    fixed = model.fixed_lower_squared
    if not model.q:
        return math.sqrt(max(0., fixed-guard)), {'method': 'fixed'}
    mat = np.vstack((np.column_stack((gradients, -np.ones(len(values)))),
                     np.column_stack((model.L, np.zeros(len(model.r))))))
    rhs = np.r_[gradients @ z - values, model.r]
    lp = linprog(np.r_[np.zeros(model.q), 1.], A_ub=mat, b_ub=rhs,
                 bounds=[(-model.box, model.box)]*model.q + [(0, None)], method='highs',
                 options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    if not lp.success:
        return math.sqrt(max(0., fixed-guard)), {'method': 'fixed_fallback', 'message': lp.message}
    raw = np.maximum(0., -np.asarray(lp.ineqlin.marginals))
    total = raw[:len(values)].sum()
    if total <= 1e-14:
        return math.sqrt(max(0., fixed-guard)), {'method': 'fixed_fallback'}
    raw /= total
    C, v = model.C.astype(np.longdouble), model.v.astype(np.longdouble)
    L, r = model.L.astype(np.longdouble), model.r.astype(np.longdouble)
    candidates = [(fixed, 'fixed')]
    # Near-degenerate LP bases can return tiny unnecessary weights. Removing
    # these never assumes they should be zero: each nonnegative normalized
    # weight vector separately defines a valid convex-dual lower bound.
    for cutoff in (0., 1e-12, 1e-10, 1e-8, 1e-6):
        w = raw.astype(np.longdouble).copy()
        w[w < cutoff] = 0.
        total = w[:len(values)].sum()
        if total <= 1e-14:
            continue
        w /= total
        lam, mu = w[:len(values)], w[len(values):]
        def support(at):
            zz = np.asarray(at, dtype=np.longdouble)
            diff = v + np.einsum('ecq,q->ec', C, zz)
            qvalue = lam @ np.einsum('ec,ec->e', diff, diff) + mu @ (L @ zz-r)
            grad = 2*np.einsum('e,ec,ecq->q', lam, diff, C) + mu @ L
            return float(qvalue-grad@zz-model.box*np.sum(abs(grad)))
        candidates.append((support(z), f'tangent_cutoff_{cutoff:g}'))
        hessian_half = np.einsum('e,eci,ecj->ij', np.asarray(lam,float), model.C, model.C)
        grad = np.asarray(lam, float) @ gradients + np.asarray(mu, float) @ model.L
        eigenvalues, U = np.linalg.eigh(hessian_half)
        keep = eigenvalues > max(1e-12, float(eigenvalues.max(initial=0.))*1e-10)
        if np.any(keep):
            at = z - U[:, keep] @ ((U[:, keep].T @ grad)/(2*eigenvalues[keep]))
            if np.all(np.isfinite(at)) and np.linalg.norm(at) < 1e5:
                candidates.append((support(at), f'quadratic_cutoff_{cutoff:g}'))
    squared, method = max(candidates)
    squared -= guard
    return math.sqrt(max(0., squared)), {'method': 'residual_corrected_convex_dual',
                                        'selected_support': method,
                                        'squared_arithmetic_allowance': guard}


def optimize(model: ps.Model, start: np.ndarray, options: Options) -> dict:
    q = model.q
    best_z = model.repair(start)
    best_s = float(model.pair_values(best_z)[0].max())
    iterations = 0
    codes = []
    def consider(z):
        nonlocal best_z, best_s
        if np.all(np.isfinite(z)):
            zz = model.repair(z)
            value = float(model.pair_values(zz)[0].max())
            if value < best_s:
                best_z, best_s = zz.copy(), value
    if q:
        ids = model.variable_pairs
        objjac = np.r_[np.zeros(q), 1.]
        constraints = []
        if len(ids):
            constraints.append(dict(type='ineq', fun=lambda x: x[-1]-model.pair_values(x[:q], ids)[0],
                                    jac=lambda x: np.column_stack((-model.pair_values(x[:q], ids)[1], np.ones(len(ids))))))
        if len(model.r):
            ljac = np.column_stack((-model.L, np.zeros(len(model.r))))
            constraints.append(dict(type='ineq', fun=lambda x: model.r-model.L@x[:q], jac=lambda x: ljac))
        lower = 0.
        for attempt in range(options.polish_attempts):
            res = minimize(lambda x: x[-1], np.r_[best_z, best_s], jac=lambda x: objjac,
                           method='SLSQP', constraints=constraints,
                           bounds=[(-model.box, model.box)]*q + [(model.fixed_lower_squared, None)],
                           callback=lambda x: consider(x[:q]),
                           options={'maxiter': options.maxiter, 'ftol': max(1e-15, options.ftol*10**(-attempt)), 'disp': False})
            consider(res.x[:q]); iterations += int(res.nit); codes.append(int(res.status))
            lower, bound = dual_lower_bound(model, best_z, options.bound_guard)
            upper = math.sqrt(best_s)+options.upper_guard
            if upper-lower <= options.diameter_tol:
                break
    lower, bound = dual_lower_bound(model, best_z, options.bound_guard)
    diameter = max(v['diameter'] for v in ps.hull_diameters(model.points(best_z), model.groups))
    upper = diameter+options.upper_guard
    if lower > upper:
        raise BalanceError('Numerical lower bound exceeds feasible upper bound; increase --bound-guard.')
    geometry = model.geometry_check(best_z)
    geometry_tol = min(1e-10, max(options.upper_guard, options.bound_guard)/10)
    if (not geometry['valid_numerically'] or
            max(geometry['max_plane_residual'], geometry['max_outside_plane_violation']) > geometry_tol):
        raise BalanceError('Continuation failed the numerical coverage/containment check.')
    return dict(z=best_z, diameter=diameter, lower=lower, upper=upper, gap=upper-lower,
                geometry=geometry, bound=bound, iterations=iterations, solver_statuses=codes)


class Pool:
    def __init__(self, branches: list[Branch], options: Options):
        self.branches, self.options = branches, options
    def evaluate(self, delta: float) -> dict:
        results = [b.evaluate(delta, self.options) for b in self.branches]
        selected = min(range(len(results)), key=lambda i: results[i]['upper'])
        return dict(lower=min(r['lower'] for r in results), upper=min(r['upper'] for r in results),
                    diameter=results[selected]['diameter'], selected=selected, results=results)
    def checkpoint(self, value: dict) -> dict:
        i = value['selected']
        rec = self.branches[i].checkpoint(value['results'][i])
        rec['pool_lower'] = value['lower']
        rec['pool_upper'] = value['upper']
        return rec


def sign_of(ev: dict) -> int:
    return 1 if ev['f_lower'] > 0 else -1 if ev['f_upper'] < 0 else 0


def brief(ev: dict) -> dict:
    return dict(delta=ev['delta'], f_interval=[ev['f_lower'], ev['f_upper']],
                dummy_interval=[ev['dummy']['lower'], ev['dummy']['upper']],
                upper_interval=[ev['upper']['lower'], ev['upper']['upper']],
                selected_dummy=ev['dummy']['selected'], selected_upper=ev['upper']['selected'])


def bracket_root(evaluate: Callable[[float], dict], lo: float, hi: float, tolerance: float,
                 maxiter: int = 80, checkpoint: Callable | None = None) -> dict:
    """Keep opposite *bounded* signs. An undecided midpoint never chooses a side."""
    if not 0 <= lo < hi or tolerance <= 0:
        raise BalanceError('Need 0 <= lo < hi and positive delta tolerance.')
    left, right = evaluate(lo), evaluate(hi)
    if not sign_of(left) or not sign_of(right) or sign_of(left) == sign_of(right):
        return dict(status='not_bracketed', left=left, right=right, iterations=0,
                    message='Endpoints do not have resolved opposite signs. Choose another bracket or better seeds.')
    status = 'max_iterations'
    it = 0
    for it in range(maxiter+1):
        if checkpoint:
            checkpoint(left, right)
        if right['delta']-left['delta'] <= tolerance:
            status = 'converged'; break
        mid = (left['delta']+right['delta'])/2
        if mid in (left['delta'], right['delta']):
            status = 'floating_point_limit'; break
        middle = evaluate(mid)
        sm = sign_of(middle)
        if sm:
            if sm == sign_of(left): left = middle
            else: right = middle
            continue
        # Near equality, use two side probes instead of making up a sign or
        # turning objective error into an unjustified delta accuracy estimate.
        q1 = evaluate((left['delta']+mid)/2)
        q3 = evaluate((mid+right['delta'])/2)
        known = sorted([e for e in (left,q1,q3,right) if sign_of(e)], key=lambda e:e['delta'])
        pairs = [(a,b) for a,b in zip(known,known[1:]) if sign_of(a)!=sign_of(b)]
        a,b = min(pairs, key=lambda ab:ab[1]['delta']-ab[0]['delta'])
        if b['delta']-a['delta'] >= right['delta']-left['delta']:
            status = 'objective_uncertainty_or_equality_plateau'; break
        left,right = a,b
    middle = evaluate((left['delta']+right['delta'])/2)
    return dict(status=status, left=left, right=right, middle=middle, iterations=it,
                message='Bracket is for the supplied model pool and uses floating-point primal/dual bounds.')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dummy', type=Path, action='append', required=True, help='Family A best.json/directory; repeat for more reference models.')
    parser.add_argument('--upper', type=Path, action='append', required=True, help='Family B best.json/directory; repeat for more reference models.')
    parser.add_argument('--case', choices=('adjacent','opposite'), default='adjacent')
    parser.add_argument('--dummy-a', type=float, default=.5)
    parser.add_argument('--seed-delta', type=float, help='Required only for a bare .txt seed without a JSON model.')
    parser.add_argument('--lo', type=float, default=0.)
    parser.add_argument('--hi', type=float, default=.02)
    parser.add_argument('--delta-tol', type=float, default=1e-11, help='Requested full root-bracket width, not an objective tolerance.')
    parser.add_argument('--diameter-tol', type=float, default=3e-12)
    parser.add_argument('--bound-guard', type=float, default=1e-12, help='Squared-diameter numerical allowance; not an interval certificate.')
    parser.add_argument('--upper-guard', type=float, default=5e-13)
    parser.add_argument('--ftol', type=float, default=1e-13)
    parser.add_argument('--maxiter', type=int, default=1200)
    parser.add_argument('--polish-attempts', type=int, default=3)
    parser.add_argument('--max-bisections', type=int, default=80)
    parser.add_argument('--output-dir', type=Path, default=Path('delta_balance'))
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    try:
        for key in ('delta_tol','diameter_tol','bound_guard','upper_guard','ftol'):
            if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0:
                raise BalanceError(f'--{key.replace("_","-")} must be positive and finite.')
        if not math.isfinite(args.dummy_a) or args.dummy_a <= 1/math.sqrt(2)-.5+1e-9:
            raise BalanceError('--dummy-a must make the upper plane strictly redundant.')
        if min(args.maxiter,args.polish_attempts,args.max_bisections)<1:
            raise BalanceError('Iteration counts must be positive.')
        options = Options(args.maxiter,args.ftol,args.diameter_tol,args.bound_guard,args.upper_guard,args.polish_attempts)
        pools = {name: Pool([Branch(p,name,args.case,args.dummy_a,args.seed_delta) for p in paths], options)
                 for name,paths in (('dummy',args.dummy),('upper',args.upper))}
        for pool in pools.values():
            for branch in pool.branches:
                branch.validate_delta(args.lo, independent=True)
                branch.validate_delta(args.hi, independent=True)
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        # The original seed files must never be overwritten by output rotation.
        output_paths = {out/n for n in ('dummy.json','upper.json','dummy.txt','upper.txt','result.json','bracket.json',
                                        'previous_dummy.json','previous_upper.json','previous_dummy.txt','previous_upper.txt','previous_result.json')}
        inputs = {Path(b.path).resolve() for p in pools.values() for b in p.branches}
        if output_paths & inputs:
            raise BalanceError('Use a new output directory when reusing continuation checkpoints.')
        cache: dict[float,dict] = {}
        starttime = time.monotonic()
        def evaluate(delta):
            delta=float(delta)
            if delta not in cache:
                a,b=pools['dummy'].evaluate(delta),pools['upper'].evaluate(delta)
                cache[delta]=dict(delta=delta,dummy=a,upper=b,f_lower=a['lower']-b['upper'],f_upper=a['upper']-b['lower'])
            return cache[delta]
        previous_score = math.inf
        def save_solution(ev, summary):
            nonlocal previous_score
            score=max(ev['dummy']['upper'],ev['upper']['upper'])
            for stem in ('dummy','upper'):
                rec=pools[stem].checkpoint(ev[stem])
                for ext in ('txt','json'):
                    dest=out/f'{stem}.{ext}'
                    if dest.exists(): shutil.copy2(dest,out/f'previous_{stem}.{ext}')
                json_write(out/f'{stem}.json',rec)
                ps.write_partition(out/f'{stem}.txt',np.asarray(rec['points']),rec['groups'],rec['diameter'])
            if (out/'result.json').exists(): shutil.copy2(out/'result.json',out/'previous_result.json')
            json_write(out/'result.json',summary)
            if not args.quiet and score < previous_score:
                print(f'delta={ev["delta"]:.12f}  A={ev["dummy"]["diameter"]:.15f}  B={ev["upper"]["diameter"]:.15f}',flush=True)
            previous_score=min(previous_score,score)
        def checkpoint(left,right):
            summ=dict(status='in_progress', bracket=[left['delta'],right['delta']], left=brief(left),right=brief(right), evaluations=len(cache))
            json_write(out/'bracket.json',summ)
            ev=min((left,right),key=lambda e:max(e['dummy']['upper'],e['upper']['upper']))
            if max(ev['dummy']['upper'],ev['upper']['upper']) < previous_score:
                save_solution(ev,{**summ,'delta':ev['delta'],'scope':'Supplied reference models only; numerical bounds.'})
        result=bracket_root(evaluate,args.lo,args.hi,args.delta_tol,args.max_bisections,checkpoint)
        lo,hi=result['left']['delta'],result['right']['delta']
        summary=dict(status=result['status'],delta_bracket=[lo,hi],bracket_width=hi-lo,
                     left=brief(result['left']),right=brief(result['right']),evaluations=len(cache),
                     elapsed_seconds=time.monotonic()-starttime,case=args.case,dummy_a=args.dummy_a,
                     options=vars(options),source_paths={k:[b.path for b in v.branches] for k,v in pools.items()},
                     scope='Root of the two lower envelopes over the supplied membership/incidence models. Not globally optimal over all partitions.',
                     arithmetic='Floating-point primal/dual bounds with explicit guards; not exact or outward-rounded interval arithmetic.',
                     message=result['message'])
        if 'middle' in result:
            ev=result['middle']
            summary.update(delta=ev['delta'],delta_max_bracket_error=(hi-lo)/2,
                           dummy_diameter=ev['dummy']['diameter'],upper_diameter=ev['upper']['diameter'],
                           max_diameter_upper=max(ev['dummy']['upper'],ev['upper']['upper']),middle=brief(ev),
                           same_rounding_8_decimal_places=format(lo,'.8f')==format(hi,'.8f'),
                           same_rounding_8_significant_digits=format(lo,'.8g')==format(hi,'.8g'))
            save_solution(ev,summary)
        else:
            json_write(out/'result.json',summary)
        json_write(out/'bracket.json',summary)
        if not args.quiet:
            print(f'Status: {summary["status"]}')
            print(f'delta bracket: [{lo:.14f}, {hi:.14f}], width {hi-lo:.3g}')
            if 'delta' in summary:
                print(f'delta estimate: {summary["delta"]:.14f}')
                print(f'Balanced diameter upper estimate: {summary["max_diameter_upper"]:.15f}')
            print(f'Saved: {out}')
        return 0 if result['status']=='converged' else 2
    except KeyboardInterrupt:
        print('Interrupted. Latest bracket and verified solution files are retained.',file=sys.stderr)
        return 130
    except (BalanceError,ps.GeometryError,ValueError,KeyError,OSError,TypeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
