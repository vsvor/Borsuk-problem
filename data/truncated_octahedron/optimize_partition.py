#!/usr/bin/env python3
"""Minimize the largest diameter of a fixed-incidence polyhedral cover.

Example:
    python optimize_partition.py r_dod_4_0966.txt --fixed-count 23 \
        --output r_dod_4_optimized.txt

Dependencies: numpy, scipy, shapely. Python >= 3.10.

The first --fixed-count points are the FIXED vertices of the target polyhedron.
Other points move within their original target face/edge, or within the target
if initially interior. Point memberships in the covering hulls are unchanged.

Before optimizing, the program constructs and checks a labeled triangulation
of the target boundary. Every triangle belongs to one input hull. Keeping its
vertices on their original target faces gives a face-preserving continuous map
of the boundary, hence a surjection on each target face. A point common to all
hulls then extends boundary coverage to coverage of the entire target. Folding
of the mapped triangulation is allowed; injectivity is NOT required for a cover.

The optimization is convex: minimize s subject to squared pair distances <= s
and linear target-face constraints. SLSQP uses analytic derivatives. An
independent tangent-relaxation LP provides a numerical global lower bound FOR
THIS FIXED-INCIDENCE MODEL, not for all possible partitions. The JSON report
includes dual weights with a residual-corrected bound. Computations are floating
point, not outward-rounded interval certificates. See optimize_partition_README.md.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, Delaunay, QhullError
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union


class InputError(ValueError):
    """Invalid input, unsupported incidence pattern, or failed validation."""


@dataclass
class Data:
    points: np.ndarray
    groups: list[list[int]]
    stored: float


def read_data(path: Path) -> Data:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(lines) != 5:
        raise InputError("Expected five nonempty lines: k, stored diameter, n, groups, points.")
    k, n = int(lines[0]), int(lines[2])
    stored = float(lines[1])
    groups = ast.literal_eval(lines[3])
    points = np.asarray(ast.literal_eval(lines[4]), dtype=float)
    if k < 1 or n < 4 or points.shape != (n, 3) or not np.all(np.isfinite(points)):
        raise InputError("Expected n finite three-dimensional points and k >= 1.")
    if not np.isfinite(stored) or not isinstance(groups, list) or len(groups) != k:
        raise InputError("Invalid stored value or group count.")
    for group in groups:
        if (not isinstance(group, list) or len(group) < 4
                or any(type(j) is not int or not 0 <= j < n for j in group)
                or len(set(group)) != len(group)):
            raise InputError("Each group must list at least four distinct, valid integer indices.")
    if set.union(*(set(g) for g in groups)) != set(range(n)):
        raise InputError("Every input point must belong to at least one hull.")
    return Data(points, groups, stored)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_data(path: Path, points: np.ndarray, groups: list[list[int]], diameter: float) -> None:
    atomic_text(path, "\n".join((str(len(groups)), repr(float(diameter)), str(len(points)),
                                repr(groups), repr(points.tolist()))) + "\n")


def diameters(points: np.ndarray, groups: list[list[int]]) -> list[dict[str, Any]]:
    result = []
    for k, group in enumerate(groups):
        pairs = np.array(list(itertools.combinations(group, 2)), dtype=int)
        values = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        j = int(values.argmax())
        result.append({"hull": k + 1, "diameter": float(values[j]),
                       "pair": pairs[j].tolist(), "listed_point_count": len(group)})
    return result


def max_diameter(points: np.ndarray, groups: list[list[int]]) -> float:
    return max(item["diameter"] for item in diameters(points, groups))


def target_planes(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = ConvexHull(vertices)
    if len(hull.vertices) != len(vertices):
        raise InputError("The first --fixed-count points must all be distinct extreme vertices.")
    equations: list[np.ndarray] = []
    for row in hull.equations:
        row = row / np.linalg.norm(row[:3])
        if not equations or min(np.linalg.norm(row - e) for e in equations) > 1e-10:
            equations.append(row)
    eq = np.array(equations)
    return eq[:, :3], -eq[:, 3]


@dataclass
class Model:
    data: Data
    fixed: int
    A: np.ndarray
    b: np.ndarray
    reference: np.ndarray
    active: list[list[int]]
    B: np.ndarray                   # p_i = reference_i + B_i @ z
    slices: list[tuple[int, int, int]]
    pairs: np.ndarray
    C: np.ndarray                   # relative motion of each co-hull point pair
    v: np.ndarray                   # reference displacement of each pair
    L: np.ndarray                   # L @ z <= r: nonconstant target inequalities
    r: np.ndarray
    box: float
    common: int
    scale: float
    certificate: dict[str, Any] | None = None

    @property
    def q(self) -> int:
        return self.B.shape[2]

    def positions(self, z: np.ndarray) -> np.ndarray:
        return self.reference + np.einsum("nck,k->nc", self.B, z)

    def pair_values(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        difference = self.v + np.einsum("edk,k->ed", self.C, z)
        values = np.einsum("ed,ed->e", difference, difference)
        gradients = 2 * np.einsum("ed,edk->ek", difference, self.C)
        return values, gradients

    def point_violation(self, z: np.ndarray) -> float:
        return max(0.0, float(np.max(self.positions(z) @ self.A.T - self.b)),
                   float(np.max(np.abs(z)) - self.box) if len(z) else 0.0)

    def coordinates(self, points: np.ndarray) -> np.ndarray:
        z = np.zeros(self.q)
        for j, lo, hi in self.slices:
            z[lo:hi] = np.linalg.lstsq(self.B[j, :, lo:hi],
                                       points[j] - self.reference[j], rcond=None)[0]
        return z


def build_model(data: Data, fixed: int, incidence_tol: float,
                saved_model: dict[str, Any] | None = None) -> Model:
    X = data.points
    if not 4 <= fixed <= len(X):
        raise InputError("--fixed-count must be between 4 and the number of points.")
    A, b = target_planes(X[:fixed])
    scale = max(1.0, float(np.linalg.norm(np.ptp(X[:fixed], axis=0))))
    if saved_model is None:
        reference = X.copy()
        active = [np.flatnonzero(np.abs(A @ x - b) <= incidence_tol).tolist() for x in X]
        for j in range(fixed, len(X)):
            ids = active[j]
            if ids:
                reference[j] += np.linalg.lstsq(A[ids], b[ids] - A[ids] @ X[j], rcond=None)[0]
    else:
        if saved_model["fixed_count"] != fixed or saved_model["groups"] != data.groups:
            raise InputError("Saved model has different fixed-count or memberships.")
        reference = np.asarray(saved_model["reference_points"], dtype=float)
        active = saved_model["active_faces"]
        saved_A, saved_b = np.asarray(saved_model["A"]), np.asarray(saved_model["b"])
        if (reference.shape != X.shape or len(active) != len(X)
                or not np.allclose(reference[:fixed], X[:fixed], atol=1e-13, rtol=0)):
            raise InputError("Saved model does not match the input target vertices/point count.")
        # Plane order is part of the saved incidence model; validate, then reuse it.
        if (saved_A.shape != A.shape or saved_b.shape != b.shape
                or any(np.min(np.linalg.norm(np.column_stack((A, b)) - row, axis=1)) > 1e-10
                       for row in np.column_stack((saved_A, saved_b)))):
            raise InputError("Saved target planes do not match the target polyhedron.")
        A, b = saved_A, saved_b
        for ids in active:
            if any(type(f) is not int or not 0 <= f < len(A) for f in ids):
                raise InputError("Invalid face index in saved model.")
    if float(np.max(reference @ A.T - b)) > 1e-9 * scale:
        raise InputError("Snapping to nearby faces leaves a point outside the target. "
                         "Check the file or --incidence-tol; do not ignore this error.")
    for j, ids in enumerate(active):
        if ids and np.max(np.abs(A[ids] @ reference[j] - b[ids])) > 1e-9 * scale:
            raise InputError(f"Inconsistent target-face equalities at point {j}.")
    owners = [set(k for k, group in enumerate(data.groups) if j in group) for j in range(len(X))]
    common_indices = sorted(set.intersection(*(set(g) for g in data.groups)))
    if not common_indices:
        raise InputError("This coverage proof requires a listed point shared by every hull.")
    common = max(common_indices, key=lambda j: float(np.min(b - A @ reference[j])))
    blocks: list[tuple[int, np.ndarray, int, int]] = []
    q = 0
    for j in range(fixed, len(X)):
        basis = null_space(A[active[j]], rcond=1e-11) if active[j] else np.eye(3)
        if basis.shape[1]:
            blocks.append((j, basis, q, q + basis.shape[1]))
            q += basis.shape[1]
    if q == 0:
        raise InputError("No movable degrees of freedom remain.")
    B = np.zeros((len(X), 3, q))
    for j, basis, lo, hi in blocks:
        B[j, :, lo:hi] = basis
    pairs = np.array(sorted({tuple(sorted(pair)) for group in data.groups
                             for pair in itertools.combinations(group, 2)}), dtype=int)
    a, c = pairs.T
    L_all = np.einsum("fc,nck->nfk", A, B).reshape(-1, q)
    r_all = (b - reference @ A.T).ravel()
    keep = np.linalg.norm(L_all, axis=1) > 1e-11
    # A deliberately loose explicit box makes a residual-corrected dual bound easy.
    # Orthonormal local bases imply |z_j| <= diameter(P) <= ||bbox diagonal||.
    box = 2 * scale + 1
    model = Model(data, fixed, A, b, reference, active, B,
                  [(j, lo, hi) for j, _, lo, hi in blocks], pairs,
                  B[a] - B[c], reference[a] - reference[c],
                  L_all[keep], r_all[keep], box, common, scale)
    model.certificate = make_mesh_certificate(model, owners)
    return model


def face_basis(normal: np.ndarray) -> np.ndarray:
    u = np.eye(3)[int(np.argmin(np.abs(normal)))].copy()
    u -= normal * np.dot(u, normal)
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(normal, u)))


def make_mesh_certificate(model: Model, owners: list[set[int]]) -> dict[str, Any]:
    """Check a genuine reference boundary mesh; never just add patch areas."""
    triangles: list[dict[str, Any]] = []
    face_reports = []
    edge_counts: Counter = Counter()
    eps = 1e-10 * model.scale
    area_eps = 1e-12 * model.scale**2
    for f, normal in enumerate(model.A):
        basis = face_basis(normal)
        ids = np.array([j for j, a in enumerate(model.active) if f in a], dtype=int)
        fixed_ids = np.array([j for j in ids if j < model.fixed], dtype=int)
        face = MultiPoint(model.reference[fixed_ids] @ basis).convex_hull
        if not isinstance(face, Polygon) or face.area <= area_eps:
            raise InputError(f"Invalid target face {f}.")
        xy = model.reference[ids] @ basis
        dt = Delaunay(xy)
        patches = []
        face_edges: Counter = Counter()
        for local in dt.simplices:
            t = ids[local]
            p = model.reference[t]
            signed_twice_area = float(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), normal))
            if abs(signed_twice_area) < area_eps:
                continue  # Qhull can generate roundoff-scale collinear slivers.
            if signed_twice_area < 0:
                t = t[[0, 2, 1]]
            common_owners = set.intersection(*(owners[j] for j in t))
            if not common_owners:
                raise InputError(f"Cannot certify coverage: triangle {t.tolist()} on face {f} "
                                 "has no common hull. Another triangulation or membership "
                                 "pattern is needed; no unverified optimization is performed.")
            triangles.append({"face": f, "vertices": t.tolist(),
                              "owner": min(common_owners) + 1})
            patches.append(Polygon(model.reference[t] @ basis))
            for edge in itertools.combinations(t.tolist(), 2):
                key = tuple(sorted(edge))
                edge_counts[key] += 1
                face_edges[key] += 1
        union = unary_union(patches)
        missing, outside = face.difference(union).area, union.difference(face).area
        overlap = sum(p.area for p in patches) - union.area
        if max(missing, outside, abs(overlap)) > 1e-10 * model.scale**2:
            raise InputError(f"Reference triangles do not tile target face {f}.")
        for (i, j), count in face_edges.items():
            if count not in (1, 2):
                raise InputError("Nonmanifold reference face mesh.")
            if count == 1:
                other_faces = (set(model.active[i]) & set(model.active[j])) - {f}
                if not other_faces:
                    raise InputError("A reference face-boundary segment is not on a target edge.")
        face_reports.append({"face": f, "triangle_count": len(patches),
                             "area": float(face.area), "missing_area": float(missing),
                             "outside_area": float(outside), "overlap_area": float(overlap)})
    bad = [edge for edge, count in edge_counts.items() if count != 2]
    used = {j for triangle in triangles for j in triangle["vertices"]}
    expected = {j for j, active in enumerate(model.active) if active}
    if bad or used != expected or len(used) - len(edge_counts) + len(triangles) != 2:
        raise InputError("Reference boundary is not a conforming closed sphere mesh.")
    return {"valid_numerically": True, "vertices": len(used), "edges": len(edge_counts),
            "triangle_count": len(triangles), "common_point": model.common,
            "triangles": triangles, "faces": face_reports,
            "coverage_argument": "Face-preserving continuous maps of each target face are "
            "surjective; all mapped triangles belong to their owner hulls. The common point "
            "extends the boundary cover to the target interior. Folding is permitted."}


def check_cover(model: Model, points: np.ndarray, tolerance: float = 1e-9) -> dict[str, Any]:
    """Independent 2-D polygon unions on ALL target faces, plus the common point.

    The face sets use the recorded incidence pattern, not proximity-based
    reclassification of the optimized points. Affine residuals are checked first.
    Small nonzero plane residuals are reported, not claimed to be exact zero.
    """
    affine_error = 0.0
    for j, active in enumerate(model.active):
        if active:
            affine_error = max(affine_error,
                               float(np.max(np.abs(model.A[active] @ points[j] - model.b[active]))))
    outside_error = max(0.0, float(np.max(points @ model.A.T - model.b)))
    fixed_error = float(np.max(np.linalg.norm(points[:model.fixed] - model.reference[:model.fixed], axis=1)))
    records = []
    for f, normal in enumerate(model.A):
        basis = face_basis(normal)
        ids = [j for j in range(model.fixed) if f in model.active[j]]
        face = MultiPoint(model.reference[ids] @ basis).convex_hull
        patches = []
        for group in model.data.groups:
            js = [j for j in group if f in model.active[j]]
            if len(js) >= 3:
                polygon = MultiPoint(points[js] @ basis).convex_hull
                if isinstance(polygon, Polygon):
                    patches.append(polygon)
        union = unary_union(patches)
        records.append({"face": f, "area": float(face.area),
                        "missing_area": float(face.difference(union).area),
                        "outside_area": float(union.difference(face).area),
                        "overlap_area": float(sum(p.area for p in patches) - union.area)})
    maximum_missing = max(r["missing_area"] for r in records)
    maximum_outside_area = max(r["outside_area"] for r in records)
    valid = (max(affine_error, outside_error, fixed_error) <= tolerance * model.scale
             and max(maximum_missing, maximum_outside_area) <= tolerance * model.scale**2)
    return {"valid_numerically": bool(valid), "tolerance": tolerance,
            "max_affine_plane_residual": affine_error,
            "max_normalized_outside_violation": outside_error,
            "max_fixed_vertex_displacement": fixed_error,
            "max_missing_face_area": maximum_missing,
            "total_missing_face_area": sum(r["missing_area"] for r in records),
            "common_point": model.common, "faces": records,
            "arithmetic_note": "Floating-point polygon unions and incidence residuals; "
            "not an interval/exact-arithmetic coverage certificate. Areas are not distances."}


def solve_model(model: Model, start: np.ndarray, maxiter: int, ftol: float,
                feasibility_tol: float, print_every: int) -> tuple[np.ndarray, dict[str, Any]]:
    q = model.q
    initial_values, _ = model.pair_values(start)
    if model.point_violation(start) > feasibility_tol * model.scale:
        raise InputError("Starting points do not satisfy the saved incidence model/target.")
    best_z, best_s = start.copy(), float(initial_values.max())
    x0 = np.r_[start, best_s]
    objective_jac = np.r_[np.zeros(q), 1.0]
    linear_jac = np.column_stack((-model.L, np.zeros(len(model.r))))
    iterations = 0

    def inequalities(x: np.ndarray) -> np.ndarray:
        values, _ = model.pair_values(x[:q])
        return x[-1] - values

    def inequalities_jac(x: np.ndarray) -> np.ndarray:
        _, gradients = model.pair_values(x[:q])
        return np.column_stack((-gradients, np.ones(len(gradients))))

    def consider(z: np.ndarray) -> None:
        nonlocal best_z, best_s
        if np.all(np.isfinite(z)) and model.point_violation(z) <= feasibility_tol * model.scale:
            s = float(model.pair_values(z)[0].max())
            if s < best_s:
                best_z, best_s = z.copy(), s

    def callback(x: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1
        consider(x[:q])
        if print_every and iterations % print_every == 0:
            print(f"  iteration {iterations}: best feasible diameter {np.sqrt(best_s):.15f}", flush=True)

    started = time.perf_counter()
    try:
        result = minimize(lambda x: x[-1], x0, jac=lambda x: objective_jac, method="SLSQP",
                          bounds=[(-model.box, model.box)] * q + [(0, None)],
                          constraints=[{"type": "ineq", "fun": inequalities, "jac": inequalities_jac},
                                       {"type": "ineq", "fun": lambda x: model.r - model.L @ x[:q],
                                        "jac": lambda x: linear_jac}],
                          callback=callback, options={"ftol": ftol, "maxiter": maxiter, "disp": False})
        consider(result.x[:q])
        info = {"method": "SLSQP", "success": bool(result.success), "status": int(result.status),
                "message": str(result.message), "iterations": int(result.nit),
                "function_evaluations": int(result.nfev), "ftol": ftol,
                "final_epigraph_squared_diameter": float(result.x[-1])}
    except KeyboardInterrupt:
        info = {"method": "SLSQP", "success": False, "status": "interrupted",
                "message": "Interrupted; retained the best numerically feasible iterate.",
                "iterations": iterations, "ftol": ftol}
    info["elapsed_seconds"] = time.perf_counter() - started
    info["incumbent_squared_diameter"] = best_s
    return best_z, info


def lower_bound(model: Model, z: np.ndarray, upper: float) -> dict[str, Any]:
    """Global tangent LP and a dual bound corrected for stationarity residuals.

    Each squared distance f is convex: f(w) >= f(z)+grad f(z).(w-z).
    Minimize the maximum of these affine lower bounds over the same linear
    feasible set. For any nonnegative normalized distance multipliers lambda
    and target multipliers mu, a valid lower affine function is
      sum lambda*(f-grad.z) - mu.r + (sum lambda*grad + mu.L).w.
    Minimize its residual linear term over the explicit parameter box. Thus
    approximate stationarity need not be treated as exact stationarity.
    """
    values, gradients = model.pair_values(z)
    matrix = np.vstack((np.column_stack((gradients, -np.ones(len(values)))),
                        np.column_stack((model.L, np.zeros(len(model.r))))))
    rhs = np.r_[gradients @ z - values, model.r]
    lp = linprog(np.r_[np.zeros(model.q), 1.0], A_ub=matrix, b_ub=rhs,
                 bounds=[(-model.box, model.box)] * model.q + [(0, None)],
                 method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                         "dual_feasibility_tolerance": 1e-9})
    if not lp.success:
        return {"success": False, "message": lp.message,
                "note": "A feasible cover was found, but no numerical optimality bound is supplied."}
    weights = np.maximum(0, -np.asarray(lp.ineqlin.marginals))
    normalizer = weights[:len(values)].sum()
    if normalizer <= 1e-15:
        return {"success": False, "message": "Degenerate tangent-LP distance multipliers."}
    weights /= normalizer
    lam, mu = weights[:len(values)], weights[len(values):]
    # Extended precision mitigates cancellation but is not interval arithmetic.
    LD = np.longdouble
    ld_values, ld_grad, ld_z = values.astype(LD), gradients.astype(LD), z.astype(LD)
    ld_lam, ld_mu = lam.astype(LD), mu.astype(LD)
    residual = ld_lam @ ld_grad + ld_mu @ model.L.astype(LD)
    constant = ld_lam @ (ld_values - ld_grad @ ld_z) - ld_mu @ model.r.astype(LD)
    corrected = float(constant - LD(model.box) * np.abs(residual).sum())
    # An explicitly disclosed numerical safety allowance, not a formal certificate.
    allowance = 1e-11 * max(1.0, upper**2, abs(corrected))
    guarded = max(0.0, corrected - allowance)
    lower = float(np.sqrt(guarded))
    return {"success": True, "scope": "Fixed memberships, fixed target vertices, and recorded face/edge incidences only.",
            "tangent_lp_squared_lower": float(lp.fun),
            "residual_corrected_squared_lower": corrected,
            "squared_numerical_safety_allowance": allowance,
            "diameter_lower_with_allowance": lower, "diameter_upper": upper,
            "diameter_gap_with_allowance": upper - lower,
            "dual_stationarity_l1_residual": float(np.abs(residual).sum()),
            "box_half_width": model.box,
            "distance_dual_weights": [{"pair": model.pairs[i].tolist(), "weight": float(w)}
                                      for i, w in enumerate(lam) if w > 0],
            "target_dual_weights": mu.tolist(),
            "arithmetic_note": "Numerical dual bound with explicit residual correction and safety allowance; "
            "not outward-rounded interval arithmetic or a proof about all partitions."}


def check_derivatives(model: Model, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.01, model.q)
    _, jac = model.pair_values(z)
    h, error = 1e-6, 0.0
    for j in range(model.q):
        step = np.zeros(model.q)
        step[j] = h
        finite = (model.pair_values(z + step)[0] - model.pair_values(z - step)[0]) / (2 * h)
        error = max(error, float(np.max(np.abs(finite - jac[:, j]))))
    return error


def model_record(model: Model) -> dict[str, Any]:
    return {"fixed_count": model.fixed, "groups": model.data.groups,
            "A": model.A.tolist(), "b": model.b.tolist(),
            "reference_points": model.reference.tolist(), "active_faces": model.active,
            "parameter_box_half_width": model.box,
            "point_degrees_of_freedom": [{"point": j, "dimension": hi - lo}
                                         for j, lo, hi in model.slices],
            "coverage_mesh": model.certificate}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Five-line input partition file.")
    parser.add_argument("--fixed-count", type=int, default=23, help="Number of initial fixed target vertices (default: 23).")
    parser.add_argument("--output", "-o", type=Path, help="Output in the same five-line format.")
    parser.add_argument("--report", type=Path, help="JSON diagnostics and reusable incidence model.")
    parser.add_argument("--model", type=Path, help="Previous report: reuse its incidence model on a restart.")
    parser.add_argument("--incidence-tol", type=float, default=1e-6, help="Initial target-face identification tolerance (default: 1e-6).")
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--ftol", type=float, default=1e-13)
    parser.add_argument("--feasibility-tol", type=float, default=1e-10)
    parser.add_argument("--print-every", type=int, default=10, help="Progress interval; zero disables iteration messages.")
    parser.add_argument("--check-gradients", action="store_true", help="Also check analytic derivatives before optimizing.")
    args = parser.parse_args(argv)
    for name in ("incidence_tol", "ftol", "feasibility_tol"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite.")
    if args.maxiter < 1 or args.print_every < 0:
        parser.error("--maxiter must be positive and --print-every nonnegative.")
    output = args.output or args.input.with_name(args.input.stem + "_optimized.txt")
    report_path = args.report or output.with_suffix(".json")
    paths = [args.input.resolve(), output.resolve(), report_path.resolve()]
    if len(set(paths)) != 3:
        parser.error("Input, output, and report must be different paths; input is never overwritten.")
    if args.model and args.model.resolve() in (output.resolve(), report_path.resolve()):
        parser.error("Use new output/report paths when supplying --model; do not overwrite the model file.")
    try:
        data = read_data(args.input)
        saved_model = json.loads(args.model.read_text(encoding="utf-8"))["model"] if args.model else None
        model = build_model(data, args.fixed_count, args.incidence_tol, saved_model)
        z0 = model.coordinates(data.points)
        initial_adjusted = model.positions(z0)
        initial_check = check_cover(model, initial_adjusted)
        if not initial_check["valid_numerically"]:
            raise InputError("Initial face-compatible coordinates failed the independent coverage check.")
        raw_d = max_diameter(data.points, data.groups)
        adjusted_d = max_diameter(initial_adjusted, data.groups)
        adjustment = float(np.max(np.linalg.norm(initial_adjusted - data.points, axis=1)))
        print(f"Target: {args.fixed_count} fixed vertices, {len(model.A)} faces.")
        print(f"Model: {len(data.points)} points, {len(data.groups)} hulls, {model.q} movable coordinates, "
              f"{len(model.pairs)} distinct pair constraints.")
        print(f"Boundary certificate: {model.certificate['triangle_count']} triangles; "
              f"common point {model.common}.")
        print(f"Stored input value: {data.stored:.15f}")
        print(f"Actual initial maximum diameter: {raw_d:.15f}")
        print(f"Largest initial face/edge correction: {adjustment:.3e}")
        gradient_error = None
        if args.check_gradients:
            gradient_error = check_derivatives(model)
            print(f"Maximum analytic/finite-difference gradient discrepancy: {gradient_error:.3e}")
            if gradient_error > 1e-6:
                raise InputError("Analytic derivative self-check failed.")
        z, solver = solve_model(model, z0, args.maxiter, args.ftol,
                                args.feasibility_tol, args.print_every)
        optimized = model.positions(z)
        coverage = check_cover(model, optimized, max(1e-10, 5 * args.feasibility_tol))
        if not coverage["valid_numerically"]:
            raise InputError("Optimized point set failed the independent coverage check; not saved.")
        final_d = max_diameter(optimized, data.groups)
        bound = lower_bound(model, z, final_d)
        report = {"format_version": 1, "input": str(args.input),
                  "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "output": str(output), "indexing": "zero-based; hull numbers one-based",
                  "stored_input_diameter": data.stored, "actual_initial_diameter": raw_d,
                  "initial_face_compatible_diameter": adjusted_d,
                  "max_initial_face_correction": adjustment,
                  "optimized_max_diameter": final_d, "absolute_improvement": raw_d - final_d,
                  "relative_improvement": 1 - final_d / raw_d,
                  "initial_hulls": diameters(data.points, data.groups),
                  "optimized_hulls": diameters(optimized, data.groups),
                  "solver": solver, "optimality_bound": bound,
                  "coverage_check": coverage, "initial_coverage_after_snapping": initial_check,
                  "gradient_check_max_error": gradient_error,
                  "model": model_record(model)}
        write_data(output, optimized, data.groups, final_d)
        atomic_text(report_path, json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"\nMaximum diameter: {raw_d:.15f} -> {final_d:.15f}")
        for item in report["optimized_hulls"]:
            print(f"  K{item['hull']}: {item['diameter']:.15f}, pair {item['pair']}")
        print(f"Solver: {solver['message']}")
        if bound["success"]:
            print(f"Numerical global diameter interval for this model: "
                  f"[{bound['diameter_lower_with_allowance']:.15f}, {final_d:.15f}]")
            print(f"Numerical bound gap: {bound['diameter_gap_with_allowance']:.3e}")
        else:
            print("WARNING: independent numerical optimality bound unavailable.")
        print(f"Maximum missing target-face area: {coverage['max_missing_face_area']:.3e}")
        print(f"Maximum affine incidence residual: {coverage['max_affine_plane_residual']:.3e}")
        print(f"Saved: {output}\nReport: {report_path}")
        print("Bounds and coverage checks use floating point, not exact/interval arithmetic.")
        return 0
    except (InputError, OSError, ValueError, KeyError, TypeError, QhullError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
