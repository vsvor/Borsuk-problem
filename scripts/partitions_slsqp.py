#!/usr/bin/env python3
"""SLSQP multistart search for small-diameter covers of a convex 3-D polytope.

Python >= 3.10; numpy, scipy, shapely. No PyTorch dependency.
See README_partitions_slsqp.md for the formulation, coverage argument and limits.

Quick start:
  python partitions_slsqp.py r_dod_4_0966.txt --fixed-count 23 \
      --starts 1000 --workers 4 --output-dir search

Only best/previous-best records and a compact restart state are written.
Floating-point geometric checks and LP bounds are NOT interval certificates.
"""
from __future__ import annotations

# Set before importing numerical libraries, also in spawned workers. Override
# deliberately with PARTITION_BLAS_THREADS, not by accidentally inheriting 32.
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = os.environ.get("PARTITION_BLAS_THREADS", "1")

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, asdict, field
import hashlib
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable
import warnings

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, Delaunay, HalfspaceIntersection, QhullError
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

VERSION = 1


class GeometryError(ValueError):
    """Invalid/near-degenerate reference geometry; this start is rejected."""


class EarlyStop(Exception):
    pass


def _json_write(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_partition(path: str | Path) -> tuple[np.ndarray, list[list[int]], float]:
    lines = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if len(lines) != 5:
        raise ValueError("Expected the five-line partition format: k, D, n, groups, points.")
    k, d, n = int(lines[0]), float(lines[1]), int(lines[2])
    groups = ast.literal_eval(lines[3])
    points = np.asarray(ast.literal_eval(lines[4]), dtype=float)
    if points.shape != (n, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Expected finite three-dimensional coordinates.")
    if not isinstance(groups, list) or len(groups) != k or k < 1:
        raise ValueError("Invalid number of hulls.")
    for g in groups:
        if (not isinstance(g, list) or len(g) < 4 or len(set(g)) != len(g)
                or any(type(j) is not int or not 0 <= j < n for j in g)):
            raise ValueError("Hull lists must contain at least four distinct valid integer indices.")
    if set.union(*(set(g) for g in groups)) != set(range(n)):
        raise ValueError("Some points do not belong to any hull.")
    return points, groups, d


def write_partition(path: str | Path, points: np.ndarray,
                    groups: list[list[int]], diameter: float | None = None) -> None:
    actual = max(x["diameter"] for x in hull_diameters(points, groups))
    if diameter is not None and abs(diameter - actual) > 1e-9 * max(1, actual):
        raise ValueError("Refusing to write an objective inconsistent with the coordinates.")
    _atomic_text(Path(path), "\n".join((str(len(groups)), repr(actual), str(len(points)),
                  json.dumps(groups), json.dumps(points.tolist()))) + "\n")


def hull_diameters(points: np.ndarray, groups: list[list[int]]) -> list[dict]:
    answer = []
    for k, g in enumerate(groups):
        pairs = np.array(list(itertools.combinations(g, 2)), dtype=int)
        v = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        j = int(v.argmax())
        answer.append(dict(hull=k + 1, diameter=float(v[j]), pair=pairs[j].tolist()))
    return answer


def _planes(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = ConvexHull(vertices)
    if len(hull.vertices) != len(vertices):
        raise ValueError("All target points must be distinct extreme vertices of a 3-D hull.")
    rows = []
    for row in hull.equations:
        row = row / np.linalg.norm(row[:3])
        if not rows or min(np.linalg.norm(row - r) for r in rows) > 1e-10:
            rows.append(row)
    rows = np.array(rows)
    # Deterministic face order, independent of Qhull's triangulation ordering.
    ids = sorted(range(len(rows)), key=lambda i: tuple(np.round(rows[i], 12)))
    return rows[ids, :3], -rows[ids, 3]


def _basis(normal: np.ndarray) -> np.ndarray:
    u = np.eye(3)[int(np.argmin(abs(normal)))].copy()
    u -= (u @ normal) * normal
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(normal, u)))


def _interior(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    # Normals need not be normalized here. Radius is a Euclidean radius.
    lp = linprog([0., 0., 0., -1.], A_ub=np.column_stack((A, np.linalg.norm(A, axis=1))),
                 b_ub=b, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    if not lp.success or lp.x[-1] <= 1e-10:
        raise GeometryError("Empty or numerically thin cell.")
    return lp.x[:3], float(lp.x[-1])


@dataclass
class Target:
    vertices: np.ndarray                # physical coordinates
    center: np.ndarray
    scale: float                        # target diameter; normalized diameter=1
    V: np.ndarray                       # normalized coordinates
    A: np.ndarray                       # A x <= b, unit normals
    b: np.ndarray
    volume: float

    @classmethod
    def from_vertices(cls, vertices: np.ndarray) -> 'Target':
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.all(np.isfinite(v)):
            raise ValueError("Supply at least four finite 3-D target vertices.")
        # Work near the origin even if the input has a large translation.
        offset = v.mean(axis=0)
        scale = float(np.linalg.norm(v[:, None] - v[None, :], axis=2).max())
        if scale <= 0:
            raise ValueError("Degenerate target.")
        u = (v - offset) / scale
        a, b = _planes(u)
        c, _ = _interior(a, b)
        center = offset + scale * c
        V = (v - center) / scale
        a, b = _planes(V)
        return cls(v.copy(), center, scale, V, a, b, float(ConvexHull(V).volume))

    def normalize(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - self.center) / self.scale

    def physical(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points) * self.scale + self.center

    @property
    def digest(self) -> str:
        return hashlib.sha256(np.asarray(self.vertices, dtype='<f8').tobytes()).hexdigest()

    def record(self) -> dict:
        return dict(vertices=self.vertices.tolist(), sha256=self.digest,
                    center=self.center.tolist(), scale=self.scale)


def _bisectors(sites: np.ndarray, weights: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
    others = [j for j in range(len(sites)) if j != i]
    delta = sites[others] - sites[i]
    norms = np.linalg.norm(delta, axis=1)
    if len(norms) and np.min(norms) < 1e-7:
        raise GeometryError("Coincident or extremely close sites.")
    a = delta / norms[:, None]
    b = (np.sum(sites[others]**2, axis=1) - sites[i] @ sites[i]
         + weights[i] - weights[others]) / (2 * norms)
    return a, b


def build_diagram(target: Target, sites: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Direct clipped (power) Voronoi cells; no distant artificial sites.

    Return normalized reference points, memberships, active target faces and
    enough data to revalidate the reference partition on restart.
    """
    sites = np.asarray(sites, dtype=float)
    k = len(sites)
    weights = np.zeros(k) if weights is None else np.asarray(weights, dtype=float)
    if sites.shape != (k, 3) or weights.shape != (k,) or k < 1:
        raise ValueError("Bad sites/weights shape.")
    if not np.all(np.isfinite(sites)) or not np.all(np.isfinite(weights)):
        raise ValueError("Nonfinite sites or weights.")
    weights = weights - weights.mean()
    points = [v.copy() for v in target.V]
    cell_planes = []
    for i in range(k):
        a, b = _bisectors(sites, weights, i)
        a = np.vstack((target.A, a))
        b = np.r_[target.b, b]
        cell_planes.append((a, b))
        interior = sites[i]
        if np.min(b - a @ interior) <= 1e-8:
            interior, _ = _interior(a, b)
        hs = HalfspaceIntersection(np.column_stack((a, -b)), interior)
        for x in hs.intersections:
            if not np.all(np.isfinite(x)):
                raise GeometryError("Nonfinite halfspace intersection.")
            # Refine a vertex from all tight planes; reject ill-conditioned
            # near-degeneracies rather than silently merging distinct features.
            tight = np.flatnonzero(abs(a @ x - b) <= 2e-9)
            if np.linalg.matrix_rank(a[tight], tol=1e-9) < 3:
                raise GeometryError("Numerically unresolved reference vertex.")
            y = np.linalg.lstsq(a[tight], b[tight], rcond=None)[0]
            if np.max(abs(a[tight] @ y - b[tight])) > 2e-9 or np.max(a @ y - b) > 2e-9:
                raise GeometryError("Inconsistent reference vertex planes.")
            near = np.linalg.norm(np.asarray(points) - y, axis=1)
            if near.min() > 3e-9:
                points.append(y)
    points = np.array(points)
    # Global containment assignment includes shared vertices in BOTH cells.
    groups = [np.flatnonzero(np.max(points @ a.T - b, axis=1) <= 5e-9).tolist()
              for a, b in cell_planes]
    active = [np.flatnonzero(abs(target.A @ p - target.b) <= 5e-9).tolist() for p in points]
    if any(len(g) < 4 for g in groups):
        raise GeometryError("Empty/lower-dimensional cell after vertex reconstruction.")
    ref = dict(points=points, groups=groups, active=active, kind="power_complex",
               sites=sites, weights=weights)
    validate_power_reference(target, ref)
    return ref


def validate_power_reference(target: Target, reference: dict) -> None:
    """Check every cell against its defining halfspaces, not only total volume."""
    p, groups = np.asarray(reference["points"]), reference["groups"]
    sites, weights = np.asarray(reference["sites"]), np.asarray(reference["weights"])
    if len(groups) != len(sites):
        raise GeometryError("Reference cell/site count mismatch.")
    volume = 0.0
    for i, g in enumerate(groups):
        a, b = _bisectors(sites, weights, i)
        a, b = np.vstack((target.A, a)), np.r_[target.b, b]
        if np.max(p[g] @ a.T - b) > 1e-8:
            raise GeometryError("A reference point is outside its power cell.")
        expected = set(np.flatnonzero(np.max(p @ a.T - b, axis=1) <= 5e-9).tolist())
        if set(g) != expected:
            raise GeometryError("Nonconforming shared vertex memberships.")
        c, _ = _interior(a, b)
        exact_vertices = HalfspaceIntersection(np.column_stack((a, -b)), c).intersections
        dist = np.linalg.norm(exact_vertices[:, None] - p[np.array(g)][None, :], axis=2)
        if dist.min(axis=1).max() > 2e-8:
            raise GeometryError("Missing vertex in a reference power cell.")
        volume += ConvexHull(p[g]).volume
    if abs(volume - target.volume) > 2e-8 * target.volume:
        raise GeometryError("Reference cell volume check failed.")
    if set.union(*(set(g) for g in groups)) != set(range(len(p))):
        raise GeometryError("Unused reference vertex.")


def reference_from_input(target: Target, points: np.ndarray, groups: list[list[int]],
                         incidence_tol: float = 1e-6, old_model: dict | None = None) -> dict:
    """Certify the supplied four-hull cover by a boundary mesh and common point.

    Generated diagrams use the more general power-complex certificate. A
    generic arbitrary cover without a suitable common point is NOT silently
    accepted by this input fallback.
    """
    p = target.normalize(points).copy()
    nfixed = len(target.V)
    if p.shape[0] < nfixed or not np.allclose(p[:nfixed], target.V, atol=1e-12, rtol=0):
        raise GeometryError("Input prefix does not match the target vertices.")
    if old_model is not None:
        old_model = old_model.get("model", old_model)
        if old_model["groups"] != groups or old_model["fixed_count"] != nfixed:
            raise GeometryError("The supplied incidence model does not match the input.")
        p = target.normalize(np.asarray(old_model["reference_points"], dtype=float))
        if p.shape != points.shape or not np.allclose(p[:nfixed], target.V, atol=1e-12, rtol=0):
            raise GeometryError("Saved reference does not match the target/point count.")
    tol = incidence_tol / target.scale
    if old_model is not None:
        old_a = np.asarray(old_model["A"], dtype=float)
        old_b = (np.asarray(old_model["b"], dtype=float) - old_a @ target.center) / target.scale
        old_rows, rows = np.column_stack((old_a, old_b)), np.column_stack((target.A, target.b))
        mapping = []
        for row in old_rows:
            distances = np.linalg.norm(rows - row, axis=1)
            if distances.min() > 1e-8:
                raise GeometryError("Saved incidence model has different target planes.")
            mapping.append(int(distances.argmin()))
        active = [sorted(mapping[f] for f in ids) for ids in old_model["active_faces"]]
        if len(active) != len(p):
            raise GeometryError("Saved active-face list has the wrong point count.")
    else:
        active = [np.flatnonzero(abs(target.A @ x - target.b) <= tol).tolist() for x in p]
    for j in range(nfixed, len(p)):
        ids = active[j]
        if ids:
            p[j] += np.linalg.lstsq(target.A[ids], target.b[ids] - target.A[ids] @ p[j], rcond=None)[0]
    if np.max(p @ target.A.T - target.b) > 2e-9:
        raise GeometryError("Face projection of the input leaves a point outside the target.")
    common = set.intersection(*(set(g) for g in groups))
    if not common:
        raise GeometryError("Input fallback requires a listed point common to every hull.")
    owners = [set(k for k, g in enumerate(groups) if j in g) for j in range(len(p))]
    edges = Counter()
    used = set()
    nt = 0
    for f, normal in enumerate(target.A):
        ids = np.array([j for j, a in enumerate(active) if f in a], dtype=int)
        basis = _basis(normal)
        face_ids = [j for j in range(nfixed) if f in active[j]]
        face = MultiPoint(p[face_ids] @ basis).convex_hull
        pieces = []
        for loc in Delaunay(p[ids] @ basis).simplices:
            t = ids[loc].tolist()
            piece = Polygon(p[t] @ basis)
            if piece.area < 1e-14:
                continue
            if not set.intersection(*(owners[j] for j in t)):
                raise GeometryError("Input boundary triangulation has an ownerless triangle; use its saved reference model.")
            pieces.append(piece)
            used.update(t)
            nt += 1
            for e in itertools.combinations(t, 2):
                edges[tuple(sorted(e))] += 1
        union = unary_union(pieces)
        if (face.difference(union).area > 1e-10 or union.difference(face).area > 1e-10
                or abs(sum(t.area for t in pieces) - union.area) > 1e-10):
            raise GeometryError("Input boundary mesh does not tile a target face.")
    expected = {j for j, a in enumerate(active) if a}
    if used != expected or any(v != 2 for v in edges.values()) or len(used) - len(edges) + nt != 2:
        raise GeometryError("Input boundary mesh is not a conforming sphere.")
    return dict(points=p, groups=groups, active=active, kind="boundary_cone",
                sites=None, weights=None,
                common=max(common, key=lambda j: float(np.min(target.b - target.A @ p[j]))))


@dataclass
class Model:
    target: Target
    reference: dict
    X: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    pairs: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)
    L: np.ndarray = field(init=False)
    r: np.ndarray = field(init=False)
    variable_pairs: np.ndarray = field(init=False)
    fixed_lower_squared: float = field(init=False)
    box: float = 1.01

    def __post_init__(self):
        self.X = np.asarray(self.reference["points"], dtype=float).copy()
        fixed = len(self.target.V)
        if not np.allclose(self.X[:fixed], self.target.V, atol=1e-12, rtol=0):
            raise GeometryError("Reference changes a target vertex.")
        blocks = []
        q = 0
        for j in range(fixed, len(self.X)):
            ids = self.reference["active"][j]
            a = self.target.A[ids]
            if len(ids):
                self.X[j] += np.linalg.lstsq(a, self.target.b[ids] - a @ self.X[j], rcond=None)[0]
                if np.max(abs(a @ self.X[j] - self.target.b[ids])) > 1e-9:
                    raise GeometryError("Inconsistent target-face incidence.")
            basis = null_space(a, rcond=1e-10) if len(ids) else np.eye(3)
            blocks.append((j, basis, q, q + basis.shape[1]))
            q += basis.shape[1]
        self.B = np.zeros((len(self.X), 3, q))
        for j, basis, lo, hi in blocks:
            self.B[j, :, lo:hi] = basis
        self.reference = {**self.reference, "points": self.X}
        if np.max(self.X @ self.target.A.T - self.target.b) > 2e-9:
            raise GeometryError("Reference lies outside target.")
        self.pairs = np.array(sorted({tuple(sorted(pair)) for g in self.groups
                                     for pair in itertools.combinations(g, 2)}), dtype=int)
        a, b = self.pairs.T
        self.v, self.C = self.X[a] - self.X[b], self.B[a] - self.B[b]
        if q:
            all_L = np.einsum('fc,ncq->nfq', self.target.A, self.B).reshape(-1, q)
            all_r = (self.target.b - self.X @ self.target.A.T).ravel()
            keep = np.linalg.norm(all_L, axis=1) > 1e-10
            self.L, self.r = all_L[keep], np.maximum(0., all_r[keep])
        else:
            self.L, self.r = np.zeros((0, 0)), np.zeros(0)
        moving = np.linalg.norm(self.C.reshape(len(self.pairs), -1), axis=1) > 1e-12 if q else np.zeros(len(self.pairs), bool)
        self.variable_pairs = np.flatnonzero(moving)
        self.fixed_lower_squared = float(np.max(np.sum(self.v[~moving]**2, axis=1), initial=0.))

    @property
    def q(self) -> int:
        return self.B.shape[-1]

    @property
    def groups(self) -> list[list[int]]:
        return self.reference["groups"]

    def points(self, z: np.ndarray) -> np.ndarray:
        return self.X + np.einsum('ncq,q->nc', self.B, z)

    def coordinates(self, points: np.ndarray) -> np.ndarray:
        return np.einsum('ncq,nc->q', self.B, np.asarray(points) - self.X)

    def pair_values(self, z: np.ndarray, ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        v, c = (self.v, self.C) if ids is None else (self.v[ids], self.C[ids])
        diff = v + np.einsum('ecq,q->ec', c, z)
        return np.einsum('ec,ec->e', diff, diff), 2 * np.einsum('ec,ecq->eq', diff, c)

    def repair(self, z: np.ndarray) -> np.ndarray:
        """Radially repair tiny linear infeasibility toward feasible z=0.

        This is NOT projection onto an approximate penalty surface. The exact
        affine face subspaces are never left. Larger excursions are repaired
        too, but their actual diameter is then recomputed.
        """
        z = np.asarray(z, dtype=float).copy()
        if not np.all(np.isfinite(z)):
            raise ValueError("Nonfinite SLSQP iterate.")
        if not self.q:
            return z
        alpha = min(1., self.box / max(self.box, float(abs(z).max())))
        u = self.L @ z
        bad = u > self.r
        if np.any(bad):
            alpha = min(alpha, float(np.min(self.r[bad] / u[bad])))
        if alpha < 1.:
            z *= max(0., alpha * (1. - 16 * np.finfo(float).eps))
        return z

    def signature(self) -> str:
        """Canonical under cell-label permutations for k<=6.

        Only target incidences, fixed vertex identities and hull memberships
        enter the convex problem. Reference vertex positions do not.
        """
        k = len(self.groups)
        owners = [[i for i, g in enumerate(self.groups) if j in g] for j in range(len(self.X))]
        perms = itertools.permutations(range(k)) if k <= 6 else [tuple(range(k))]
        best = None
        fixed = len(self.target.V)
        for perm in perms:
            desc = tuple(sorted((j if j < fixed else -1,
                                 tuple(self.reference["active"][j]),
                                 tuple(sorted(perm[i] for i in owners[j])))
                                for j in range(len(self.X))))
            if best is None or desc < best:
                best = desc
        return hashlib.sha256(repr(best).encode()).hexdigest()

    def geometry_check(self, z: np.ndarray) -> dict:
        """Independent unions of boundary patches plus affine/containment checks.

        For power references, interior coverage follows from the reference
        complex degree argument (not from boundary coverage alone).
        """
        p = self.points(z)
        affine = 0.
        for j, a in enumerate(self.reference["active"]):
            if a:
                affine = max(affine, float(np.max(abs(self.target.A[a] @ p[j] - self.target.b[a]))))
        outside = max(0., float(np.max(p @ self.target.A.T - self.target.b)))
        missing, extra = 0., 0.
        fixed = len(self.target.V)
        for f, normal in enumerate(self.target.A):
            basis = _basis(normal)
            js = [j for j in range(fixed) if f in self.reference["active"][j]]
            face = MultiPoint(self.X[js] @ basis).convex_hull
            patches = []
            for g in self.groups:
                ids = [j for j in g if f in self.reference["active"][j]]
                if len(ids) >= 3:
                    poly = MultiPoint(p[ids] @ basis).convex_hull
                    if isinstance(poly, Polygon):
                        patches.append(poly)
            union = unary_union(patches)
            missing = max(missing, face.difference(union).area)
            extra = max(extra, union.difference(face).area)
        valid = max(affine, outside) < 2e-9 and max(missing, extra) < 2e-9
        return dict(valid_numerically=bool(valid), max_plane_residual=affine * self.target.scale,
                    max_outside_plane_violation=outside * self.target.scale,
                    max_missing_face_area=missing * self.target.scale**2,
                    max_extra_face_area=extra * self.target.scale**2,
                    coverage_method=self.reference["kind"],
                    note="Boundary areas are not distances; interior coverage uses the reference deformation argument. Floating-point, not interval-certified.")

    def record(self, z: np.ndarray) -> dict:
        p = self.target.physical(self.points(z))
        p[:len(self.target.V)] = self.target.vertices
        r = dict(kind=self.reference["kind"], points=self.target.physical(self.X).tolist(),
                 active_faces=self.reference["active"], groups=self.groups)
        if r["kind"] == "power_complex":
            r["sites"] = self.target.physical(self.reference["sites"]).tolist()
            r["weights"] = (np.asarray(self.reference["weights"]) * self.target.scale**2).tolist()
        else:
            r["common"] = self.reference["common"]
        diams = hull_diameters(p, self.groups)
        return dict(points=p.tolist(), groups=self.groups, diameter=max(t["diameter"] for t in diams),
                    hull_diameters=diams, reference=r, geometry=self.geometry_check(z),
                    variables=self.q, pair_constraints=len(self.pairs), signature=self.signature())


def tangent_lower_bound(model: Model, z: np.ndarray) -> float:
    """Numerical diameter lower bound in NORMALIZED units for THIS topology.

    Dual feasibility is corrected by minimizing its stationarity residual
    over the explicit coordinate box. A numerical safety allowance is also
    subtracted. LP failure returns the elementary fixed-pair bound.
    """
    fixed = model.fixed_lower_squared
    if not model.q:
        return math.sqrt(fixed)
    values, gradients = model.pair_values(z)
    mat = np.vstack((np.column_stack((gradients, -np.ones(len(values)))),
                     np.column_stack((model.L, np.zeros(len(model.r))))))
    rhs = np.r_[gradients @ z - values, model.r]
    lp = linprog(np.r_[np.zeros(model.q), 1.], A_ub=mat, b_ub=rhs,
                 bounds=[(-model.box, model.box)] * model.q + [(0, None)], method="highs",
                 options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    if not lp.success:
        return math.sqrt(fixed)
    weights = np.maximum(0., -np.asarray(lp.ineqlin.marginals))
    total = weights[:len(values)].sum()
    if total <= 1e-14:
        return math.sqrt(fixed)
    weights /= total
    lam, mu = weights[:len(values)].astype(np.longdouble), weights[len(values):].astype(np.longdouble)
    g, v, zz = gradients.astype(np.longdouble), values.astype(np.longdouble), z.astype(np.longdouble)
    residual = lam @ g + mu @ model.L.astype(np.longdouble)
    constant = lam @ (v - g @ zz) - mu @ model.r.astype(np.longdouble)
    squared = float(constant - model.box * abs(residual).sum())
    squared -= 1e-10 * max(1., abs(squared))  # disclosed numerical guard, not interval arithmetic
    return math.sqrt(max(fixed, squared, 0.))


@dataclass
class SearchConfig:
    starts: int = 1000                     # additional random starts; 0=until stopped
    workers: int = 1
    seed: int = 1
    parts: int = 4
    output_dir: str = "partition_search"
    resume: bool = False
    hours: float | None = None             # whole invocation, cooperative deadline
    target_diameter: float | None = None   # stop ALL starts once verified
    run_target: float | None = None        # stop local run when good enough
    seconds_per_start: float | None = None
    coarse_maxiter: int = 120
    coarse_ftol: float = 1e-8
    polish_maxiter: int = 500
    polish_ftol: float = 1e-12
    polish_window: float = .003            # physical length; near-best starts
    improvement_tol: float = 1e-9          # physical length
    bound_start: int = 6
    bound_every: int = 8
    bound_screen: bool = True
    deduplicate: bool = True
    seed_method: str = "mixed"             # sphere, random, packing, mixed
    seed_maxiter: int = 100
    mutation_probability: float = .45
    mutation_min: float = .015             # relative to target diameter
    mutation_max: float = .18
    elite_size: int = 12
    power_weight_scale: float = 0.         # normalized squared-distance units
    reject_above: float | None = None      # OPTIONAL HEURISTIC, physical diameter
    reject_after: int = 25
    quiet: bool = False

    def validate(self) -> None:
        for name in ("workers", "parts", "coarse_maxiter", "polish_maxiter", "seed_maxiter", "elite_size", "bound_every", "reject_after"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.starts < 0 or self.seed < 0 or self.bound_start < 0:
            raise ValueError("starts, seed and bound_start must be nonnegative.")
        for name in ("hours", "target_diameter", "run_target", "seconds_per_start", "coarse_ftol", "polish_ftol", "reject_above"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("polish_window", "improvement_tol", "power_weight_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")
        if not 0 <= self.mutation_probability <= 1 or not 0 < self.mutation_min <= self.mutation_max:
            raise ValueError("Invalid mutation probabilities/scales.")
        if self.seed_method not in ("mixed", "sphere", "packing", "random"):
            raise ValueError("Unknown seed method.")


def solve_model(model: Model, cfg: SearchConfig, start: np.ndarray | None = None,
                incumbent: Callable[[], float] = lambda: math.inf,
                stopped: Callable[[], bool] = lambda: False,
                deadline: float = math.inf) -> tuple[np.ndarray, dict]:
    """Minimize squared diameter with linear geometry and analytic Jacobians.

    incumbent returns PHYSICAL diameter. Return best repaired feasible iterate,
    never a stale loss value or merely SLSQP's final (possibly infeasible) point.
    """
    q, scale = model.q, model.target.scale
    best_z = model.repair(np.zeros(q) if start is None else start)
    best_s = float(model.pair_values(best_z)[0].max())
    lower = math.sqrt(model.fixed_lower_squared)
    iterations = 0
    status, converged = "initial", False
    counts = []
    begin = time.monotonic()
    ids = model.variable_pairs
    obj_jac = np.r_[np.zeros(q), 1.]
    lin_jac = np.column_stack((-model.L, np.zeros(len(model.r))))

    def consider(z):
        nonlocal best_z, best_s
        if np.all(np.isfinite(z)):
            z = model.repair(z)
            s = float(model.pair_values(z)[0].max())
            if s < best_s:
                best_z, best_s = z.copy(), s

    def stop_test(x):
        nonlocal iterations, lower, status
        iterations += 1
        consider(x[:q])
        if stopped():
            status = "cancelled"
            raise EarlyStop
        if time.monotonic() >= deadline:
            status = "time_limit"
            raise EarlyStop
        if cfg.run_target is not None and math.sqrt(best_s) * scale <= cfg.run_target:
            status = "run_target"
            raise EarlyStop
        if cfg.target_diameter is not None and math.sqrt(best_s) * scale <= cfg.target_diameter:
            status = "global_target_candidate"
            raise EarlyStop
        if (cfg.reject_above is not None and iterations >= cfg.reject_after
                and math.sqrt(best_s) * scale > cfg.reject_above):
            status = "heuristic_rejection"
            raise EarlyStop
        if (cfg.bound_screen and np.isfinite(incumbent()) and iterations >= cfg.bound_start
                and (iterations - cfg.bound_start) % cfg.bound_every == 0):
            lower = max(lower, tangent_lower_bound(model, best_z))
            if lower * scale >= incumbent() - cfg.improvement_tol:
                status = "lower_bound_pruned"
                raise EarlyStop

    def cons(x):
        vals, _ = model.pair_values(x[:q], ids)
        return x[-1] - vals

    def cons_jac(x):
        _, grad = model.pair_values(x[:q], ids)
        return np.column_stack((-grad, np.ones(len(ids))))

    if not q or not len(ids):
        return best_z, dict(status="fixed", converged=True, iterations=0,
                            lower_bound=math.sqrt(best_s) * scale, elapsed_seconds=0.)
    for goal, reason in ((cfg.target_diameter, "global_target_candidate"), (cfg.run_target, "run_target")):
        if goal is not None and math.sqrt(best_s) * scale <= goal:
            return best_z, dict(status=reason, converged=False, iterations=0,
                                lower_bound=lower * scale, elapsed_seconds=0.)
    if cfg.bound_screen and lower * scale >= incumbent() - cfg.improvement_tol:
        return best_z, dict(status="fixed_pair_pruned", converged=False, iterations=0,
                            lower_bound=lower * scale, elapsed_seconds=0.)
    constraints = [{"type": "ineq", "fun": cons, "jac": cons_jac}]
    if len(model.r):
        constraints.append({"type": "ineq", "fun": lambda x: model.r - model.L @ x[:q],
                            "jac": lambda x: lin_jac})
    try:
        for stage, maxiter, ftol in (("coarse", cfg.coarse_maxiter, cfg.coarse_ftol),
                                    ("polish", cfg.polish_maxiter, cfg.polish_ftol)):
            if stage == "polish" and math.sqrt(best_s) * scale > incumbent() + cfg.polish_window:
                break
            if stopped() or time.monotonic() >= deadline:
                status = "cancelled" if stopped() else "time_limit"
                break
            result = minimize(lambda x: x[-1], np.r_[best_z, best_s],
                              jac=lambda x: obj_jac, method="SLSQP",
                              bounds=[(-model.box, model.box)] * q + [(model.fixed_lower_squared, None)],
                              constraints=constraints, callback=stop_test,
                              options={"ftol": ftol, "maxiter": maxiter, "disp": False})
            consider(result.x[:q])
            counts.append(dict(stage=stage, status=int(result.status), iterations=int(result.nit)))
            converged = bool(result.success)
            status = "converged" if converged else f"slsqp_status_{result.status}"
            if cfg.bound_screen:
                lower = max(lower, tangent_lower_bound(model, best_z))
                if lower * scale >= incumbent() - cfg.improvement_tol:
                    status = "lower_bound_pruned"
                    break
    except EarlyStop:
        pass
    except KeyboardInterrupt:
        status = "interrupted"
    # Avoid additional work after user cancellation/deadline. The current
    # feasible record may still be accepted and independently checked by parent.
    if status not in ("cancelled", "interrupted", "time_limit", "run_target", "global_target_candidate"):
        lower = max(lower, tangent_lower_bound(model, best_z))
    return best_z, dict(status=status, converged=converged, iterations=iterations,
                        stages=counts, lower_bound=lower * scale,
                        elapsed_seconds=time.monotonic() - begin)


def _shrink_inside(target: Target, sites: np.ndarray, margin: float = 1e-5) -> np.ndarray:
    """Move external sites toward the incenter, retaining their directions."""
    out = np.array(sites, dtype=float).copy()
    for i, s in enumerate(out):
        u = target.A @ s
        pos = u > 0
        alpha = min(1., float(np.min((target.b[pos] - margin) / u[pos], initial=1.)))
        out[i] *= max(0., alpha * (1 - 1e-12))
    return out


def _random_sites(target: Target, rng: np.random.Generator, k: int) -> np.ndarray:
    lo, hi = target.V.min(axis=0), target.V.max(axis=0)
    found = []
    for _ in range(1000):
        p = rng.uniform(lo, hi, size=(max(32, 3 * k), 3))
        p = p[np.max(p @ target.A.T - target.b, axis=1) < -1e-5]
        found.extend(p.tolist())
        if len(found) >= k:
            return np.array(found[:k])
    raise GeometryError("Rejection sampling failed for a very thin target.")


def sphere_directions(k: int, rng: np.random.Generator, maxiter: int = 100,
                      stopped: Callable[[], bool] = lambda: False,
                      deadline: float = math.inf) -> np.ndarray:
    """The original spherical-repulsion start, with exact unit constraints.

    For k=4 use a randomly rotated regular tetrahedron directly: no need to
    repeatedly optimize the same four-point spherical energy minimum.
    """
    if k == 4:
        p = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / math.sqrt(3)
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.where(np.diag(r) < 0, -1., 1.))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        return p @ q
    p = rng.normal(size=(k, 3))
    p /= np.linalg.norm(p, axis=1)[:, None]
    if k == 1:
        return p
    pairs = np.array(list(itertools.combinations(range(k), 2)), dtype=int)
    i, j = pairs.T

    def energy(x):
        p = x.reshape(k, 3)
        diff = p[i] - p[j]
        ss = np.sum(diff * diff, axis=1) + 1e-15
        grad = np.zeros_like(p)
        g = -diff / ss[:, None]**1.5
        np.add.at(grad, i, g)
        np.add.at(grad, j, -g)
        return float(np.sum(ss**-.5)), grad.ravel()

    def equality_jac(x):
        out = np.zeros((k, 3 * k))
        for a in range(k):
            out[a, 3 * a:3 * a + 3] = 2 * x.reshape(k, 3)[a]
        return out

    last = p.ravel().copy()

    def cb(x):
        nonlocal last
        last = x.copy()
        if stopped() or time.monotonic() >= deadline:
            raise EarlyStop

    try:
        res = minimize(energy, last, jac=True, method="SLSQP",
                       constraints=[dict(type="eq", fun=lambda x: np.sum(x.reshape(k, 3)**2, axis=1) - 1.,
                                         jac=equality_jac)], callback=cb,
                       options=dict(maxiter=maxiter, ftol=1e-9, disp=False))
        last = res.x
    except EarlyStop:
        pass
    p = last.reshape(k, 3)
    norm = np.linalg.norm(p, axis=1)
    if not np.all(np.isfinite(p)) or norm.min() < 1e-8:
        raise GeometryError("Invalid spherical seed relaxation.")
    return p / norm[:, None]


def packing_sites(target: Target, rng: np.random.Generator, k: int, maxiter: int,
                  stopped: Callable[[], bool] = lambda: False,
                  deadline: float = math.inf) -> np.ndarray:
    """SLSQP analogue of random_packing(): maximize equal-ball radius.

    This seed-generation problem is nonconvex. It is not the later convex
    diameter subproblem, and is only a source of starting topologies.
    """
    p = _random_sites(target, rng, k)
    if k < 2:
        return np.zeros((k, 3))
    pairs = np.array(list(itertools.combinations(range(k), 2)), dtype=int)
    i, j = pairs.T
    q, f = 3 * k, len(target.b)
    lin = np.zeros((k * f, q + 1))
    for a in range(k):
        lin[a * f:(a + 1) * f, 3 * a:3 * a + 3] = -target.A
    lin[:, -1] = -1

    def radius(p):
        return min(float(np.min(target.b - p @ target.A.T)),
                   float(np.linalg.norm(p[i] - p[j], axis=1).min() / 2))

    best, best_r = p.copy(), max(0., radius(p))

    def constraints(x):
        diff = x[:q].reshape(k, 3)[i] - x[:q].reshape(k, 3)[j]
        return np.sum(diff**2, axis=1) - 4 * x[-1]**2

    def jac(x):
        diff = x[:q].reshape(k, 3)[i] - x[:q].reshape(k, 3)[j]
        out = np.zeros((len(i), q + 1))
        for a in range(3):
            out[np.arange(len(i)), 3 * i + a] = 2 * diff[:, a]
            out[np.arange(len(i)), 3 * j + a] = -2 * diff[:, a]
        out[:, -1] = -8 * x[-1]
        return out

    def cb(x):
        nonlocal best, best_r
        p = _shrink_inside(target, x[:q].reshape(k, 3))
        r = radius(p)
        if r > best_r:
            best, best_r = p, r
        if stopped() or time.monotonic() >= deadline:
            raise EarlyStop

    try:
        res = minimize(lambda x: -x[-1], np.r_[p.ravel(), best_r], method="SLSQP",
                       jac=lambda x: np.r_[np.zeros(q), -1.],
                       bounds=[(None, None)] * q + [(0., float(target.b.min()))],
                       constraints=[dict(type="ineq", fun=constraints, jac=jac),
                                    dict(type="ineq", fun=lambda x: np.tile(target.b, k) + lin @ x,
                                         jac=lambda x: lin)], callback=cb,
                       options=dict(maxiter=maxiter, ftol=1e-8, disp=False))
        cb(res.x)
    except EarlyStop:
        pass
    return best


def generate_sites(target: Target, cfg: SearchConfig, run_id: int, parent: dict | None = None,
                   stopped: Callable[[], bool] = lambda: False,
                   deadline: float = math.inf) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, run_id, 371]))
    k = cfg.parts
    if parent is not None:
        sites = np.asarray(parent["sites"], dtype=float)
        sigma = math.exp(rng.uniform(math.log(cfg.mutation_min), math.log(cfg.mutation_max)))
        sites = _shrink_inside(target, sites + rng.normal(0., sigma, size=(k, 3)))
        weights = np.asarray(parent["weights"], dtype=float).copy()
        if cfg.power_weight_scale:
            weights += rng.normal(0., cfg.power_weight_scale, k)
        else:
            weights[:] = 0.
        method = "mutated_power" if np.any(weights) else "mutated_voronoi"
    else:
        method = cfg.seed_method
        if method == "mixed":
            method = rng.choice(["sphere", "random", "packing"], p=[.55, .25, .20])
        if method == "sphere":
            sites = sphere_directions(k, rng, cfg.seed_maxiter, stopped, deadline) * (.6 * target.b.min())
            # Original algorithm has a common centered sphere. With "mixed"
            # starts, sometimes add radial/translation perturbations to avoid
            # searching only centrally aligned diagrams.
            if cfg.seed_method == "mixed" and rng.random() < .6:
                sites *= rng.uniform(.75, 1.25, size=(k, 1))
                sites += rng.normal(0., .035, size=3)
                sites = _shrink_inside(target, sites)
        elif method == "packing":
            sites = packing_sites(target, rng, k, cfg.seed_maxiter, stopped, deadline)
        else:
            sites = _random_sites(target, rng, k)
        weights = rng.normal(0., cfg.power_weight_scale, k) if cfg.power_weight_scale else np.zeros(k)
    weights -= weights.mean()
    return sites, weights, str(method)


def model_from_record(target: Target, record: dict) -> tuple[Model, np.ndarray]:
    """Read the saved ORIGINAL reference, not incidences inferred at an optimum."""
    if record.get("target", {}).get("sha256") != target.digest:
        raise ValueError("Checkpoint belongs to a different target or target-vertex order.")
    r = record["reference"]
    ref = dict(points=target.normalize(np.asarray(r["points"])), groups=r["groups"],
               active=r["active_faces"], kind=r["kind"])
    if r["kind"] == "power_complex":
        ref.update(sites=target.normalize(np.asarray(r["sites"])),
                   weights=np.asarray(r["weights"]) / target.scale**2)
        validate_power_reference(target, ref)
    else:
        # Rebuild/validate the boundary mesh on the saved reference, not on the
        # folded/optimized coordinates. This also checks common memberships.
        ref = reference_from_input(target, np.asarray(r["points"]), r["groups"], incidence_tol=1e-8 * target.scale)
        if ref["active"] != r["active_faces"]:
            raise ValueError("Saved face-incidence model changed on reload.")
    model = Model(target, ref)
    z = model.coordinates(target.normalize(np.asarray(record["points"])))
    reconstructed = target.physical(model.points(z))
    if np.max(np.linalg.norm(reconstructed - np.asarray(record["points"]), axis=1)) > 1e-8 * target.scale:
        raise ValueError("Saved coordinates are incompatible with their reference model.")
    if not model.geometry_check(z)["valid_numerically"]:
        raise ValueError("Saved incumbent failed geometric checks.")
    return model, z


_WORKER: dict = {}


def _init_worker(target, cfg, best_value, stop_event, cache, cache_lock, deadline):
    global _WORKER
    _WORKER = dict(target=target, cfg=cfg, best=best_value, stop=stop_event,
                   cache=cache, lock=cache_lock, deadline=deadline)
    # Only parent handles Ctrl-C, then asks workers to retain feasible iterates.
    if mp.current_process().name != "MainProcess":
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_run(job: dict) -> dict:
    w = _WORKER
    t, cfg = w["target"], w["cfg"]
    start_time = time.monotonic()
    deadline = min(w["deadline"], start_time + cfg.seconds_per_start
                   if cfg.seconds_per_start is not None else math.inf)
    stopped = w["stop"].is_set
    incumbent = lambda: w["best"].value
    if stopped():
        return dict(status="cancelled", run_id=job["id"])
    signature, claimed, lower = None, False, None
    try:
        sites, weights, source = generate_sites(t, cfg, job["id"], job.get("parent"), stopped, deadline)
        if stopped() or time.monotonic() >= deadline:
            return dict(status="cancelled" if stopped() else "time_limit", run_id=job["id"])
        model = Model(t, build_diagram(t, sites, weights))
        signature = model.signature()
        if cfg.deduplicate:
            with w["lock"]:
                old = w["cache"].get(signature)
                if old == -1.:
                    return dict(status="duplicate_in_progress", run_id=job["id"])
                if old is not None and cfg.bound_screen and old >= incumbent() - cfg.improvement_tol:
                    return dict(status="cached_lower_bound", run_id=job["id"])
                w["cache"][signature] = -1.
                claimed = True
        z, info = solve_model(model, cfg, incumbent=incumbent, stopped=stopped, deadline=deadline)
        lower = info["lower_bound"]
        if info["status"] == "interrupted":
            w["stop"].set()
        diameter = math.sqrt(float(model.pair_values(z)[0].max())) * t.scale
        output = dict(status=info["status"], run_id=job["id"], diameter=diameter,
                      source=source, signature=signature, solver=info,
                      elite=dict(sites=sites.tolist(), weights=weights.tolist(),
                                 diameter=diameter, signature=signature))
        if diameter < incumbent() - cfg.improvement_tol:
            record = model.record(z)
            if not record["geometry"]["valid_numerically"]:
                return dict(status="geometry_rejected", run_id=job["id"],
                            error="Candidate failed post-optimization geometry checks.")
            output["record"] = dict(**record, target=t.record(), solver=info,
                                    run_id=job["id"], source=source)
        return output
    except (GeometryError, QhullError, ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        return dict(status="geometry_rejected", run_id=job["id"], error=str(e)[:500])
    except Exception as e:
        # Do not silently label unexpected programming/runtime errors as bad
        # geometry. Parent counts them separately and reports an example.
        return dict(status="worker_error", run_id=job["id"], error=f"{type(e).__name__}: {e}"[:500])
    finally:
        if claimed:
            with w["lock"]:
                if lower is not None and np.isfinite(lower):
                    w["cache"][signature] = float(lower)
                else:
                    w["cache"].pop(signature, None)


class _Box:
    def __init__(self, value):
        self.value = value


def _run_search_unlocked(vertices: np.ndarray, config: SearchConfig,
               initial_partition: tuple[np.ndarray, list[list[int]]] | None = None,
               initial_model: dict | None = None) -> dict:
    """Public multistart API. With workers>1 call under an __main__ guard.

    starts is the number of ADDITIONAL random attempts in this invocation.
    Input polishing is separate. Resume restores the base seed and starts at
    the next submitted ID; unfinished attempts from an interrupted invocation
    are skipped. Only best.json is authoritative; .txt is a compatible export.
    """
    config.validate()
    config.output_dir = str(config.output_dir)
    t = Target.from_vertices(vertices)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path, prev_path, state_path = out / "best.json", out / "previous_best.json", out / "state.json"
    if not config.resume and any(p.exists() for p in (best_path, prev_path, state_path)):
        raise ValueError("Output directory already has search checkpoints. Use --resume or a new directory.")
    state = json.loads(state_path.read_text()) if config.resume and state_path.exists() else {}
    if state and (state["target_sha256"] != t.digest or state["parts"] != config.parts):
        raise ValueError("Restart state has a different target or number of parts.")
    if state:
        config.seed = int(state["seed"])
    next_id = int(state.get("next_start_id", 0))
    counters = Counter(state.get("counts", {}))
    prior_completed = int(state.get("completed_total", 0))
    completed, submitted = 0, 0
    best = None
    elites = []
    started = time.monotonic()
    deadline = started + config.hours * 3600 if config.hours is not None else math.inf
    last_error = None
    termination = "starts_complete"

    def save_state():
        _json_write(state_path, dict(format_version=VERSION, target_sha256=t.digest,
                                    parts=config.parts, seed=config.seed, next_start_id=next_id,
                                    completed_total=prior_completed + completed, counts=dict(counters),
                                    note="Compact counters, not run logs. Interrupted in-flight IDs are skipped on resume."))

    def accept(record: dict, shared_best=None) -> bool:
        nonlocal best
        # Reconstruct the saved reference independently of the worker and
        # recompute the diameter from coordinates before any file rotation.
        if best and record["diameter"] >= best["diameter"] - config.improvement_tol:
            return False
        m, z = model_from_record(t, record)
        verified = m.record(z)
        if not verified["geometry"]["valid_numerically"]:
            raise GeometryError("Parent rejected a candidate's geometry.")
        record.update(verified)
        if best and record["diameter"] >= best["diameter"] - config.improvement_tol:
            return False
        previous = best
        record.update(format_version=VERSION, target=t.record(), search_seed=config.seed,
                      config=asdict(config), previous_best_diameter=previous["diameter"] if previous else None)
        if previous is not None:
            _json_write(prev_path, previous)
            write_partition(out / "previous_best.txt", np.array(previous["points"]), previous["groups"])
        # Write authoritative JSON first. On resume the redundant TXT is
        # regenerated, so a crash between these replacements is recoverable.
        _json_write(best_path, record)
        write_partition(out / "best.txt", np.array(record["points"]), record["groups"])
        best = record
        if shared_best is not None:
            shared_best.value = best["diameter"]
        if not config.quiet:
            prev = "none" if previous is None else f"{previous['diameter']:.12f}"
            print(f"best={best['diameter']:.12f} previous={prev} "
                  f"start={best['run_id']} points={len(best['points'])} source={best['source']}", flush=True)
        return True

    def add_elite(item: dict | None):
        nonlocal elites
        if item is None or not np.isfinite(item["diameter"]):
            return
        old = next((e for e in elites if e["signature"] == item["signature"]), None)
        if old is not None and old["diameter"] <= item["diameter"]:
            return
        elites = [e for e in elites if e["signature"] != item["signature"]]
        elites.append(item)
        elites.sort(key=lambda e: e["diameter"])
        elites = elites[:config.elite_size]

    if config.resume and best_path.exists():
        best = json.loads(best_path.read_text())
        m, z = model_from_record(t, best)
        if len(m.groups) != config.parts:
            raise ValueError("Checkpoint has a different number of parts.")
        actual = m.record(z)
        if abs(best["diameter"] - actual["diameter"]) > 1e-8 * t.scale:
            raise ValueError("Checkpoint objective disagrees with saved coordinates.")
        best.update(actual)
        write_partition(out / "best.txt", np.array(best["points"]), best["groups"])
        if prev_path.exists():
            prev = json.loads(prev_path.read_text())
            write_partition(out / "previous_best.txt", np.asarray(prev["points"]), prev["groups"])
    elif initial_partition is not None:
        try:
            p, groups = initial_partition
            if len(groups) != config.parts:
                raise ValueError("Initial partition does not match --parts. Use --no-initial to use only its target.")
            ref = reference_from_input(t, p, groups, old_model=initial_model)
            m = Model(t, ref)
            z0 = m.coordinates(t.normalize(p))
            local_deadline = min(deadline, time.monotonic() + config.seconds_per_start
                                 if config.seconds_per_start else math.inf)
            z, info = solve_model(m, config, z0, deadline=local_deadline)
            accept(dict(**m.record(z), solver=info, target=t.record(), run_id=-1, source="input"))
            if info["status"] == "interrupted":
                termination = "interrupted"
        except GeometryError as e:
            last_error = f"Initial cover was not certified: {e}"
            counters["initial_not_certified"] += 1
            if not config.quiet:
                print(f"Warning: {last_error} New reference partitions will still be searched.", file=sys.stderr)

    if best:
        r = best["reference"]
        if r["kind"] == "power_complex":
            sites = t.normalize(np.asarray(r["sites"]))
            weights = np.asarray(r["weights"]) / t.scale**2
        else:
            pp = t.normalize(np.asarray(best["points"]))
            sites = np.array([pp[g].mean(axis=0) for g in best["groups"]])
            weights = np.zeros(config.parts)
        add_elite(dict(sites=sites.tolist(), weights=weights.tolist(),
                       diameter=best["diameter"], signature=best["signature"]))

    def target_met():
        return (best is not None and config.target_diameter is not None
                and best["diameter"] <= config.target_diameter)

    def make_job() -> dict:
        nonlocal next_id, submitted
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, next_id, 991]))
        parent = None
        if elites and rng.random() < config.mutation_probability:
            # Bias toward good elites, but retain genuinely independent starts.
            probabilities = 1. / (1. + np.arange(len(elites)))
            probabilities /= probabilities.sum()
            parent = elites[int(rng.choice(len(elites), p=probabilities))]
        job = dict(id=next_id, parent=parent)
        next_id += 1
        submitted += 1
        return job

    def handle(result, shared_best):
        nonlocal completed, last_error
        completed += 1
        counters[result["status"]] += 1
        if "error" in result:
            last_error = result["error"]
        if "record" in result:
            accept(result["record"], shared_best)
        add_elite(result.get("elite"))
        save_state()

    def more_starts():
        return not config.starts or submitted < config.starts

    save_state()
    if termination == "interrupted":
        pass
    elif target_met():
        termination = "target_reached"
    elif config.workers == 1:
        import threading
        shared_best = _Box(best["diameter"] if best else math.inf)
        stop = threading.Event()
        _init_worker(t, config, shared_best, stop, {}, threading.RLock(), deadline)
        try:
            while more_starts() and not stop.is_set():
                if time.monotonic() >= deadline:
                    termination = "time_limit"
                    break
                job = make_job()
                save_state()
                handle(_worker_run(job), shared_best)
                if stop.is_set():
                    termination = "interrupted"
                    break
                if target_met():
                    termination = "target_reached"
                    break
        except KeyboardInterrupt:
            stop.set()
            termination = "interrupted"
    else:
        ctx = mp.get_context("spawn")
        shared_best = ctx.Value('d', best["diameter"] if best else math.inf)
        stop = ctx.Event()
        with ctx.Manager() as manager:
            cache, cache_lock = manager.dict(), manager.RLock()
            with ProcessPoolExecutor(max_workers=config.workers, mp_context=ctx,
                                     initializer=_init_worker,
                                     initargs=(t, config, shared_best, stop, cache, cache_lock, deadline)) as pool:
                pending = {}
                while pending or (more_starts() and not stop.is_set()):
                    try:
                        if time.monotonic() >= deadline:
                            termination = "time_limit"
                            stop.set()
                        if target_met():
                            termination = "target_reached"
                            stop.set()
                        while len(pending) < config.workers and more_starts() and not stop.is_set():
                            job = make_job()
                            pending[pool.submit(_worker_run, job)] = job["id"]
                            save_state()
                        if not pending:
                            break
                        done, _ = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
                        for fut in done:
                            run_id = pending.pop(fut)
                            try:
                                result = fut.result()
                            except Exception as e:
                                result = dict(status="worker_error", run_id=run_id,
                                              error=f"{type(e).__name__}: {e}"[:500])
                            handle(result, shared_best)
                    except KeyboardInterrupt:
                        termination = "interrupted"
                        stop.set()
                        # Drain in-flight cooperative tasks: no new ones are
                        # submitted, their best feasible iterates can be saved.
    save_state()
    summary = dict(termination=termination, completed_this_invocation=completed,
                   submitted_this_invocation=submitted, elapsed_seconds=time.monotonic() - started,
                   counts=dict(counters), best_diameter=best["diameter"] if best else None,
                   best_path=str(best_path) if best else None, last_error=last_error)
    if not config.quiet:
        dd = "none" if best is None else f"{best['diameter']:.12f}"
        print(f"Finished: {termination}; completed={completed}; best={dd}; counts={dict(counters)}", flush=True)
        if counters["worker_error"] or best is None:
            if last_error:
                print(f"Last diagnostic: {last_error}", file=sys.stderr)
    return summary


@contextmanager
def _directory_lock(directory: Path):
    """Prevent two search controllers from overwriting the same checkpoints."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".search.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                raise ValueError("Another search is using this output directory.") from e
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                raise ValueError("Another search is using this output directory.") from e
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_search(vertices: np.ndarray, config: SearchConfig,
               initial_partition: tuple[np.ndarray, list[list[int]]] | None = None,
               initial_model: dict | None = None) -> dict:
    """Public API. Run under an __main__ guard when workers > 1.

    See _run_search_unlocked for implementation. Keep the original reference
    geometry in best.json when restarting; best.txt alone loses that model.
    """
    with _directory_lock(Path(config.output_dir)):
        return _run_search_unlocked(vertices, config, initial_partition, initial_model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path, help="Five-line partition file. Its first --fixed-count points define the target.")
    parser.add_argument("--vertices", type=Path, help="Alternative: JSON array of target vertices or {\"vertices\": [...]}. No initial cover.")
    parser.add_argument("--fixed-count", type=int, default=23)
    parser.add_argument("--parts", type=int, help="Default: input's group count, or 4 with --vertices.")
    parser.add_argument("--initial-model", type=Path, help="Previous optimize_partition.py JSON: reuse its reference when polishing the input.")
    parser.add_argument("--no-initial", action="store_true", help="Use input only to define target; do not polish its supplied cover.")
    parser.add_argument("--starts", type=int, default=1000, help="Additional random starts in this invocation; 0 means until interrupted/deadline.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, help="Random seed, default 1. Resume restores original seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("partition_search"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hours", type=float, help="Cooperative whole-search wall-time budget.")
    parser.add_argument("--target-diameter", "--target", type=float, help="Stop entire search after verifying a cover with D <= this value.")
    parser.add_argument("--run-target", type=float, help="Stop each local SLSQP solve at D <= this value; continue multistart.")
    parser.add_argument("--seconds-per-start", type=float, help="Cooperative budget for seed generation + geometry + local solve.")
    parser.add_argument("--coarse-maxiter", type=int, default=120)
    parser.add_argument("--coarse-ftol", type=float, default=1e-8)
    parser.add_argument("--polish-maxiter", type=int, default=500)
    parser.add_argument("--polish-ftol", type=float, default=1e-12)
    parser.add_argument("--polish-window", type=float, default=.003)
    parser.add_argument("--improvement-tol", type=float, default=1e-9)
    parser.add_argument("--bound-start", type=int, default=6)
    parser.add_argument("--bound-every", type=int, default=8)
    parser.add_argument("--no-bound-screen", action="store_true", help="Disable fixed-pair/LP early pruning (final LP diagnostics still computed).")
    parser.add_argument("--no-deduplicate", action="store_true", help="Disable shared topology-cache skips.")
    parser.add_argument("--seed-method", choices=["mixed", "sphere", "random", "packing"], default="mixed")
    parser.add_argument("--seed-maxiter", type=int, default=100)
    parser.add_argument("--mutation-probability", type=float, default=.45)
    parser.add_argument("--mutation-min", type=float, default=.015)
    parser.add_argument("--mutation-max", type=float, default=.18)
    parser.add_argument("--elite-size", type=int, default=12)
    parser.add_argument("--power-weight-scale", type=float, default=0., help="Optional broadened search: random power weights in normalized squared-distance units; 0=ordinary Voronoi.")
    parser.add_argument("--reject-above", type=float, help="OPTIONAL HEURISTIC: reject if best D still exceeds this after --reject-after iterations. Can miss good topologies.")
    parser.add_argument("--reject-after", type=int, default=25)
    parser.add_argument("--quiet", action="store_true", help="Suppress record and final-summary messages; errors still cause nonzero exit.")
    args = parser.parse_args(argv)
    if bool(args.input) == bool(args.vertices):
        parser.error("Supply exactly one input partition or --vertices JSON.")
    if args.initial_model and (not args.input or args.no_initial):
        parser.error("--initial-model requires an input partition and initial polishing.")
    try:
        if args.input and not args.resume and args.input.resolve() in {
                (args.output_dir / name).resolve() for name in ("best.txt", "previous_best.txt")}:
            raise ValueError("Input would be overwritten by a checkpoint; use a new output directory or a valid --resume checkpoint.")
        initial = None
        if args.input:
            p, groups, stored = read_partition(args.input)
            if not 4 <= args.fixed_count <= len(p):
                raise ValueError("Invalid --fixed-count.")
            vertices = p[:args.fixed_count]
            parts = args.parts if args.parts is not None else len(groups)
            if not args.no_initial:
                initial = (p, groups)
        else:
            obj = json.loads(args.vertices.read_text())
            vertices = np.array(obj["vertices"] if isinstance(obj, dict) else obj, dtype=float)
            parts = args.parts if args.parts is not None else 4
        values = vars(args).copy()
        for name in ("input", "vertices", "fixed_count", "initial_model", "no_initial", "no_bound_screen", "no_deduplicate"):
            values.pop(name)
        values.update(parts=parts, seed=args.seed if args.seed is not None else 1,
                      output_dir=str(args.output_dir), bound_screen=not args.no_bound_screen,
                      deduplicate=not args.no_deduplicate)
        if args.resume and args.seed is not None and (args.output_dir / "state.json").exists():
            old_seed = json.loads((args.output_dir / "state.json").read_text())["seed"]
            if args.seed != old_seed:
                raise ValueError("Changing --seed during resume is not allowed; use a new directory.")
        cfg = SearchConfig(**values)
        old_model = json.loads(args.initial_model.read_text()) if args.initial_model else None
        summary = run_search(vertices, cfg, initial, old_model)
        return 0 if summary["best_diameter"] is not None and not summary["counts"].get("worker_error", 0) else 2
    except (ValueError, OSError, GeometryError, QhullError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; existing checkpoints are preserved.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
